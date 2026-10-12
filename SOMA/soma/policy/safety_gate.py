"""Confidence / interruptibility gate: decides whether SOMA speaks at all.

``NO_SUGGESTION`` must often win. The gate stays silent when:

* the foreground app is in ``quiet_apps``,
* there are no (eligible) candidates,
* a suggestion was shown less than ``cooldown_s`` ago,
* the best non-abstain probability is below ``relevance_threshold``,
* it does not beat ``NO_SUGGESTION`` by ``margin_over_abstain``,
* the candidate's key was dismissed or already shown within ``duplicate_window_s``,
* the candidate references a blocked source / unauthorized file,
* its UI target no longer exists on the latest observation.

Every decision returns a ``Suggestion`` with the full distribution, the screen
version and a human-readable ``reason``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Container, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from soma.config import PolicyConfig
from soma.policy.grounding import verify_target
from soma.types import ActionType, Candidate, RetrievedMemory, ScreenObservation, Suggestion

NO_SUGGESTION_LABEL = ActionType.NO_SUGGESTION.value


def _to_vector(scores: Any) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().float().cpu().numpy()
    else:
        arr = np.asarray(scores, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"expected scores shaped [A+1] or [1, A+1], got {arr.shape}")
    return arr


def to_probs(scores: Any, is_logits: bool | None = None) -> np.ndarray:
    """Convert logits or probabilities to a probability vector.

    With ``is_logits=None`` the input is treated as probabilities when it is
    non-negative and sums to ~1, otherwise as logits (masked ``-inf`` is fine).
    Non-finite probabilities become 0.
    """
    arr = _to_vector(scores)
    if is_logits is None:
        finite = np.isfinite(arr)
        is_logits = not (finite.all() and (arr >= 0).all() and (arr <= 1).all()
                         and abs(arr.sum() - 1.0) < 1e-3)
    if is_logits:
        arr = np.where(np.isnan(arr), -np.inf, arr)
        m = np.max(arr)
        if not np.isfinite(m):
            out = np.zeros_like(arr)
            out[0] = 1.0  # nothing valid: abstain
            return out
        e = np.exp(arr - m)
        return e / e.sum()
    arr = np.where(np.isfinite(arr), np.clip(arr, 0.0, None), 0.0)
    s = arr.sum()
    if s <= 0:
        out = np.zeros_like(arr)
        out[0] = 1.0
        return out
    return arr / s


def distribution_labels(candidates: Sequence[Candidate]) -> list[str]:
    """Unique labels: ``NO_SUGGESTION`` then action names (``#2`` for repeats)."""
    labels = [NO_SUGGESTION_LABEL]
    counts: dict[str, int] = {}
    for c in candidates:
        name = c.action.value
        counts[name] = counts.get(name, 0) + 1
        labels.append(name if counts[name] == 1 else f"{name}#{counts[name]}")
    return labels


def _norm_app(name: str) -> str:
    name = (name or "").strip().lower()
    return name[:-4] if name.endswith(".exe") else name


class SafetyGate:
    """Stateful interruption policy around the scorer's output.

    Thread-safe; ``clock`` is injectable for tests (seconds, monotonic).
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        clock: Callable[[], float] = time.monotonic,
        iou_threshold: float = 0.5,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.iou_threshold = iou_threshold
        self._lock = threading.Lock()
        self._last_shown_at: float | None = None
        self._shown: dict[str, float] = {}
        self._dismissed: dict[str, float] = {}

    # ------------------------------------------------------------ history

    def record_shown(self, key: str) -> None:
        """Mark a candidate key as shown now (starts cooldown + duplicate window)."""
        with self._lock:
            now = self.clock()
            self._shown[key] = now
            self._last_shown_at = now

    def record_dismissed(self, key: str) -> None:
        """Mark a candidate key as dismissed; it is suppressed for the duplicate window."""
        with self._lock:
            self._dismissed[key] = self.clock()

    def reset(self) -> None:
        with self._lock:
            self._last_shown_at = None
            self._shown.clear()
            self._dismissed.clear()

    def in_cooldown(self) -> bool:
        with self._lock:
            return self._in_cooldown(self.clock())

    def is_duplicate(self, key: str) -> bool:
        with self._lock:
            return self._duplicate_reason(key, self.clock()) is not None

    def _in_cooldown(self, now: float) -> bool:
        return self._last_shown_at is not None and now - self._last_shown_at < self.cfg.cooldown_s

    def _duplicate_reason(self, key: str, now: float) -> str | None:
        window = self.cfg.duplicate_window_s
        t = self._dismissed.get(key)
        if t is not None and now - t < window:
            return "dismissed recently"
        t = self._shown.get(key)
        if t is not None and now - t < window:
            return "already shown recently"
        return None

    def _prune(self, now: float) -> None:
        window = self.cfg.duplicate_window_s
        for store in (self._shown, self._dismissed):
            for k in [k for k, t in store.items() if now - t >= window]:
                del store[k]

    # ------------------------------------------------------------ blocking

    @staticmethod
    def blocked_mask(
        candidates: Sequence[Candidate],
        authorized_file: Callable[[str], bool] | None = None,
        blocked_sources: Container[str] = (),
        memories_by_id: Mapping[str, RetrievedMemory] | None = None,
    ) -> list[bool]:
        """True for candidates that must never be shown.

        A candidate is blocked if it references a file that ``authorized_file``
        does not allow (no predicate means deny), a memory id or source URI in
        ``blocked_sources``, a memory that is not user-approved, or (when
        ``memories_by_id`` is given) a memory that was not retrieved.
        """
        out: list[bool] = []
        for c in candidates:
            blocked = False
            if c.file_ref is not None:
                blocked = authorized_file is None or not authorized_file(c.file_ref)
                blocked = blocked or c.file_ref in blocked_sources
            if c.memory_ref is not None and not blocked:
                if c.memory_ref in blocked_sources:
                    blocked = True
                elif memories_by_id is not None:
                    mem = memories_by_id.get(c.memory_ref)
                    if mem is None or not mem.item.user_approved:
                        blocked = True
                    elif mem.item.source_uri and (
                        mem.item.source_uri in blocked_sources
                        or (authorized_file is not None and not authorized_file(mem.item.source_uri))
                    ):
                        blocked = True
            out.append(blocked)
        return out

    # ------------------------------------------------------------ decision

    def decide(
        self,
        scores: Any,
        candidates: Sequence[Candidate],
        obs: ScreenObservation | None,
        app_name: str | None = None,
        *,
        blocked: Sequence[bool] | None = None,
        is_logits: bool | None = None,
        record: bool = True,
    ) -> Suggestion:
        """Pick one candidate to show, or stay silent.

        Args:
            scores: logits or probabilities over ``[NO_SUGGESTION] + candidates``
                (``[A+1]`` or ``[1, A+1]``; extra padded slots are ignored).
            candidates: the real candidates, in the scorer's order.
            obs: latest observation; used for grounding and ``screen_version``.
            app_name: foreground app; defaults to ``obs.window.app_name``.
            blocked: optional per-candidate block flags (see ``blocked_mask``).
            is_logits: force interpretation of ``scores`` (auto-detect if None).
            record: when the gate speaks, record the key as shown (starts cooldown).
        """
        cands = list(candidates)
        vec = _to_vector(scores)
        if vec.size < len(cands) + 1:
            raise ValueError(f"got {vec.size} scores for {len(cands)} candidates (+1 abstain)")
        probs = to_probs(vec, is_logits)[: len(cands) + 1]
        total = probs.sum()
        probs = probs / total if total > 0 else probs
        labels = distribution_labels(cands)
        dist = {lab: float(p) for lab, p in zip(labels, probs)}
        version = obs.version if obs is not None else -1
        p_abstain = float(probs[0])

        def silent(reason: str) -> Suggestion:
            return Suggestion(candidate=None, confidence=p_abstain, screen_version=version,
                              reason=reason, distribution=dist)

        app = app_name if app_name is not None else (obs.window.app_name if obs is not None else "")
        quiet = {_norm_app(a) for a in self.cfg.quiet_apps}
        if app and _norm_app(app) in quiet:
            return silent(f"quiet mode for app '{app}'")
        if not cands:
            return silent("no candidates for this screen")

        with self._lock:
            now = self.clock()
            self._prune(now)
            if self._in_cooldown(now):
                left = self.cfg.cooldown_s - (now - (self._last_shown_at or now))
                return silent(f"cooldown: {left:.0f}s left since the last suggestion")

            order = sorted(range(len(cands)), key=lambda i: -probs[i + 1])
            best_p = float(probs[order[0] + 1])
            if best_p < self.cfg.relevance_threshold:
                return silent(f"low relevance: best {best_p:.2f} < threshold "
                              f"{self.cfg.relevance_threshold:.2f}")

            rejected: list[str] = []
            for i in order:
                c = cands[i]
                p = float(probs[i + 1])
                label = labels[i + 1]
                if p < self.cfg.relevance_threshold:
                    rejected.append(f"{label} below threshold ({p:.2f})")
                    break  # sorted: everything after is lower
                if p - p_abstain < self.cfg.margin_over_abstain:
                    rejected.append(f"{label} does not beat NO_SUGGESTION by "
                                    f"{self.cfg.margin_over_abstain:.2f} ({p:.2f} vs {p_abstain:.2f})")
                    break
                if blocked is not None and i < len(blocked) and blocked[i]:
                    rejected.append(f"{label} references a blocked source")
                    continue
                dup = self._duplicate_reason(c.key, now)
                if dup:
                    rejected.append(f"{label} {dup}")
                    continue
                if obs is not None and not verify_target(c, obs, self.iou_threshold):
                    rejected.append(f"{label} target no longer on screen")
                    continue
                if record:
                    self._shown[c.key] = now
                    self._last_shown_at = now
                skipped = f" (skipped: {'; '.join(rejected)})" if rejected else ""
                reason = (f"{label} p={p:.2f} >= threshold {self.cfg.relevance_threshold:.2f} "
                          f"and beats NO_SUGGESTION ({p_abstain:.2f}){skipped}")
                return Suggestion(candidate=c, confidence=p, screen_version=version,
                                  reason=reason, distribution=dist)
            return silent("; ".join(rejected) or "no eligible candidate")

