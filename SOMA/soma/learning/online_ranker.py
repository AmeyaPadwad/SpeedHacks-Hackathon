"""Stage C: lightweight online preference model that never touches base weights.

Per-app Beta-Bernoulli "arms" track explicit feedback for

* each action type in that app (``type`` arms),
* each concrete candidate key in that app (``key`` arms), and
* abstention in that app (``abstain`` arm: a rejection is evidence that
  silence was better; an accept/edit is evidence against silence).

``bias(candidates, app)`` returns an additive ``[A + 1]`` logit bias (index 0
= ``NO_SUGGESTION``) equal to the clipped posterior log-odds of each arm.
With no evidence every bias is exactly 0, magnitudes are bounded by
``max_bias``, and all evidence decays geometrically so stale preferences fade.
``IGNORED`` outcomes are ambiguous and are not counted.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Sequence

import torch

from soma.types import ActionType, Candidate, FeedbackOutcome

#: Logits at or below this are treated as blocked candidates (model uses ~-1e4).
BLOCKED_THRESHOLD = -1e3

# (success, failure) increments for a shown candidate's arms.
_CANDIDATE_EVIDENCE: dict[FeedbackOutcome, tuple[float, float]] = {
    FeedbackOutcome.ACCEPTED: (1.0, 0.0),
    FeedbackOutcome.EDITED: (0.5, 0.0),
    FeedbackOutcome.REJECTED: (0.0, 1.0),
}
# (success, failure) for the abstain arm: success = "silence would have been better".
_ABSTAIN_EVIDENCE: dict[FeedbackOutcome, tuple[float, float]] = {
    FeedbackOutcome.ACCEPTED: (0.0, 1.0),
    FeedbackOutcome.EDITED: (0.0, 0.5),
    FeedbackOutcome.REJECTED: (1.0, 0.0),
}


class OnlinePreferenceRanker:
    """Bounded, decaying Beta-Bernoulli logit biases per (app, type / key).

    Parameters
    ----------
    prior:      pseudo-count on each side of every arm (Beta(prior, prior)).
    scale:      multiplier on candidate log-odds.
    abstain_scale: multiplier on the abstain arm's log-odds.
    key_weight: share of the key arm relative to the type arm.
    max_bias:   absolute clip on every returned bias.
    decay:      evidence multiplier applied to all arms on each update.
    """

    def __init__(self, *, prior: float = 1.0, scale: float = 1.0, abstain_scale: float = 1.0,
                 key_weight: float = 0.5, max_bias: float = 2.0, decay: float = 0.98) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.prior = float(prior)
        self.scale = float(scale)
        self.abstain_scale = float(abstain_scale)
        self.key_weight = float(key_weight)
        self.max_bias = float(max_bias)
        self.decay = float(decay)
        self._arms: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.updates = 0

    # ------------------------------------------------------------ evidence
    @staticmethod
    def _arm(kind: str, app: str, name: str = "") -> str:
        return f"{kind}|{(app or '').lower()}|{name}"

    def _add(self, arm: str, s: float, f: float) -> None:
        a = self._arms.setdefault(arm, [0.0, 0.0])
        a[0] += s
        a[1] += f

    def _log_odds(self, arm: str) -> float:
        s, f = self._arms.get(arm, (0.0, 0.0))
        return math.log((s + self.prior) / (f + self.prior))

    def _clip(self, x: float) -> float:
        return max(-self.max_bias, min(self.max_bias, x))

    def update(self, app: str, action: ActionType | str, outcome: FeedbackOutcome | str,
               candidate_key: str | None = None) -> bool:
        """Count one explicit outcome. Returns False (no-op) for ignored/unknown outcomes."""
        outcome = FeedbackOutcome(outcome)
        if outcome not in _CANDIDATE_EVIDENCE:
            return False
        action = ActionType(action)
        with self._lock:
            for arm in self._arms.values():
                arm[0] *= self.decay
                arm[1] *= self.decay
            if action is not ActionType.NO_SUGGESTION:
                s, f = _CANDIDATE_EVIDENCE[outcome]
                self._add(self._arm("type", app, action.value), s, f)
                if candidate_key:
                    self._add(self._arm("key", app, candidate_key), s, f)
            s, f = _ABSTAIN_EVIDENCE[outcome]
            self._add(self._arm("abstain", app), s, f)
            self.updates += 1
        return True

    def update_candidate(self, candidate: Candidate, app: str, outcome: FeedbackOutcome | str) -> bool:
        """Convenience: update from a shown :class:`Candidate`."""
        return self.update(app, candidate.action, outcome, candidate.key)

    # ------------------------------------------------------------ scoring
    def candidate_bias(self, app: str, action: ActionType | str, candidate_key: str | None = None) -> float:
        action = ActionType(action)
        with self._lock:
            b = self.scale * self._log_odds(self._arm("type", app, action.value))
            if candidate_key:
                b += self.scale * self.key_weight * self._log_odds(self._arm("key", app, candidate_key))
        return self._clip(b)

    def abstain_bias(self, app: str) -> float:
        with self._lock:
            return self._clip(self.abstain_scale * self._log_odds(self._arm("abstain", app)))

    def bias(self, candidates: Sequence[Candidate | None], app: str, n_slots: int | None = None,
             candidate_keys: Sequence[str | None] | None = None) -> torch.Tensor:
        """Additive logit bias ``[A + 1]`` (index 0 = NO_SUGGESTION).

        ``candidates`` may contain ``None`` for empty/padded slots (bias 0).
        ``n_slots`` pads the result to a model's fixed ``A``.
        """
        A = max(len(candidates), n_slots or 0)
        out = torch.zeros(A + 1)
        out[0] = self.abstain_bias(app)
        for i, c in enumerate(candidates):
            if c is None:
                continue
            key = candidate_keys[i] if candidate_keys is not None else c.key
            out[i + 1] = self.candidate_bias(app, c.action, key)
        return out

    def bias_for_types(self, types: Sequence[ActionType | str | None], app: str) -> torch.Tensor:
        """Bias ``[A + 1]`` from action types only (no candidate keys)."""
        out = torch.zeros(len(types) + 1)
        out[0] = self.abstain_bias(app)
        for i, t in enumerate(types):
            if t is not None and ActionType(t) is not ActionType.NO_SUGGESTION:
                out[i + 1] = self.candidate_bias(app, t)
        return out

    @staticmethod
    def apply(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """``logits + bias`` that leaves blocked entries (``-inf`` or the model's
        large finite ``BLOCKED_LOGIT``) untouched. ``bias`` broadcasts over ``[B, A+1]``."""
        bias = bias.to(device=logits.device, dtype=logits.dtype)
        open_ = torch.isfinite(logits) & (logits > BLOCKED_THRESHOLD)
        return torch.where(open_, logits + bias, logits)

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"prior": self.prior, "scale": self.scale, "abstain_scale": self.abstain_scale,
                    "key_weight": self.key_weight, "max_bias": self.max_bias, "decay": self.decay,
                    "updates": self.updates, "arms": {k: list(v) for k, v in self._arms.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lock:
            for name in ("prior", "scale", "abstain_scale", "key_weight", "max_bias", "decay"):
                setattr(self, name, float(state[name]))
            self.updates = int(state.get("updates", 0))
            self._arms = {k: [float(v[0]), float(v[1])] for k, v in state["arms"].items()}

    def reset(self, app: str | None = None) -> None:
        """Forget all evidence, or only one app's."""
        with self._lock:
            if app is None:
                self._arms.clear()
            else:
                needle = f"|{app.lower()}|"
                self._arms = {k: v for k, v in self._arms.items() if needle not in f"|{k.split('|', 1)[1]}"}
