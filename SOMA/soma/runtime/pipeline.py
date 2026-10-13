"""SOMA runtime loop (build plan section 9): one synchronous step per call.

``SomaRuntime.step()`` runs::

    consent / pause / exclusion  ->  ScreenCapturer.capture()   (None -> skipped, no OCR/model)
    OCR (+ accessibility)        ->  parse_screen -> redact
    featurize                    ->  frame_to_tensor + observation_to_tensors
    SomaModel.encode_screen      ->  screen_state [1, D]  (pushed to the rolling history afterwards)
    retrieval                    ->  TextEmbedder(redacted screen text) -> permission-filtered top-k
    candidates                   ->  generate_candidates -> blocked_mask -> CandidateEncoder
    policy                       ->  temporal -> memory fusion -> action logits [1, A+1] (+ online ranker bias)
    gate                         ->  SafetyGate.decide -> Suggestion (or silence)

Design notes (DECISIONS-ready):

* **Retrieval query space.** The memory index is built by ``TextEmbedder`` at
  ingestion time, not by the model, so the query must live in the same space:
  we embed the *redacted* observation text (app name, title, OCR text) with
  the same ``TextEmbedder`` rather than using ``SomaModel.query_embedding``.
  The factory builds ``TextEmbedder(dim=cfg.model.d_model)`` so retrieved
  vectors feed ``memory_embeddings`` directly; if dims ever differ, vectors
  are truncated / zero-padded and re-normalised (:func:`fit_dim`).
* **Single screen encode.** ``encode_screen`` runs once per frame; the
  temporal, memory-fusion and policy sub-modules are then applied to that
  state (identical to ``SomaModel.forward``; checked in tests), so the
  visual encoder is not run twice.
* **Nothing executes.** The runtime only returns suggestions. It imports no
  process/shell/network-execution modules and never calls an executor.
* **Nothing persists.** Frames and pixel tensors are dropped after each step.
  Snapshots kept for feedback have their images zeroed. Only consented
  feedback outcomes and suggestion metadata (no screen text) reach SQLite.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch

from soma.capture.redaction import is_excluded, redact_elements, redact_text
from soma.capture.window import neutral_window
from soma.config import SomaConfig
from soma.consent import ConsentManager, Scope
from soma.device import select_device
from soma.learning.datasets import SomaItem, item_from_batch
from soma.learning.feedback import FeedbackRecorder
from soma.learning.online_ranker import OnlinePreferenceRanker
from soma.memory.retrieval import to_tensors
from soma.models.batch import SomaBatch
from soma.perception.featurize import app_bucket, observation_to_tensors
from soma.perception.ocr import NullOCREngine, OCREngine
from soma.perception.preprocess import frame_to_tensor
from soma.perception.screen_parser import parse_screen
from soma.policy.candidate_encoder import CandidateEncoder
from soma.policy.candidate_generator import generate_candidates
from soma.policy.context_candidates import add_context_candidates
from soma.policy.decision import ApprovalLedger
from soma.policy.safety_gate import NO_SUGGESTION_LABEL, SafetyGate
from soma.runtime.history import ScreenHistory
from soma.runtime.telemetry import Telemetry
from soma.types import (
    Candidate,
    FeedbackEvent,
    FeedbackOutcome,
    OCRElement,
    RetrievedMemory,
    ScreenFrame,
    ScreenObservation,
    Suggestion,
    WindowInfo,
)

if TYPE_CHECKING:  # pragma: no cover
    from soma.capture.screen import ScreenCapturer
    from soma.models.soma_model import SomaModel
    from soma.runtime.factory import MemoryBundle

# Skip reasons (``StepResult.skipped_reason``); capture adds its own
# ("rate-limited", "unchanged", "excluded:<why>", "grab-error:<type>").
SKIP_PAUSED = "paused"
SKIP_NO_CONSENT = "no-consent"
SKIP_EXCLUDED = "excluded"
SKIP_NO_CAPTURER = "no-capturer"
SKIP_STALE = "paused-during-step"

#: Max characters of redacted screen text used for the retrieval query.
QUERY_CHARS = 4000
#: How many recent shown suggestions are remembered for feedback routing.
MAX_PENDING = 64

AccessibilityFn = Callable[[WindowInfo], Sequence[OCRElement]]
AuthorizedFile = Callable[[str], bool]


@dataclass
class StepResult:
    frame_version: int
    window: WindowInfo
    observation: ScreenObservation
    candidates: list[Candidate]
    suggestion: Suggestion            # suggestion.is_silent when NO_SUGGESTION / gated
    retrieved: list[RetrievedMemory]
    timings_ms: dict[str, float]      # capture, ocr, parse, encode, retrieve, policy, gate, total
    skipped_reason: str | None = None # set when no frame was processed


@dataclass
class PreparedStep:
    """Everything computed before scoring (exposed for tests and data collection)."""

    frame_version: int
    window: WindowInfo
    observation: ScreenObservation
    retrieved: list[RetrievedMemory]
    candidates: list[Candidate]
    blocked: list[bool]
    batch: SomaBatch                  # B = 1, on the runtime device
    screen_state: torch.Tensor        # [1, D]
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_item(self, target: int = 0, *, app: str | None = None, task: str = "",
                keep_images: bool = False) -> SomaItem:
        """Detached CPU :class:`SomaItem` snapshot (images zeroed unless ``keep_images``)."""
        item = item_from_batch(self.batch, 0, target=target, app=app or self.window.app_name,
                               task=task)
        if not keep_images:
            item.images = torch.zeros_like(item.images)
        return item


@dataclass
class _Shown:
    suggestion: Suggestion
    candidate: Candidate
    shown_index: int                  # logits index (1..A)
    app: str
    item: SomaItem                    # images zeroed
    resolved: bool = False


def fit_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    """Truncate / zero-pad a vector to ``dim`` and re-normalise (unit length if non-zero)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] != dim:
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, v.shape[0])
        out[:n] = v[:n]
        v = out
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def seeded_candidate_encoder(cfg: SomaConfig, seed: int | None = None) -> CandidateEncoder:
    """A ``CandidateEncoder`` with a deterministic init that leaves the global RNG alone."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.train.seed if seed is None else seed)
        enc = CandidateEncoder.from_config(cfg.model)
    return enc.eval()


def _silent(version: int, reason: str) -> Suggestion:
    return Suggestion(candidate=None, confidence=1.0, screen_version=version, reason=reason,
                      distribution={NO_SUGGESTION_LABEL: 1.0})


class SomaRuntime:
    """Synchronous SOMA loop. Thread-safe: ``step`` calls are serialised;
    ``feedback``/``pause``/``resume`` may be called from any thread.

    Collaborators are injected (tests pass fakes); ``soma.runtime.factory``
    wires the real ones. ``memory`` is a :class:`~soma.runtime.factory.MemoryBundle`
    (or any object with ``retriever``, ``text_embedder``, ``store`` and
    optionally ``index``/``ingestor``); ``None`` runs with empty memory.
    """

    def __init__(self, cfg: SomaConfig, consent: ConsentManager, *,
                 capturer: "ScreenCapturer | None" = None, ocr: OCREngine | None = None,
                 model: "SomaModel | None" = None, memory: "MemoryBundle | None" = None,
                 gate: SafetyGate | None = None, candidate_encoder: CandidateEncoder | None = None,
                 device: torch.device | str | None = None,
                 authorized_file: AuthorizedFile | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 accessibility: AccessibilityFn | None = None,
                 ranker: OnlinePreferenceRanker | None = None,
                 recorder: FeedbackRecorder | None = None,
                 telemetry: Telemetry | None = None,
                 model_seed: int | None = None) -> None:
        self.cfg = cfg
        self.consent = consent
        self.clock = clock
        self.device = torch.device(device) if device is not None else select_device(cfg.device)
        self.capturer = capturer
        self.ocr: OCREngine = ocr if ocr is not None else NullOCREngine()
        if model is None:
            from soma.models.soma_model import build_model
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.train.seed if model_seed is None else model_seed)
                model = build_model(cfg.model, pretrained=False)
        self.model = model.to(self.device).eval()
        self.candidate_encoder = (candidate_encoder if candidate_encoder is not None
                                  else seeded_candidate_encoder(cfg)).to(self.device).eval()
        self.memory = memory
        self.gate = gate if gate is not None else SafetyGate(cfg.policy, clock=clock)
        self.ranker = ranker if ranker is not None else OnlinePreferenceRanker()
        sink = memory.store if memory is not None else None
        self.recorder = recorder if recorder is not None else FeedbackRecorder(consent, sink)
        self.telemetry = telemetry if telemetry is not None else Telemetry(enabled=False, clock=clock)
        self.accessibility = accessibility
        self._authorized_file = authorized_file
        self.ledger = ApprovalLedger()          # suggestions are never executed (no executors)
        self.history = ScreenHistory(cfg.model.history_len, cfg.model.d_model)

        self._step_lock = threading.Lock()
        self._lock = threading.RLock()
        self._paused = False
        self._gen = 0
        self._last_result: StepResult | None = None
        self._last_skip_reason: str | None = None
        self._last_version = 0
        self._shown: OrderedDict[str, _Shown] = OrderedDict()
        self.steps = 0
        self.processed = 0

    # ------------------------------------------------------------ state
    @property
    def paused(self) -> bool:
        return self._paused or self.consent.paused

    @property
    def generation(self) -> int:
        """Bumped on every pause; results computed under an older generation are stale."""
        return self._gen

    @property
    def last_result(self) -> StepResult | None:
        """The most recent result for which a frame was actually processed."""
        with self._lock:
            return self._last_result

    @property
    def last_skip_reason(self) -> str | None:
        """Reason the most recent ``step`` was skipped (``None`` if it processed a frame)."""
        return self._last_skip_reason

    def pause(self) -> None:
        """Stop instantly: capture is denied, history is cleared, in-flight results go stale."""
        with self._lock:
            self._paused = True
            self._gen += 1
        self.consent.pause()
        self.history.clear()
        self.telemetry.mark_inactive()

    def resume(self) -> None:
        self.consent.resume()
        reset = getattr(self.capturer, "reset", None)
        if callable(reset):
            reset()
        with self._lock:
            self._paused = False

    def authorized_file(self, path: str) -> bool:
        """Default file ACL: inside a user-allowed folder and not in a blocked source."""
        if self._authorized_file is not None:
            return bool(self._authorized_file(path))
        if not self.consent.path_allowed(path):
            return False
        store = getattr(self.memory, "store", None)
        return not (store is not None and store.is_source_blocked(path))

    def current_suggestion_valid(self, current_version: int) -> bool:
        """True while the last shown suggestion still matches the current screen version
        and has not received feedback."""
        with self._lock:
            r = self._last_result
            if r is None or r.suggestion.is_silent or self.paused:
                return False
            entry = self._shown.get(r.suggestion.id)
            if entry is not None and entry.resolved:
                return False
            return r.suggestion.screen_version == int(current_version)

    def known_suggestion(self, suggestion_id: str) -> bool:
        with self._lock:
            return suggestion_id in self._shown

    # ------------------------------------------------------------ loop
    def step(self, force: bool = False) -> StepResult:
        """One loop iteration: capture (if allowed and meaningful), then process."""
        t0 = time.perf_counter()
        with self._step_lock:
            self.steps += 1
            if self.paused:
                return self._skipped(SKIP_PAUSED, t0)
            if not self.consent.allowed(Scope.SCREEN_CAPTURE):
                return self._skipped(SKIP_NO_CONSENT, t0)
            if self.capturer is None:
                return self._skipped(SKIP_NO_CAPTURER, t0)
            frame = self.capturer.capture(force=force)
            capture_ms = (time.perf_counter() - t0) * 1000.0
            if frame is None:
                reason = getattr(self.capturer, "last_skip_reason", "") or "no-frame"
                return self._skipped(reason, t0, window=getattr(self.capturer, "last_window", None),
                                     timings={"capture": capture_ms})
            try:
                return self._process(frame, t0, capture_ms)
            finally:
                del frame   # never keep pixels beyond the step

    def process_frame(self, frame: ScreenFrame) -> StepResult:
        """Process an already captured frame (tests, synthetic demo). Same privacy gates."""
        t0 = time.perf_counter()
        with self._step_lock:
            self.steps += 1
            return self._process(frame, t0, 0.0)

    def _gate_frame(self, frame: ScreenFrame) -> str | None:
        if self.paused:
            return SKIP_PAUSED
        if not self.consent.allowed(Scope.SCREEN_CAPTURE):
            return SKIP_NO_CONSENT
        excluded, why = is_excluded(frame.window, self.cfg.capture)
        if excluded:
            return f"{SKIP_EXCLUDED}:{why}"
        return None

    def _process(self, frame: ScreenFrame, t0: float, capture_ms: float) -> StepResult:
        reason = self._gate_frame(frame)
        if reason is not None:
            return self._skipped(reason, t0, window=frame.window, timings={"capture": capture_ms},
                                 version=frame.version)
        gen = self._gen
        prepared = self.prepare(frame)
        prepared.timings_ms["capture"] = capture_ms
        timings = prepared.timings_ms

        t = time.perf_counter()
        logits = self.score(prepared)
        timings["policy"] = timings.get("policy", 0.0) + (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        suggestion = self.gate.decide(logits[0], prepared.candidates, prepared.observation,
                                      blocked=prepared.blocked, record=False)
        timings["gate"] = (time.perf_counter() - t) * 1000.0

        # history gets the current state only after it was used as "previous" context
        self.history.push(prepared.screen_state[0])

        if gen != self._gen or self.paused:
            # paused while computing: drop the result, record nothing as shown
            return self._skipped(SKIP_STALE, t0, window=prepared.window,
                                 version=prepared.frame_version)

        if not suggestion.is_silent:
            self._remember_shown(suggestion, prepared)

        timings["total"] = (time.perf_counter() - t0) * 1000.0
        for k in ("capture", "ocr", "parse", "encode", "retrieve", "policy", "gate"):
            timings.setdefault(k, 0.0)
        result = StepResult(frame_version=prepared.frame_version, window=prepared.window,
                            observation=prepared.observation, candidates=prepared.candidates,
                            suggestion=suggestion, retrieved=prepared.retrieved,
                            timings_ms=timings, skipped_reason=None)
        with self._lock:
            self._last_result = result
            self._last_skip_reason = None
            self._last_version = prepared.frame_version
            self.processed += 1
        self.telemetry.record_step(timings, silent=suggestion.is_silent)
        return result

    def _skipped(self, reason: str, t0: float, window: WindowInfo | None = None,
                 timings: dict[str, float] | None = None, version: int | None = None) -> StepResult:
        # Only the app name of a skipped window is kept; titles of excluded windows may be sensitive.
        win = WindowInfo(app_name=window.app_name, title="") if window is not None else neutral_window()
        if version is None:
            version = int(getattr(self.capturer, "version", self._last_version) or 0)
        tm = {k: 0.0 for k in ("capture", "ocr", "parse", "encode", "retrieve", "policy", "gate")}
        tm.update(timings or {})
        tm["total"] = (time.perf_counter() - t0) * 1000.0
        result = StepResult(frame_version=version, window=win,
                            observation=ScreenObservation(window=win, elements=[], version=version),
                            candidates=[], suggestion=_silent(version, f"skipped: {reason}"),
                            retrieved=[], timings_ms=tm, skipped_reason=reason)
        self._last_skip_reason = reason
        self.telemetry.record_step(tm, silent=True, skipped_reason=reason)
        return result

    # ------------------------------------------------------------ stages
    def prepare(self, frame: ScreenFrame) -> PreparedStep:
        """OCR -> parse -> redact -> featurize -> encode_screen -> retrieve -> candidates.

        Does not check consent (``step``/``process_frame`` do) and does not
        touch the gate or history.
        """
        mc = self.cfg.model
        timings: dict[str, float] = {}

        t = time.perf_counter()
        ocr_elements = self.ocr.recognize(frame.image, frame.window.window_id, frame.timestamp_ms)
        a11y: Sequence[OCRElement] = ()
        if self.accessibility is not None and self.consent.allowed(Scope.APP_MONITORING):
            a11y = self.accessibility(frame.window)
        timings["ocr"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        window = WindowInfo(app_name=frame.window.app_name, title=redact_text(frame.window.title),
                            window_id=frame.window.window_id, pid=frame.window.pid,
                            bounds=frame.window.bounds)
        obs = parse_screen(frame, ocr_elements, a11y, window=window)
        obs.elements = redact_elements(obs.elements)
        timings["parse"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        image, _ = frame_to_tensor(frame.image, mc.image_size)
        ids, boxes, ocr_mask = observation_to_tensors(obs, mc)
        app_id = torch.tensor([app_bucket(window, mc.n_apps)], dtype=torch.long)
        dev = self.device
        with torch.inference_mode():
            screen_state = self.model.encode_screen(image[None].to(dev), ids[None].to(dev),
                                                    boxes[None].to(dev), ocr_mask[None].to(dev))
        self.history.observe_app(window.app_name)
        hist, hist_mask = self.history.tensors()
        timings["encode"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        retrieved = self._retrieve(obs)
        mem_emb, mem_mask = to_tensors([RetrievedMemory(item=r.item, embedding=fit_dim(r.embedding, mc.d_model),
                                                        score=r.score) for r in retrieved],
                                       mc.max_memories, mc.d_model)
        timings["retrieve"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        A = mc.max_candidates
        candidates = generate_candidates(obs, retrieved, A, self.authorized_file)
        if self.cfg.policy.context_templates:
            candidates = add_context_candidates(obs, candidates, A)
        mem_by_id = {r.item.id: r for r in retrieved}
        blocked = SafetyGate.blocked_mask(candidates, self.authorized_file,
                                          memories_by_id=mem_by_id)
        with torch.inference_mode():
            c_emb, c_types, c_mask = self.candidate_encoder.encode(candidates, mem_by_id, A, mc.d_model,
                                                                   blocked=blocked)
        batch = SomaBatch(
            images=image[None],               # transient; dropped with the PreparedStep
            ocr_ids=ids[None], ocr_boxes=boxes[None], ocr_padding_mask=ocr_mask[None],
            history=hist[None], history_padding_mask=hist_mask[None], app_ids=app_id,
            memory_embeddings=mem_emb[None], memory_padding_mask=mem_mask[None],
            candidate_embeddings=c_emb[None].cpu(), candidate_types=c_types[None].cpu(),
            candidate_mask=c_mask[None].cpu(),
        ).to(dev)
        timings["policy"] = (time.perf_counter() - t) * 1000.0  # candidate prep; scoring added later
        return PreparedStep(frame_version=frame.version, window=window, observation=obs,
                            retrieved=retrieved, candidates=candidates, blocked=blocked,
                            batch=batch, screen_state=screen_state, timings_ms=timings)

    def _retrieve(self, obs: ScreenObservation) -> list[RetrievedMemory]:
        mem = self.memory
        if mem is None or getattr(mem, "retriever", None) is None:
            return []
        k = min(int(self.cfg.memory.top_k), int(self.cfg.model.max_memories))
        text = redact_text(f"{obs.window.app_name} {obs.window.title}\n{obs.text}")[:QUERY_CHARS]
        if not text.strip():
            return []
        query = mem.text_embedder.embed_text(text)
        return list(mem.retriever.retrieve(query, k=k))

    def score(self, prepared: PreparedStep) -> torch.Tensor:
        """Action logits ``[1, A+1]`` (index 0 = NO_SUGGESTION) plus the online ranker bias.

        Applies temporal -> memory fusion -> policy to the already encoded
        screen state (same maths as ``SomaModel.forward``).
        """
        m, b = self.model, prepared.batch
        with torch.inference_mode():
            if all(hasattr(m, a) for a in ("temporal_encoder", "memory_fusion", "action_policy")):
                temporal = m.temporal_encoder(prepared.screen_state, b.history,
                                              b.history_padding_mask, b.app_ids)
                mem = m.memory_fusion(temporal, b.memory_embeddings, b.memory_padding_mask)
                logits = m.action_policy(mem.context, b.candidate_embeddings, b.candidate_types,
                                         b.candidate_mask)
            else:  # any model exposing the SomaModel forward contract
                logits = m(b)["action_logits"]
            logits = logits.float().cpu()
            if self.ranker is not None and prepared.candidates:
                bias = self.ranker.bias(prepared.candidates, prepared.window.app_name,
                                        n_slots=logits.shape[1] - 1)
                logits = OnlinePreferenceRanker.apply(logits, bias[: logits.shape[1]])
        return logits

    # ------------------------------------------------------------ feedback
    def _remember_shown(self, suggestion: Suggestion, prepared: PreparedStep) -> None:
        cand = suggestion.candidate
        assert cand is not None
        idx = next(i for i, c in enumerate(prepared.candidates) if c is cand) + 1
        self.gate.record_shown(cand.key)
        entry = _Shown(suggestion=suggestion, candidate=cand, shown_index=idx,
                       app=prepared.window.app_name,
                       item=prepared.to_item(idx, task=cand.action.value))
        with self._lock:
            self._shown[suggestion.id] = entry
            while len(self._shown) > MAX_PENDING:
                self._shown.popitem(last=False)
        store = getattr(self.memory, "store", None)
        if store is not None and self.consent.allowed(Scope.FEEDBACK_STORAGE):
            store.log_suggestion(cand.action.value, cand.key, suggestion.screen_version,
                                 suggestion.confidence)

    def feedback(self, suggestion_id: str, outcome: FeedbackOutcome,
                 edited_text: str | None = None) -> None:
        """Route explicit feedback on a shown suggestion.

        * REJECTED: the candidate is dismissed in the gate (not repeated within
          the duplicate window) -- always, regardless of storage consent.
        * With FEEDBACK_STORAGE consent: the event goes to the recorder (and the
          memory store), the online ranker is updated, and an edited text is
          kept as a redacted FEEDBACK memory.
        * IGNORED is recorded as ambiguous and is never treated as a negative.
        Unknown ids are ignored.
        """
        outcome = FeedbackOutcome(outcome)
        with self._lock:
            entry = self._shown.get(suggestion_id)
            if entry is None:
                return
            entry.resolved = True
        cand = entry.candidate
        if outcome is FeedbackOutcome.REJECTED:
            self.gate.record_dismissed(cand.key)
        self.telemetry.record_feedback(outcome.value)
        if not self.consent.allowed(Scope.FEEDBACK_STORAGE):
            return
        event = FeedbackEvent(suggestion_id=suggestion_id, outcome=outcome,
                              context_ref=cand.memory_ref)
        self.recorder.record(event, item=entry.item, shown_index=entry.shown_index,
                             app=entry.app, task=cand.action.value)
        if outcome is not FeedbackOutcome.IGNORED:
            self.ranker.update_candidate(cand, entry.app, outcome)
        ingestor = getattr(self.memory, "ingestor", None)
        if (outcome is FeedbackOutcome.EDITED and edited_text and edited_text.strip()
                and ingestor is not None):
            ingestor.add_feedback_memory(edited_text.strip(), suggestion_id,
                                         tags=["edited", cand.action.value])

    def suggestion_info(self, suggestion_id: str) -> dict[str, Any] | None:
        """Plain-dict view of a remembered suggestion (no screen text beyond the suggestion)."""
        with self._lock:
            entry = self._shown.get(suggestion_id)
        if entry is None:
            return None
        return {"id": suggestion_id, "action": entry.candidate.action.value,
                "app": entry.app, "resolved": entry.resolved}


__all__ = ["PreparedStep", "SomaRuntime", "StepResult", "fit_dim",
           "seeded_candidate_encoder"]
