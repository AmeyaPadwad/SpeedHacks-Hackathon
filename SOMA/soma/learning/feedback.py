"""Explicit feedback collection and conversion to labeled examples (Stage C/D input).

Rules (build plan sections 8 and 10):

* Nothing is recorded without ``Scope.FEEDBACK_STORAGE`` consent.
* ``FeedbackEvent.consented_for_training`` is forced to reflect
  ``Scope.MODEL_TRAINING`` *at record time*; a caller cannot set it to True.
* ``IGNORED`` suggestions are recorded (they matter for interruption metrics)
  but flagged **ambiguous**: an ignored suggestion is never treated as a
  rejection or used as a negative label.
* Labeled examples hold a detached tensor snapshot (ids, boxes, embeddings).
  By default screen pixels are dropped (zeroed) so no screenshot lingers in a
  replay buffer; pass ``keep_images=True`` only for consented sessions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from soma.consent import ConsentManager, Scope
from soma.learning.datasets import NO_SUGGESTION_INDEX, SomaItem
from soma.types import FeedbackEvent, FeedbackOutcome, now_ms

# Relative training weights per outcome. An edit means "right idea, wrong
# details", so it is a softer positive; a rejection teaches "silence was
# better here" with a softer weight because the user may have wanted another
# candidate instead.
OUTCOME_WEIGHTS: dict[FeedbackOutcome, float] = {
    FeedbackOutcome.ACCEPTED: 1.0,
    FeedbackOutcome.EDITED: 0.5,
    FeedbackOutcome.REJECTED: 0.5,
    FeedbackOutcome.IGNORED: 0.0,
}


@runtime_checkable
class FeedbackSink(Protocol):
    """Where raw feedback events go (``MemoryStore.add_feedback`` satisfies this)."""

    def add_feedback(self, event: FeedbackEvent) -> Any: ...


class InMemoryFeedbackSink:
    """Thread-safe list sink for tests and ephemeral sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[FeedbackEvent] = []

    def add_feedback(self, event: FeedbackEvent) -> None:
        with self._lock:
            self.events.append(event)


@dataclass
class LabeledExample:
    """A training example derived from one feedback event.

    ``item.target`` is the logits index to learn (``0`` = NO_SUGGESTION).
    ``rejected_index`` is the shown candidate's logits index for a rejection
    (usable with a pairwise ranking loss). Ambiguous examples carry no label.
    """

    item: SomaItem
    outcome: FeedbackOutcome
    consented: bool
    ambiguous: bool = False
    shown_index: int | None = None
    rejected_index: int | None = None
    source: str = "feedback"                 # "feedback" | "anchor" | "synthetic"
    feedback_id: str = ""
    created_ms: int = field(default_factory=now_ms)

    @property
    def usable_for_training(self) -> bool:
        """Only consented, unambiguous examples may be trained on."""
        return self.consented and not self.ambiguous and self.outcome is not FeedbackOutcome.IGNORED

    @property
    def app(self) -> str:
        return self.item.app

    @property
    def task(self) -> str:
        return self.item.task

    def to_state(self) -> dict[str, Any]:
        return {"item": self.item.to_state(), "outcome": self.outcome.value,
                "consented": self.consented, "ambiguous": self.ambiguous,
                "shown_index": self.shown_index, "rejected_index": self.rejected_index,
                "source": self.source, "feedback_id": self.feedback_id,
                "created_ms": self.created_ms}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LabeledExample":
        data = dict(state)
        data["item"] = SomaItem.from_state(data["item"])
        data["outcome"] = FeedbackOutcome(data["outcome"])
        return cls(**data)


def as_examples(items: list[SomaItem], source: str = "synthetic") -> list[LabeledExample]:
    """Wrap curated (non-user) items as consented, unambiguous examples."""
    return [LabeledExample(item=i, outcome=FeedbackOutcome.ACCEPTED, consented=True,
                           ambiguous=False, shown_index=i.target, source=source) for i in items]


def example_from_feedback(item: SomaItem, outcome: FeedbackOutcome, shown_index: int, *,
                          consented: bool, keep_images: bool = False,
                          feedback_id: str = "") -> LabeledExample:
    """Convert one outcome on the shown candidate (logits index) into a labeled example.

    * accepted -> target = shown candidate, weight 1.0
    * edited   -> target = shown candidate, weight 0.5
    * rejected -> target = NO_SUGGESTION, weight 0.5, ``rejected_index`` = shown
    * ignored  -> ambiguous, weight 0, no label (target left at NO_SUGGESTION
      but excluded from training via ``usable_for_training``)
    """
    snap = item.clone()
    if not keep_images:
        snap.images = torch.zeros_like(snap.images)
    ambiguous = outcome is FeedbackOutcome.IGNORED
    rejected = None
    if outcome in (FeedbackOutcome.ACCEPTED, FeedbackOutcome.EDITED):
        snap.target = int(shown_index)
    else:
        snap.target = NO_SUGGESTION_INDEX
        if outcome is FeedbackOutcome.REJECTED and shown_index != NO_SUGGESTION_INDEX:
            rejected = int(shown_index)
    snap.weight = OUTCOME_WEIGHTS[outcome]
    snap.meta = {**snap.meta, "outcome": outcome.value}
    return LabeledExample(item=snap, outcome=outcome, consented=consented, ambiguous=ambiguous,
                          shown_index=int(shown_index), rejected_index=rejected,
                          source="feedback", feedback_id=feedback_id)


class FeedbackRecorder:
    """Consent-gated recorder: event -> sink, plus an in-memory labeled example.

    ``record`` returns ``None`` (and stores nothing) without
    ``FEEDBACK_STORAGE``. With it, the event goes to ``sink`` and, if an input
    snapshot is given, a :class:`LabeledExample` is returned and kept in
    :attr:`examples`. Only examples with ``MODEL_TRAINING`` consent and a
    non-ignored outcome are returned by :meth:`training_examples`.
    """

    def __init__(self, consent: ConsentManager, sink: FeedbackSink | None = None, *,
                 keep_images: bool = False, max_examples: int = 10_000) -> None:
        self.consent = consent
        self.sink = sink if sink is not None else InMemoryFeedbackSink()
        self.keep_images = keep_images
        self.max_examples = max_examples
        self._lock = threading.Lock()
        self.examples: list[LabeledExample] = []

    def record(self, event: FeedbackEvent, *, item: SomaItem | None = None,
               shown_index: int | None = None, app: str | None = None,
               task: str | None = None) -> LabeledExample | None:
        """Record ``event``; ``item``/``shown_index`` are the inputs and shown slot."""
        if not self.consent.allowed(Scope.FEEDBACK_STORAGE):
            return None
        event.consented_for_training = self.consent.allowed(Scope.MODEL_TRAINING)
        if not event.observed_at:
            event.observed_at = str(now_ms())
        self.sink.add_feedback(event)
        if item is None or shown_index is None:
            return None
        if app is not None or task is not None:
            item = item.clone(app=app if app is not None else item.app,
                              task=task if task is not None else item.task)
        ex = example_from_feedback(item, event.outcome, shown_index,
                                   consented=event.consented_for_training,
                                   keep_images=self.keep_images, feedback_id=event.id)
        with self._lock:
            self.examples.append(ex)
            if len(self.examples) > self.max_examples:
                del self.examples[0]
        return ex

    def training_examples(self) -> list[LabeledExample]:
        """Consented, unambiguous examples. Re-checks MODEL_TRAINING consent now."""
        if not self.consent.allowed(Scope.MODEL_TRAINING):
            return []
        with self._lock:
            return [e for e in self.examples if e.usable_for_training]

    def forget(self) -> None:
        """Drop every in-memory example (e.g. after consent is revoked)."""
        with self._lock:
            self.examples.clear()
