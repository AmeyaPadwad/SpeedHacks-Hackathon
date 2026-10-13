"""Shared, framework-light data contracts used across SOMA subpackages.

Every subpackage (capture, perception, memory, policy, learning, runtime, ui)
exchanges these types. Keep them small, typed and free of heavy imports so any
module can depend on them. Tensor contracts for the neural model live in
``soma.models.batch``.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def new_id(prefix: str = "") -> str:
    """Short random identifier, optionally prefixed (``"sug_3f2a..."``)."""
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def now_ms() -> int:
    """Wall-clock time in milliseconds."""
    return int(time.time() * 1000)


# ---------------------------------------------------------------- capture


@dataclass
class WindowInfo:
    """Foreground window metadata. No pixels, no text content."""

    app_name: str            # process / executable name, e.g. "Code.exe"
    title: str               # window title (may be redacted)
    window_id: str = ""
    pid: int = 0
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h in screen px


@dataclass
class ScreenFrame:
    """One in-memory screenshot. Never persisted by default.

    ``image`` is an ``uint8`` array shaped ``[H, W, 3]`` (RGB).
    ``version`` increases monotonically per capture so suggestions can be
    invalidated when the screen changes.
    """

    image: np.ndarray
    window: WindowInfo
    version: int
    timestamp_ms: int = field(default_factory=now_ms)
    redacted_regions: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return int(self.image.shape[1]), int(self.image.shape[0])


# ---------------------------------------------------------------- perception


@dataclass
class OCRElement:
    """One recognized text element with normalized ``xyxy`` geometry."""

    text: str
    bbox_xyxy_normalized: tuple[float, float, float, float]
    confidence: float = 1.0
    source: str = "ocr"              # "ocr" | "accessibility" | "synthetic"
    ui_role: str | None = None       # "button", "text", "error", ... when known
    window_id: str = ""
    timestamp_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox_xyxy_normalized": list(self.bbox_xyxy_normalized),
            "confidence": self.confidence,
            "source": self.source,
            "ui_role": self.ui_role,
            "window_id": self.window_id,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class ScreenObservation:
    """Perceived screen state: metadata + OCR/UI elements for one frame version."""

    window: WindowInfo
    elements: list[OCRElement]
    version: int
    timestamp_ms: int = field(default_factory=now_ms)

    @property
    def text(self) -> str:
        return "\n".join(e.text for e in self.elements)


# ---------------------------------------------------------------- memory


class SourceType(str, enum.Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    PREFERENCE = "preference"
    FEEDBACK = "feedback"
    UI_EVENT = "ui-event"


@dataclass
class MemoryItem:
    """A permissioned memory record. Raw files stay at ``source_uri``."""

    id: str
    source_type: SourceType
    user_approved: bool
    text_summary: str = ""
    source_uri: str | None = None
    tags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None
    embedding_ref: str | None = None


@dataclass
class RetrievedMemory:
    """A memory returned by retrieval, with its embedding and score."""

    item: MemoryItem
    embedding: np.ndarray            # float32 [D]
    score: float


# ---------------------------------------------------------------- policy


class ActionType(str, enum.Enum):
    """Bounded action vocabulary. Index 0 of policy logits is NO_SUGGESTION."""

    NO_SUGGESTION = "NO_SUGGESTION"
    EXPLAIN_ERROR = "EXPLAIN_ERROR"
    OPEN_RELEVANT_FILE = "OPEN_RELEVANT_FILE"
    COMPARE_WITH_PRIOR_WORK = "COMPARE_WITH_PRIOR_WORK"
    SUGGEST_NEXT_STEP = "SUGGEST_NEXT_STEP"
    ASK_CLARIFYING_QUESTION = "ASK_CLARIFYING_QUESTION"


ACTION_TYPES: list[ActionType] = list(ActionType)


@dataclass
class Candidate:
    """A proposed (never executed) action from the candidate generator."""

    action: ActionType
    text: str                                    # human-readable suggestion
    target: OCRElement | None = None             # UI target on screen, if any
    memory_ref: str | None = None                # MemoryItem.id, if any
    file_ref: str | None = None                  # only authorized files
    evidence: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("cand_"))
    template: str = ""                           # context template id (soma.policy.context_candidates)
    line: int | None = None                      # line in ``file_ref``, when known (not part of the key)

    @property
    def key(self) -> str:
        """Stable de-duplication key (action + target text + refs)."""
        tgt = self.target.text if self.target else ""
        key = f"{self.action.value}|{tgt}|{self.memory_ref or ''}|{self.file_ref or ''}"
        return f"{key}|{self.template}" if self.template else key


@dataclass
class Suggestion:
    """What the UI may show. ``candidate is None`` means stay silent."""

    candidate: Candidate | None
    confidence: float
    screen_version: int
    reason: str = ""
    distribution: dict[str, float] = field(default_factory=dict)  # label -> prob
    id: str = field(default_factory=lambda: new_id("sug_"))
    created_ms: int = field(default_factory=now_ms)

    @property
    def is_silent(self) -> bool:
        return self.candidate is None


class FeedbackOutcome(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    IGNORED = "ignored"


@dataclass
class FeedbackEvent:
    suggestion_id: str
    outcome: FeedbackOutcome
    consented_for_training: bool = False
    context_ref: str | None = None
    edited_text: str | None = None
    id: str = field(default_factory=lambda: new_id("fb_"))
    observed_at: str = ""
