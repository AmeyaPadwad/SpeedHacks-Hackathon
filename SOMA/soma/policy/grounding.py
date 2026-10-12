"""Target grounding and screen-version staleness checks.

A suggestion that points at a UI element is only valid while that element is
still on screen. Before presenting (or later, before any user-approved
operation) the target is re-verified against the **latest** observation:

* the text must still match (normalized, tolerant of small OCR noise), and
* the box must overlap the original (IoU >= threshold) or sit nearby
  (the element shifted slightly, e.g. after a resize or a scroll of a few px).

Coordinates are normalized ``xyxy`` in [0, 1]; ``ground_point`` maps them to
pixels for a given frame size.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections.abc import Callable

from soma.types import Candidate, OCRElement, ScreenObservation, Suggestion

Box = tuple[float, float, float, float]

_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")
_WS_RE = re.compile(r"\s+")

TEXT_SIMILARITY = 0.9
MAX_CENTER_SHIFT = 0.05


def normalize_text(text: str) -> str:
    """Lowercase, NFKC, no control chars, collapsed whitespace, trimmed punctuation."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _WS_RE.sub(" ", text)
    text = _CTRL_RE.sub("", text).lower().strip()
    return text.strip(" .,:;!?'\"`()[]{}")


def text_matches(a: str, b: str, threshold: float = TEXT_SIMILARITY) -> bool:
    """Normalized equality, or a near-identical string (OCR jitter)."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``xyxy`` boxes (0 for degenerate boxes)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_distance(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(ax - bx, ay - by)


def find_target(
    target: OCRElement,
    latest_obs: ScreenObservation,
    iou_threshold: float = 0.5,
    max_center_shift: float = MAX_CENTER_SHIFT,
) -> OCRElement | None:
    """Best element on ``latest_obs`` matching ``target``, or None."""
    if (target.window_id and latest_obs.window.window_id
            and target.window_id != latest_obs.window.window_id):
        return None
    best: OCRElement | None = None
    best_score = -1.0
    for el in latest_obs.elements:
        if not text_matches(target.text, el.text):
            continue
        overlap = iou(target.bbox_xyxy_normalized, el.bbox_xyxy_normalized)
        shift = center_distance(target.bbox_xyxy_normalized, el.bbox_xyxy_normalized)
        if overlap < iou_threshold and shift > max_center_shift:
            continue
        score = overlap - shift
        if score > best_score:
            best, best_score = el, score
    return best


def verify_target(
    candidate: Candidate,
    latest_obs: ScreenObservation,
    iou_threshold: float = 0.5,
    max_center_shift: float = MAX_CENTER_SHIFT,
) -> bool:
    """True if the candidate has no UI target, or its target is still on screen."""
    if candidate.target is None:
        return True
    return find_target(candidate.target, latest_obs, iou_threshold, max_center_shift) is not None


def ground_point(target: OCRElement | Box, size: tuple[int, int]) -> tuple[int, int]:
    """Pixel center ``(x, y)`` of a normalized target for a frame of ``(width, height)``."""
    box = target.bbox_xyxy_normalized if isinstance(target, OCRElement) else target
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid frame size {size!r}")
    cx, cy = center(box)
    x = int(round(min(max(cx, 0.0), 1.0) * (w - 1)))
    y = int(round(min(max(cy, 0.0), 1.0) * (h - 1)))
    return x, y


def ground_box(target: OCRElement | Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel ``xyxy`` box of a normalized target, clamped to the frame."""
    box = target.bbox_xyxy_normalized if isinstance(target, OCRElement) else target
    w, h = int(size[0]), int(size[1])
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return (int(round(clamp(box[0]) * w)), int(round(clamp(box[1]) * h)),
            int(round(clamp(box[2]) * w)), int(round(clamp(box[3]) * h)))


ChangedFn = Callable[[int, int], bool]


def is_stale(
    suggestion: Suggestion,
    current_version: int,
    changed: bool | ChangedFn | None = None,
    latest_obs: ScreenObservation | None = None,
) -> bool:
    """Whether a suggestion generated on an older screen version is no longer valid.

    Args:
        suggestion: carries the ``screen_version`` it was generated from.
        current_version: version of the latest frame.
        changed: whether the UI changed relevantly since the suggestion's version.
            ``None`` (default) is conservative: any newer version is stale. A bool
            is used as-is; a callable gets ``(suggestion_version, current_version)``.
        latest_obs: if given and the suggestion has a UI target, the target must
            still verify on it.
    """
    if suggestion.screen_version > current_version:
        return True  # from the future: inconsistent, never trust it
    if suggestion.screen_version < current_version:
        if changed is None:
            return True
        relevant = changed(suggestion.screen_version, current_version) if callable(changed) else bool(changed)
        if relevant:
            return True
    if latest_obs is not None and suggestion.candidate is not None:
        return not verify_target(suggestion.candidate, latest_obs)
    return False
