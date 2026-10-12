"""SOMA capture package: consent-gated screen capture, window metadata,
accessibility labels and privacy redaction. Frames stay in memory."""

from soma.capture.accessibility import elements_from_tree, read_accessibility, uia_available
from soma.capture.redaction import (
    contains_secret,
    is_excluded,
    mask_regions,
    redact_elements,
    redact_text,
    sensitive_boxes,
)
from soma.capture.screen import CaptureStats, ScreenCapturer, mss_grab
from soma.capture.window import foreground_window, neutral_window, static_window

__all__ = [
    "CaptureStats", "ScreenCapturer", "mss_grab",
    "foreground_window", "neutral_window", "static_window",
    "read_accessibility", "elements_from_tree", "uia_available",
    "is_excluded", "redact_text", "redact_elements", "contains_secret",
    "sensitive_boxes", "mask_regions",
]
