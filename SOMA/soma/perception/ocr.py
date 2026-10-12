"""OCR engines producing normalized, redacted :class:`OCRElement` lists.

* :class:`RapidOCREngine` wraps the optional ``rapidocr-onnxruntime`` package
  (lazy import; CPU ONNX models, no network at inference time).
* :class:`SyntheticOCREngine` returns the ground truth attached by
  :mod:`soma.perception.synthetic` (exact, instant; for tests and training).
* :class:`NullOCREngine` returns nothing (OCR unavailable or disabled).
* :class:`FallbackOCREngine` tries engines in order.

All engines pass recognized text through :func:`redact_text` before returning.
:func:`get_ocr_engine` picks the best available engine.
"""

from __future__ import annotations

import dataclasses
import random
import threading
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from soma.capture.redaction import redact_text
from soma.perception.synthetic import ground_truth_for
from soma.types import OCRElement


@runtime_checkable
class OCREngine(Protocol):
    """Recognize text lines in an ``[H,W,3]`` uint8 RGB image."""

    name: str

    def recognize(self, image: np.ndarray, window_id: str = "",
                  timestamp_ms: int = 0) -> list[OCRElement]:
        """Return elements with normalized ``xyxy`` boxes in ``[0,1]``."""
        ...


class OCRUnavailableError(RuntimeError):
    """The requested OCR backend is not installed."""


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _finalize(elements: Sequence[OCRElement]) -> list[OCRElement]:
    """Redact text and drop empty elements."""
    out = []
    for e in elements:
        text = redact_text(e.text).strip()
        if text:
            out.append(dataclasses.replace(e, text=text))
    return out


class NullOCREngine:
    """OCR disabled: always returns ``[]``."""

    name = "null"

    def recognize(self, image: np.ndarray, window_id: str = "",
                  timestamp_ms: int = 0) -> list[OCRElement]:
        return []


class SyntheticOCREngine:
    """Reads ground truth attached by :func:`synthetic.render_scenario`.

    Args:
        keep_roles: keep ground-truth ``ui_role`` (default drops it, like real OCR,
            so the screen parser's heuristics are exercised).
        drop_prob: probability of dropping each element (robustness noise).
        jitter: max absolute box jitter in normalized units.
        seed: RNG seed for the noise.
    """

    name = "synthetic"

    def __init__(self, keep_roles: bool = False, drop_prob: float = 0.0, jitter: float = 0.0,
                 seed: int = 0) -> None:
        self.keep_roles = keep_roles
        self.drop_prob = drop_prob
        self.jitter = jitter
        self._rng = random.Random(seed)

    def available_for(self, image: np.ndarray) -> bool:
        return ground_truth_for(image) is not None

    def recognize(self, image: np.ndarray, window_id: str = "",
                  timestamp_ms: int = 0) -> list[OCRElement]:
        truth = ground_truth_for(image) or []
        out: list[OCRElement] = []
        for e in truth:
            if self.drop_prob and self._rng.random() < self.drop_prob:
                continue
            box = e.bbox_xyxy_normalized
            if self.jitter:
                j = self.jitter
                box = tuple(_clip01(v + self._rng.uniform(-j, j)) for v in box)  # type: ignore[assignment]
                x0, y0, x1, y1 = box
                box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            out.append(OCRElement(text=e.text, bbox_xyxy_normalized=box, confidence=e.confidence,
                                  source="synthetic", ui_role=e.ui_role if self.keep_roles else None,
                                  window_id=window_id or e.window_id, timestamp_ms=timestamp_ms))
        return _finalize(out)


def rapidocr_available() -> bool:
    """True when ``rapidocr_onnxruntime`` can be imported."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class RapidOCREngine:
    """``rapidocr-onnxruntime`` adapter (lazy-loaded, thread-safe).

    Args:
        min_confidence: drop lines below this recognition score.
        max_side: downscale images whose longest side exceeds this (speed).
    """

    name = "rapidocr"

    def __init__(self, min_confidence: float = 0.5, max_side: int = 2048) -> None:
        self.min_confidence = min_confidence
        self.max_side = max_side
        self._engine: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except Exception as exc:  # noqa: BLE001
                raise OCRUnavailableError("rapidocr-onnxruntime is not installed") from exc
            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image: np.ndarray, window_id: str = "",
                  timestamp_ms: int = 0) -> list[OCRElement]:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.size == 0:
            return []
        h, w = arr.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_side:
            from PIL import Image

            scale = self.max_side / max(h, w)
            arr = np.asarray(Image.fromarray(np.ascontiguousarray(arr[:, :, :3])).resize(
                (max(1, int(w * scale)), max(1, int(h * scale)))))
        bgr = np.ascontiguousarray(arr[:, :, 2::-1])     # rapidocr expects BGR ndarrays
        sh, sw = bgr.shape[:2]
        with self._lock:
            result, _ = self._load()(bgr)
        out: list[OCRElement] = []
        for box, text, score in result or []:
            if float(score) < self.min_confidence:
                continue
            pts = np.asarray(box, dtype=np.float32)
            x0, y0 = pts[:, 0].min() / sw, pts[:, 1].min() / sh
            x1, y1 = pts[:, 0].max() / sw, pts[:, 1].max() / sh
            out.append(OCRElement(text=str(text),
                                  bbox_xyxy_normalized=(_clip01(x0), _clip01(y0), _clip01(x1), _clip01(y1)),
                                  confidence=float(score), source="ocr", ui_role=None,
                                  window_id=window_id, timestamp_ms=timestamp_ms))
        return _finalize(out)


class FallbackOCREngine:
    """Use synthetic ground truth when the image has it, else ``primary``.

    If ``primary`` raises, the result is ``[]`` and ``last_error`` is set, so a
    broken OCR backend never crashes the capture loop.
    """

    def __init__(self, primary: OCREngine, synthetic: SyntheticOCREngine | None = None) -> None:
        self.primary = primary
        self.synthetic = synthetic or SyntheticOCREngine()
        self.name = f"auto({primary.name})"
        self.last_error: str = ""

    def recognize(self, image: np.ndarray, window_id: str = "",
                  timestamp_ms: int = 0) -> list[OCRElement]:
        if self.synthetic.available_for(image):
            return self.synthetic.recognize(image, window_id, timestamp_ms)
        try:
            self.last_error = ""
            return self.primary.recognize(image, window_id, timestamp_ms)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []


def get_ocr_engine(name: str = "auto", **kwargs: Any) -> OCREngine:
    """Return an OCR engine by name.

    ``"auto"`` -> synthetic ground truth when present, else RapidOCR if
    installed, else null. ``"rapidocr"`` raises :class:`OCRUnavailableError`
    when missing. ``"synthetic"`` and ``"null"`` always work.
    """
    key = (name or "auto").lower()
    if key == "null":
        return NullOCREngine()
    if key == "synthetic":
        return SyntheticOCREngine(**kwargs)
    if key == "rapidocr":
        if not rapidocr_available():
            raise OCRUnavailableError("rapidocr-onnxruntime is not installed")
        return RapidOCREngine(**kwargs)
    if key == "auto":
        primary: OCREngine = RapidOCREngine(**kwargs) if rapidocr_available() else NullOCREngine()
        return FallbackOCREngine(primary)
    raise ValueError(f"unknown OCR engine {name!r}")
