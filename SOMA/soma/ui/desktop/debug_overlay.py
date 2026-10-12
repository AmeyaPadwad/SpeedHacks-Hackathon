"""Debug tab: the latest frame, downscaled, with role-coloured element boxes.

The image lives only in this widget's pixmap (in memory). It is never written
to disk and is cleared on pause.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from soma.perception import draw_observation
from soma.types import ScreenObservation

MAX_W = 960
PLACEHOLDER = "No frame. The debug view is empty while SOMA is paused or off."


def to_qimage(rgb: np.ndarray) -> QImage:
    """Copy an ``[H, W, 3]`` uint8 array into a QImage that owns its pixels."""
    arr = np.ascontiguousarray(np.asarray(rgb)[:, :, :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def blank_canvas(obs: ScreenObservation, default: tuple[int, int] = (640, 400)) -> np.ndarray:
    """A neutral canvas sized like the window, used when the runtime exposes no frame."""
    _, _, w, h = obs.window.bounds
    w, h = (w, h) if w > 0 and h > 0 else default
    scale = min(1.0, MAX_W / max(1, w))
    return np.full((max(1, int(h * scale)), max(1, int(w * scale)), 3), 240, dtype=np.uint8)


def downscale(image: np.ndarray, max_w: int = MAX_W) -> np.ndarray:
    """Cheap stride downscale so boxes are drawn on a small copy, not the full frame."""
    arr = np.asarray(image)
    step = max(1, int(np.ceil(arr.shape[1] / max_w)))
    return np.ascontiguousarray(arr[::step, ::step, :3])


class DebugOverlay(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = QLabel(PLACEHOLDER)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.view.setMinimumHeight(200)
        self.caption = QLabel("")
        self.caption.setObjectName("muted")
        lay = QVBoxLayout(self)
        note = QLabel("Local inspection only: the frame is kept in memory and never saved.")
        note.setObjectName("muted")
        lay.addWidget(note)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.caption)
        self._pixmap: QPixmap | None = None

    @property
    def has_image(self) -> bool:
        return self._pixmap is not None

    @property
    def pixmap_size(self) -> tuple[int, int]:
        return (0, 0) if self._pixmap is None else (self._pixmap.width(), self._pixmap.height())

    def show_frame(self, image: Any, obs: ScreenObservation) -> None:
        """Draw boxes on ``image`` (or a blank canvas when None) and show it downscaled."""
        base = downscale(image) if image is not None else blank_canvas(obs)
        if base.ndim != 3 or base.shape[2] < 3:
            return
        drawn = draw_observation(base, obs, labels=True)
        self._pixmap = QPixmap.fromImage(to_qimage(drawn))
        self._rescale()
        src = "frame" if image is not None else "layout only (no frame available)"
        self.caption.setText(f"v{obs.version} · {len(obs.elements)} elements · {src}")

    def clear(self) -> None:
        self._pixmap = None
        self.view.clear()
        self.view.setText(PLACEHOLDER)
        self.caption.setText("")

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        size = self.view.size()
        if size.width() < 10 or size.height() < 10:
            self.view.setPixmap(self._pixmap)
            return
        self.view.setPixmap(self._pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale()
