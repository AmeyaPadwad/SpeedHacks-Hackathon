"""Image preprocessing for the screen encoder.

* :func:`frame_to_tensor` letterbox-resizes an ``[H,W,3]`` uint8 RGB frame to a
  square ``[3,size,size]`` float tensor (ImageNet-normalized) and returns the
  :class:`LetterboxTransform` so boxes can be mapped between the original frame
  and the model input.
* :func:`change_ratio` is a cheap downsampled frame difference used for
  event-triggered sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry of a letterbox resize from ``src_size`` (w,h) to a ``dst`` square.

    The image is scaled by ``scale`` and placed at offset ``(pad_x, pad_y)``
    (pixels in the destination). Boxes are normalized ``xyxy``.
    """

    src_size: tuple[int, int]
    dst_size: int
    scale: float
    pad_x: int
    pad_y: int
    new_size: tuple[int, int]   # scaled content (w,h) inside the destination

    def src_to_dst(self, box: Sequence[float]) -> Box:
        """Map a normalized box on the source frame to the model input."""
        sw, sh = self.src_size
        d = float(self.dst_size)
        x0, y0, x1, y1 = box
        return ((x0 * sw * self.scale + self.pad_x) / d, (y0 * sh * self.scale + self.pad_y) / d,
                (x1 * sw * self.scale + self.pad_x) / d, (y1 * sh * self.scale + self.pad_y) / d)

    def dst_to_src(self, box: Sequence[float]) -> Box:
        """Map a normalized box on the model input back to the source frame."""
        sw, sh = self.src_size
        d = float(self.dst_size)
        x0, y0, x1, y1 = box

        def fx(v: float) -> float:
            return min(1.0, max(0.0, (v * d - self.pad_x) / (self.scale * sw)))

        def fy(v: float) -> float:
            return min(1.0, max(0.0, (v * d - self.pad_y) / (self.scale * sh)))

        return fx(x0), fy(y0), fx(x1), fy(y1)


def letterbox_geometry(src_size: tuple[int, int], size: int) -> LetterboxTransform:
    """Compute the letterbox transform for a ``(w,h)`` frame into ``size``²."""
    w, h = max(1, int(src_size[0])), max(1, int(src_size[1]))
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return LetterboxTransform(src_size=(w, h), dst_size=size, scale=scale,
                              pad_x=(size - nw) // 2, pad_y=(size - nh) // 2, new_size=(nw, nh))


def frame_to_tensor(image: np.ndarray, size: int, normalize: bool = True,
                    ) -> tuple[torch.Tensor, LetterboxTransform]:
    """Letterbox an ``[H,W,3]`` uint8 RGB (or ``[H,W]`` gray) image to ``[3,size,size]``.

    Padding is black before normalization. Returns ``(tensor, transform)``.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected [H,W,3] image, got shape {arr.shape}")
    arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    tf = letterbox_geometry((w, h), size)
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)
    nw, nh = tf.new_size
    x = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False,
                      antialias=True).squeeze(0).clamp_(0.0, 1.0)
    out = torch.zeros(3, size, size, dtype=torch.float32)
    out[:, tf.pad_y:tf.pad_y + nh, tf.pad_x:tf.pad_x + nw] = x
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        out = (out - mean) / std
    return out, tf


def _thumb(image: np.ndarray, grid: int) -> np.ndarray:
    """Block-average a frame to a ``grid x grid`` grayscale float thumbnail."""
    arr = np.asarray(image)
    # Stride-subsample very large frames first (keeps ~4 samples per cell edge).
    step = max(1, min(arr.shape[0], arr.shape[1]) // (grid * 4))
    if step > 1:
        arr = arr[::step, ::step]
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    h, w = arr.shape[:2]
    grid = max(1, min(grid, h, w))
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    # cumulative-sum box filter: cheap and exact block means
    cs = np.pad(arr, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    tot = cs[ys[1:]][:, xs[1:]] - cs[ys[:-1]][:, xs[1:]] - cs[ys[1:]][:, xs[:-1]] + cs[ys[:-1]][:, xs[:-1]]
    area = np.outer(np.maximum(np.diff(ys), 1), np.maximum(np.diff(xs), 1)).astype(np.float32)
    return tot / area


def change_ratio(a: np.ndarray | None, b: np.ndarray | None, grid: int = 64,
                 pixel_threshold: float = 8.0) -> float:
    """Fraction of cells in a ``grid x grid`` downsampled view that changed.

    A cell counts as changed when its mean intensity differs by more than
    ``pixel_threshold`` (0-255 scale). Different frame sizes or a missing frame
    count as a full change (1.0).
    """
    if a is None or b is None:
        return 1.0
    if np.asarray(a).shape != np.asarray(b).shape:
        return 1.0
    ta, tb = _thumb(a, grid), _thumb(b, grid)
    return float((np.abs(ta - tb) > pixel_threshold).mean())
