"""Screen CV (local OpenCV): find error-looking regions in a screenshot and crop to them.

Red squiggles, red error text and failed-test output share a saturated red hue in common IDE,
terminal and browser themes. This is a heuristic locator, not OCR: it decides *where* to look
and lets the vision model and coding agent see a focused crop instead of the whole desktop.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtGui import QImage

MIN_REGION_AREA = 120           # px, after merging; drops icons and stray pixels
PAD = 60                        # context around the error region
MAX_EDGE = 1280
JPEG_QUALITY = 80


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int
    red_px: int


@dataclass
class ScreenFindings:
    regions: list[Region]         # largest first
    red_fraction: float           # share of the screen that is error-red
    crop_jpeg: bytes | None       # focused crop around the strongest regions, None if nothing found
    width: int
    height: int

    @property
    def has_errors(self) -> bool:
        return bool(self.regions)

    def fingerprint(self, cell: int = 80) -> str:
        """Coarse position signature so the same on-screen error dedupes across frames."""
        return ",".join(sorted(f"{r.x // cell}:{r.y // cell}" for r in self.regions[:3]))


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, : w * 3]
    return cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)


def error_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, (0, 110, 120), (9, 255, 255))
    high = cv2.inRange(hsv, (170, 110, 120), (180, 255, 255))
    return cv2.bitwise_or(low, high)


MIN_ASPECT = 2.2        # errors are text lines / squiggles: wide and short
MAX_DENSITY = 0.35      # badges, buttons and icons are solid blobs; text and squiggles are sparse


def find_regions(bgr: np.ndarray, exclude: list[tuple[int, int, int, int]] = ()) -> tuple[list[Region], float]:
    """`exclude` is a list of (x, y, w, h) screen rects to ignore, e.g. the taskbar and Pette itself."""
    mask = error_mask(bgr)
    for x, y, w, h in exclude:
        mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 0
    red_fraction = float(np.count_nonzero(mask)) / mask.size
    # Merge squiggle dashes and glyphs on one line into a blob, then join nearby lines.
    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_REGION_AREA or h > bgr.shape[0] * 0.6:  # a huge red block is a banner/photo, not an error
            continue
        red = int(np.count_nonzero(mask[y:y + h, x:x + w]))
        if red < 40 or w / h < MIN_ASPECT or red / (w * h) > MAX_DENSITY:
            continue
        regions.append(Region(int(x), int(y), int(w), int(h), red))
    regions.sort(key=lambda r: r.red_px, reverse=True)
    return regions, red_fraction


def crop_around(bgr: np.ndarray, regions: list[Region], top: int = 3) -> np.ndarray:
    rs = regions[:top]
    x0 = max(0, min(r.x for r in rs) - PAD)
    y0 = max(0, min(r.y for r in rs) - PAD)
    x1 = min(bgr.shape[1], max(r.x + r.w for r in rs) + PAD)
    y1 = min(bgr.shape[0], max(r.y + r.h for r in rs) + PAD)
    # Extend right to the screen edge: error text usually continues past the red squiggle.
    x1 = max(x1, min(bgr.shape[1], x0 + bgr.shape[1] // 2))
    return bgr[y0:y1, x0:x1]


def encode(bgr: np.ndarray) -> bytes:
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_EDGE:
        s = MAX_EDGE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def analyze(bgr: np.ndarray, exclude: list[tuple[int, int, int, int]] = ()) -> ScreenFindings:
    regions, frac = find_regions(bgr, exclude)
    crop = encode(crop_around(bgr, regions)) if regions else None
    return ScreenFindings(regions, frac, crop, bgr.shape[1], bgr.shape[0])
