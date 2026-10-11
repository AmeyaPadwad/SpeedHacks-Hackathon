"""Turn images/pette-spritesheet.png (checkerboard baked in) into assets/pette_sheet.png.

The output is a transparent 3x2 sheet of equally sized cells, read row-major by pet/sprites.py.
Run once after changing the source art: python tools/prep_sprites.py
"""
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "images" / "pette-spritesheet.png"
DST = ROOT / "assets" / "pette_sheet.png"

COLS, ROWS = 3, 2
CELL_W, CELL_H = 180, 184
BG_MAX = 42  # checkerboard greys are 24..39; the headphones/face are lighter or tinted


def cell_bands(empty: np.ndarray) -> list[tuple[int, int]]:
    bands, start = [], None
    for i, e in enumerate(empty):
        if not e and start is None:
            start = i
        elif e and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(empty)))
    return bands


def knock_out_background(rgb: np.ndarray) -> np.ndarray:
    """Flood-fill neutral dark pixels reachable from the cell border; return RGBA."""
    h, w, _ = rgb.shape
    a = rgb.astype(int)
    bgish = (a.max(2) <= BG_MAX) & ((a.max(2) - a.min(2)) <= 6)
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        q.extend([(0, x), (h - 1, x)])
    for y in range(h):
        q.extend([(y, 0), (y, w - 1)])
    while q:
        y, x = q.popleft()
        if 0 <= y < h and 0 <= x < w and bgish[y, x] and not seen[y, x]:
            seen[y, x] = True
            q.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
    alpha = np.where(seen, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def main() -> None:
    src = np.array(Image.open(SRC).convert("RGB"))
    fg = ~((src.max(2) <= BG_MAX) & ((src.max(2) - src.min(2)) <= 6))
    xs = cell_bands(fg.sum(0) < 3)
    ys = cell_bands(fg.sum(1) < 3)
    assert len(xs) == COLS and len(ys) == ROWS, (xs, ys)

    sheet = Image.new("RGBA", (CELL_W * COLS, CELL_H * ROWS), (0, 0, 0, 0))
    for r, (y0, y1) in enumerate(ys):
        for c, (x0, x1) in enumerate(xs):
            rgba = Image.fromarray(knock_out_background(src[y0:y1, x0:x1]), "RGBA")
            ox = c * CELL_W + (CELL_W - rgba.width) // 2
            oy = r * CELL_H + (CELL_H - rgba.height)  # feet on the cell floor
            sheet.alpha_composite(rgba, (ox, oy))
    DST.parent.mkdir(exist_ok=True)
    sheet.save(DST)
    print(f"wrote {DST} {sheet.size}")


if __name__ == "__main__":
    main()
