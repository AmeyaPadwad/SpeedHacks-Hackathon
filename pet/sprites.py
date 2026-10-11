"""Pette spritesheet loader: assets/pette_sheet.png is a 3x2 grid of equal cells, row-major."""
import sys
from pathlib import Path

from PySide6.QtGui import QPixmap

COLS, ROWS = 3, 2

# Frame indices into the sheet, by pose.
IDLE_A, LOOK, HAPPY, TYPING, WAVE, IDLE_B = range(6)

ANIMATIONS = {
    "idle": [IDLE_A, IDLE_A, IDLE_B, IDLE_A, TYPING, IDLE_A],
    "thinking": [TYPING, LOOK, TYPING, IDLE_B],
    "speaking": [HAPPY, IDLE_A, HAPPY, WAVE],
    "sleeping": [HAPPY],
}


def asset_path(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets" / name


class SpriteSheet:
    def __init__(self, path: Path | None = None):
        sheet = QPixmap(str(path or asset_path("pette_sheet.png")))
        if sheet.isNull():
            raise FileNotFoundError("assets/pette_sheet.png missing; run tools/prep_sprites.py")
        self.cell_w, self.cell_h = sheet.width() // COLS, sheet.height() // ROWS
        self.frames = [
            sheet.copy((i % COLS) * self.cell_w, (i // COLS) * self.cell_h, self.cell_w, self.cell_h)
            for i in range(COLS * ROWS)
        ]
