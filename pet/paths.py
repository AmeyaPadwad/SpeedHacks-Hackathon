"""Paths to files shipped next to the code (models/, pet/web/), also inside a PyInstaller bundle."""
import sys
from pathlib import Path


def asset_path(name: str, folder: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / folder / name
