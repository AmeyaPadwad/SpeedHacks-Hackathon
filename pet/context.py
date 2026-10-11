"""Module 2: context capture. One-shot screenshot plus the active-file snapshot written by the VS Code extension."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QPixmap

ACTIVE_FILE = Path(os.environ.get("FLOWSTATE_ACTIVE_FILE") or Path.home() / ".flowstate_current.json")
MAX_EDGE = 1280
JPEG_QUALITY = 75
MAX_FILE_CHARS = 12_000  # keep prompts bounded; the model gets the head of very large files


@dataclass
class ActiveFile:
    path: str
    language: str
    content: str
    selection: str = ""


@dataclass
class Context:
    screenshot_jpeg: bytes | None
    active_file: ActiveFile | None


def grab_screen() -> bytes | None:
    """Capture the primary screen once as a downscaled in-memory JPEG. Must run on the Qt main thread."""
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return None
    pm = screen.grabWindow(0)
    if pm.isNull():
        return None
    return encode_jpeg(pm)


def encode_jpeg(pm: QPixmap) -> bytes:
    if max(pm.width(), pm.height()) > MAX_EDGE:
        pm = pm.scaled(MAX_EDGE, MAX_EDGE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pm.save(buf, "JPEG", JPEG_QUALITY)
    return bytes(buf.data())


def read_active_file(path: Path = ACTIVE_FILE) -> ActiveFile | None:
    """Read the snapshot the companion VS Code extension keeps up to date; None if absent or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ActiveFile(
            path=str(data["path"]),
            language=str(data.get("language", "")),
            content=str(data.get("content", ""))[:MAX_FILE_CHARS],
            selection=str(data.get("selection", ""))[:MAX_FILE_CHARS],
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def capture() -> Context:
    return Context(screenshot_jpeg=grab_screen(), active_file=read_active_file())
