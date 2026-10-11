"""Module 2: context capture. One-shot screenshot (messages only) plus the active-file snapshot from the VS Code extension."""
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
MAX_DIAGNOSTICS = 20

@dataclass(frozen=True)
class Diagnostic:
    line: int
    severity: str
    message: str
    source: str = ""
    code: str = ""

    def __str__(self):
        tag = f" [{self.source}{' ' + self.code if self.code else ''}]" if self.source or self.code else ""
        return f"line {self.line} {self.severity}{tag}: {self.message}"


@dataclass
class ActiveFile:
    path: str
    language: str
    content: str
    selection: str = ""
    workspace: str = ""
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")


@dataclass
class Context:
    screenshot_jpeg: bytes | None
    active_file: ActiveFile | None


def grab_screen() -> bytes | None:
    """Capture the primary screen once as a downscaled JPEG for the message drafter.

    Must run on the Qt main thread. The full-resolution frame is dropped before returning.
    """
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return None
    pm = screen.grabWindow(0)
    return None if pm.isNull() else encode_jpeg(pm)


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
        diags = tuple(
            Diagnostic(int(d.get("line", 0)), str(d.get("severity", "")), str(d.get("message", ""))[:500],
                       str(d.get("source", "")), str(d.get("code", "")))
            for d in data.get("diagnostics", [])[:MAX_DIAGNOSTICS] if isinstance(d, dict)
        )
        return ActiveFile(
            path=str(data["path"]),
            language=str(data.get("language", "")),
            content=str(data.get("content", ""))[:MAX_FILE_CHARS],
            selection=str(data.get("selection", ""))[:MAX_FILE_CHARS],
            workspace=str(data.get("workspace", "")),
            diagnostics=diags,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def capture() -> Context:
    """Message flow: screenshot + active file. The investigate flow uses read_active_file() only."""
    return Context(screenshot_jpeg=grab_screen(), active_file=read_active_file())
