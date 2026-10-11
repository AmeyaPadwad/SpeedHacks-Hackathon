"""Module 2: context capture. One-shot screenshot plus the active-file snapshot written by the VS Code extension."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QPixmap

ACTIVE_FILE = Path(os.environ.get("FLOWSTATE_ACTIVE_FILE") or Path.home() / ".flowstate_current.json")
MAX_EDGE = 1280
JPEG_QUALITY = 75
MAX_FILE_CHARS = 12_000  # keep prompts bounded; the model gets the head of very large files
MAX_DIAGNOSTICS = 20

if TYPE_CHECKING:
    from .screen_cv import ScreenFindings


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
    screen: "ScreenFindings | None" = None   # local OpenCV findings for the same frame


def grab_screen() -> tuple[bytes | None, "ScreenFindings | None"]:
    """Capture the primary screen once: downscaled JPEG for the model plus local screen-CV findings.

    Must run on the Qt main thread. The full-resolution frame is dropped before returning.
    """
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return None, None
    pm = screen.grabWindow(0)
    if pm.isNull():
        return None, None
    return encode_jpeg(pm), find_errors(pm, screen)


def find_errors(pm: QPixmap, screen) -> "ScreenFindings | None":
    try:
        from .screen_cv import analyze, qimage_to_bgr
    except ImportError:  # OpenCV not installed: screen CV is optional
        return None
    g, a = screen.geometry(), screen.availableGeometry()
    r = pm.devicePixelRatio() or 1
    # Ignore the taskbar/docked bars: badges there are not code errors.
    exclude = [(0, int((a.bottom() + 1 - g.top()) * r), int(g.width() * r), int(g.height() * r)),
               (0, 0, int(g.width() * r), int((a.top() - g.top()) * r))]
    return analyze(qimage_to_bgr(pm.toImage()), exclude)


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
    jpeg, findings = grab_screen()
    return Context(screenshot_jpeg=jpeg, active_file=read_active_file(), screen=findings)
