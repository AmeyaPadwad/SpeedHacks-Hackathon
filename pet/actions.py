"""What "Do it" may do: open a file SOMA pointed at, or copy the suggestion. Nothing else.

Opening never runs anything: a file goes to VS Code (`code -g path:line`) when its CLI is on PATH, an image
opens in its viewer, and anything else is only selected in Explorer. `os.startfile` is never used on a
non-image (a .py, .bat or .ps1 would run). Only files that exist and that SOMA reported inside your
allowed folders reach this module.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


class ActionError(Exception):
    """The file is gone or can't be opened; shown in the bubble."""


def open_file(path: str, line: int | None = None, *, which=shutil.which, popen=subprocess.Popen,
              startfile=getattr(os, "startfile", None)) -> str:
    """Open `path` (at `line` when known). Returns a short description of what happened."""
    p = Path(path)
    if not p.is_file():
        raise ActionError(f"{p.name or path} isn't there any more.")
    code = which("code")
    if code and p.suffix.lower() not in IMAGE_EXT:
        target = f"{p}:{int(line)}" if line else str(p)
        popen([code, "-g", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Opened {p.name}" + (f" at line {int(line)}" if line else "") + " in VS Code."
    if p.suffix.lower() in IMAGE_EXT and startfile is not None:
        startfile(str(p))
        return f"Opened {p.name}."
    if sys.platform == "win32":
        popen(["explorer", f"/select,{p}"])
        return f"Showed {p.name} in Explorer."
    raise ActionError("No way to open files here.")


def copy_text(text: str, clipboard) -> str:
    clipboard(text)
    return "Copied. Paste it to your coding agent or notes."
