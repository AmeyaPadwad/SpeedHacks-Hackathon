"""Module 2: context. The active-file snapshot from the VS Code extension plus its uncommitted git changes.

No screenshots: messages carry the code itself and a summary of what changed.
"""
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ACTIVE_FILE = Path(os.environ.get("FLOWSTATE_ACTIVE_FILE") or Path.home() / ".flowstate_current.json")
MAX_FILE_CHARS = 12_000  # keep prompts bounded; the model gets the head of very large files
MAX_DIFF_CHARS = 8_000
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
    active_file: ActiveFile | None
    changes: str = ""   # unified `git diff HEAD` of the active file; "" if clean, untracked handled as new


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


def git_changes(af: ActiveFile | None, run=subprocess.run) -> str:
    """Uncommitted changes to the active file as a unified diff; "" when clean, not a repo, or git is missing."""
    if af is None or not af.path:
        return ""
    p = Path(af.path)
    if not p.parent.is_dir():
        return ""

    def git(*args) -> str | None:
        try:
            r = run(["git", *args], cwd=p.parent, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    diff = git("diff", "--no-color", "--no-ext-diff", "HEAD", "--", p.name)
    if diff is None:
        return ""
    if not diff.strip() and (git("status", "--porcelain", "--", p.name) or "").startswith("??"):
        body = "\n".join("+" + line for line in af.content.splitlines())
        diff = f"new file {p.name} (untracked)\n{body}"
    return diff.strip()[:MAX_DIFF_CHARS]


def capture() -> Context:
    """Safe off the Qt thread: just file reads and a git call."""
    af = read_active_file()
    return Context(af, git_changes(af))
