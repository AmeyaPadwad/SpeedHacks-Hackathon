"""Versioned, portable handoff between input collection and agent discovery."""
import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

MAX_FILES = 8
MAX_TEXT = 4_000
MAX_IMAGE_BYTES = 4_900_000


@dataclass(frozen=True)
class WindowContext:
    app: str = ""
    title: str = ""
    window_id: int | None = None


@dataclass(frozen=True)
class FileExcerpt:
    path: str
    start_line: int
    end_line: int
    content: str
    reason: str = ""
    truncated: bool = False

    def context_text(self) -> str:
        header = f"File: {self.path[:500]} (lines {self.start_line}-{self.end_line})\n"
        header += f"Selected because: {self.reason[:300]}\n"
        truncated = self.truncated or len(header) + len(self.content) > MAX_TEXT
        suffix = "\n[Excerpt truncated]" if truncated else ""
        return header + self.content[:MAX_TEXT - len(header) - len(suffix)] + suffix


@dataclass(frozen=True)
class ContextPacket:
    task: str
    transcript: str = ""
    project_root: str = ""
    window: WindowContext = field(default_factory=WindowContext)
    files: tuple[FileExcerpt, ...] = ()
    screenshot_png: bytes | None = None
    warnings: tuple[str, ...] = ()
    session_id: str = field(default_factory=lambda: str(uuid4()))
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = 1
    file_selection_status: str = "unknown"
    file_selection_model: str = ""
    file_selection_provider: str = ""

    def __post_init__(self):
        if self.schema_version != 1 or not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("A version-1 packet requires a nonempty task.")
        if (not isinstance(self.transcript, str) or not isinstance(self.project_root, str)
                or not isinstance(self.window, WindowContext)
                or not all(isinstance(v, str) for v in (self.window.app, self.window.title, self.session_id, self.captured_at))
                or not all(isinstance(w, str) for w in self.warnings)
                or not all(isinstance(f, FileExcerpt) and isinstance(f.path, str) and isinstance(f.content, str)
                           and isinstance(f.reason, str) and type(f.start_line) is int and type(f.end_line) is int
                           and f.start_line > 0 and f.end_line >= f.start_line for f in self.files)):
            raise ValueError("Invalid context packet fields.")
        if len(self.files) > MAX_FILES:
            raise ValueError("A packet supports at most eight file excerpts.")
        if (not isinstance(self.file_selection_status, str)
                or self.file_selection_status not in {"unknown", "not_requested", "selected", "manual", "empty", "needs_clarification", "failed", "excluded"}
                or not isinstance(self.file_selection_model, str)
                or not isinstance(self.file_selection_provider, str)
                or self.file_selection_provider not in {"", "groq", "openai"}):
            raise ValueError("Invalid file selection status.")
        if self.screenshot_png is not None:
            if not isinstance(self.screenshot_png, bytes) or not self.screenshot_png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Screenshot must contain PNG bytes.")
            if len(self.screenshot_png) > MAX_IMAGE_BYTES:
                raise ValueError("Screenshot exceeds the file size limit.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("screenshot_png")
        data["screenshot"] = None if self.screenshot_png is None else {
            "type": "file", "name": "foreground.png", "mimeType": "image/png",
            "data": base64.b64encode(self.screenshot_png).decode("ascii"),
        }
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str):
        """Load an exported packet. Unsupported versions or malformed fields raise ValueError."""
        try:
            data = json.loads(raw)
            screenshot = data.pop("screenshot")
            png = None
            if screenshot is not None:
                if screenshot["mimeType"] != "image/png" or len(screenshot["data"]) > 6_600_000:
                    raise ValueError("Unsupported screenshot.")
                png = base64.b64decode(screenshot["data"], validate=True)
            data["screenshot_png"] = png
            data["window"] = WindowContext(**data["window"])
            data["files"] = tuple(FileExcerpt(**f) for f in data["files"])
            data["warnings"] = tuple(data["warnings"])
            return cls(**data)
        except (TypeError, KeyError, AttributeError, ValueError) as e:
            raise ValueError("Invalid or unsupported context packet.") from e


def to_darwin_search_input(packet: ContextPacket) -> dict:
    """No IO. Search/refinement and execution are owned by the consuming teammate."""
    metadata = {"capturedAt": packet.captured_at, "window": {
                    "app": packet.window.app[:200], "title": packet.window.title[:500],
                    "window_id": packet.window.window_id},
                "warnings": [w[:300] for w in packet.warnings[:6]],
                "fileSelectionStatus": packet.file_selection_status,
                "fileSelectionProvider": packet.file_selection_provider,
                "note": "Saved file excerpts and one window snapshot; no unsaved editor buffer or live terminal cwd."}
    context = [{"type": "text", "text": f.context_text()} for f in packet.files]
    context.append({"type": "text", "text": json.dumps(metadata, ensure_ascii=False)[:MAX_TEXT]})
    if packet.screenshot_png is not None:
        context.append(packet.to_dict()["screenshot"])
    return {"query": packet.task.strip(), "context": context}
