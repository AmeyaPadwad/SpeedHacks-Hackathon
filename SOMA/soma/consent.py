"""Independent, explicit opt-in toggles. Everything defaults to OFF.

Every privacy-sensitive subsystem must check its own scope here before doing
anything: capture, app monitoring, folder indexing, feedback storage, training,
and remote inference. Pausing stops capture instantly without forgetting
the other grants.
"""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path


class Scope(str, enum.Enum):
    SCREEN_CAPTURE = "screen_capture"
    APP_MONITORING = "app_monitoring"
    FOLDER_INDEXING = "folder_indexing"
    FEEDBACK_STORAGE = "feedback_storage"
    MODEL_TRAINING = "model_training"
    REMOTE_INFERENCE = "remote_inference"


class ConsentError(PermissionError):
    """Raised when an operation runs without the matching opt-in."""


@dataclass
class ConsentState:
    granted: set[Scope] = field(default_factory=set)
    paused: bool = False
    allowed_folders: list[str] = field(default_factory=list)


class ConsentManager:
    """Thread-safe consent store, optionally persisted as JSON (no content)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        self._state = ConsentState()
        if self._path and self._path.exists():
            self._load()

    # -- queries
    def allowed(self, scope: Scope) -> bool:
        with self._lock:
            if scope is Scope.SCREEN_CAPTURE and self._state.paused:
                return False
            return scope in self._state.granted

    def require(self, scope: Scope) -> None:
        if not self.allowed(scope):
            raise ConsentError(f"Consent for '{scope.value}' has not been granted.")

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._state.paused

    @property
    def allowed_folders(self) -> list[str]:
        with self._lock:
            return list(self._state.allowed_folders)

    # -- mutations
    def grant(self, scope: Scope) -> None:
        with self._lock:
            self._state.granted.add(scope)
        self._save()

    def revoke(self, scope: Scope) -> None:
        with self._lock:
            self._state.granted.discard(scope)
            if scope is Scope.FOLDER_INDEXING:
                self._state.allowed_folders.clear()
        self._save()

    def pause(self) -> None:
        with self._lock:
            self._state.paused = True
        self._save()

    def resume(self) -> None:
        with self._lock:
            self._state.paused = False
        self._save()

    def add_folder(self, folder: str | Path) -> None:
        """Allow one user-selected folder. Refuses a bare home directory or drive root."""
        p = Path(folder).expanduser().resolve()
        if p == Path.home().resolve() or p.parent == p:
            raise ConsentError("Refusing to index an entire home directory or drive root.")
        with self._lock:
            if str(p) not in self._state.allowed_folders:
                self._state.allowed_folders.append(str(p))
        self._save()

    def remove_folder(self, folder: str | Path) -> None:
        p = str(Path(folder).expanduser().resolve())
        with self._lock:
            self._state.allowed_folders = [f for f in self._state.allowed_folders if f != p]
        self._save()

    def path_allowed(self, path: str | Path) -> bool:
        """True when indexing is granted and ``path`` lies inside an allowed folder."""
        if not self.allowed(Scope.FOLDER_INDEXING):
            return False
        p = Path(path).expanduser().resolve()
        for f in self.allowed_folders:
            try:
                p.relative_to(f)
                return True
            except ValueError:
                continue
        return False

    # -- persistence
    def _save(self) -> None:
        if not self._path:
            return
        with self._lock:
            data = {"granted": sorted(s.value for s in self._state.granted),
                    "paused": self._state.paused,
                    "allowed_folders": self._state.allowed_folders}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._state = ConsentState(
            granted={Scope(s) for s in data.get("granted", []) if s in Scope._value2member_map_},
            paused=bool(data.get("paused", False)),
            allowed_folders=list(data.get("allowed_folders", [])),
        )
