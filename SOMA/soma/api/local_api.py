"""Authenticated, localhost-only HTTP API for local tools (plan section 10).

* Binds to loopback only (``127.0.0.1`` / ``::1`` / ``localhost``); any other
  host is refused at construction.
* A random bearer token is generated per start; it is kept in memory and,
  only when the caller gives a path, written there with owner-only
  permissions. Tokens are compared in constant time.
* Requests with a non-loopback peer or a ``Host`` header that is not a
  loopback name are rejected (DNS-rebinding guard).
* No endpoint executes actions. Feedback, pause/resume, look-now, folder
  allow/remove and memory deletion are the only mutations, and they are the same
  user controls the desktop UI has (a folder is only added when the user picks it).

Endpoints::

    GET    /status            runtime state, consent scopes, telemetry (numbers only)
    GET    /suggestion        latest suggestion (text/evidence only, no raw screen text)
    POST   /feedback          {"suggestion_id", "outcome", "edited_text"?}
    POST   /pause             pause capture now
    POST   /resume            resume capture
    POST   /trigger           look at the screen now (one forced step; consent gates still apply)
    GET    /folders           allowed folders, their memory counts and the indexing state
    POST   /folders           {"path"}: allow one user-chosen folder and index it in the background
    POST   /folders/remove    {"path"}: disallow a folder and forget every memory from it
    GET    /memories          memory-viewer rows
    DELETE /memories/<id>     forget everywhere + verify
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from soma.consent import ConsentError, Scope
from soma.memory.deletion import forget, forget_source, verify_deleted
from soma.memory.export import list_for_viewer
from soma.types import FeedbackOutcome

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
MAX_BODY = 64 * 1024


class LocalAPIError(ValueError):
    """Raised for an unsafe API configuration (e.g. a non-loopback bind)."""


def write_token(path: str | Path, token: str) -> Path:
    """Write ``token`` to ``path`` with owner-only permissions (best effort on Windows)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(p, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, token.encode("ascii"))
    finally:
        os.close(fd)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return p


def _memory_source(runtime: Any, memory_ref: str | None) -> str | None:
    """The file behind a memory, only when it is still an authorized file (so a UI may open it)."""
    store = getattr(getattr(runtime, "memory", None), "store", None)
    if not memory_ref or store is None:
        return None
    item = store.get_item(memory_ref)
    uri = getattr(item, "source_uri", None)
    check = getattr(runtime, "authorized_file", None)
    return uri if uri and callable(check) and check(uri) else None


def suggestion_payload(runtime: Any) -> dict[str, Any]:
    """Latest suggestion as JSON-safe data: no OCR text beyond the suggestion itself.

    ``file`` / ``line`` / ``memory_source`` are paths inside the user's allowed folders only, so a
    companion UI can offer to open them. Nothing is opened or executed here."""
    r = runtime.last_result
    if r is None:
        return {"available": False, "silent": True}
    s = r.suggestion
    out: dict[str, Any] = {
        "available": True, "id": s.id, "silent": s.is_silent, "confidence": round(s.confidence, 4),
        "screen_version": s.screen_version, "frame_version": r.frame_version,
        "distribution": {k: round(v, 4) for k, v in s.distribution.items()},
        "valid": runtime.current_suggestion_valid(r.frame_version), "app": r.window.app_name,
    }
    if s.candidate is not None:
        c = s.candidate
        out.update(action=c.action.value, text=c.text, evidence=list(c.evidence),
                   target_box=list(c.target.bbox_xyxy_normalized) if c.target else None,
                   memory_ref=c.memory_ref, template=c.template, file=c.file_ref, line=c.line,
                   memory_source=_memory_source(runtime, c.memory_ref))
    return out


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"
    protocol_version = "HTTP/1.1"
    server_version = "SOMA-local"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - never log requests
        return

    # -- helpers
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        peer = self.client_address[0]
        if peer not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            self._send(403, {"error": "loopback only"})
            return False
        host = (self.headers.get("Host") or "").strip()
        hostname = urlsplit(f"//{host}").hostname or ""
        if hostname not in LOOPBACK_HOSTS:
            self._send(403, {"error": "bad host"})
            return False
        auth = self.headers.get("Authorization") or ""
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(given.encode("utf-8"), self.server.api.token.encode("utf-8")):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _body(self) -> dict[str, Any] | None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self._send(413, {"error": "body too large"})
            return None
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "invalid json"})
            return None
        if not isinstance(data, dict):
            self._send(400, {"error": "expected an object"})
            return None
        return data

    def _path(self) -> str:
        return urlsplit(self.path).path.rstrip("/") or "/"

    # -- verbs
    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        api, path = self.server.api, self._path()
        if path == "/status":
            self._send(200, api.status())
        elif path == "/suggestion":
            self._send(200, suggestion_payload(api.runtime))
        elif path == "/memories":
            store = api.store
            self._send(200, {"memories": list_for_viewer(store) if store is not None else []})
        elif path == "/folders":
            self._send(200, api.folders())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        api, path = self.server.api, self._path()
        data = self._body()
        if data is None:
            return
        if path == "/pause":
            api.runtime_pause()
            self._send(200, {"paused": True})
        elif path == "/resume":
            api.runtime_resume()
            self._send(200, {"paused": api.runtime.paused})
        elif path == "/trigger":
            self._send(200, {"triggered": api.trigger()})
        elif path in ("/folders", "/folders/remove"):
            folder = data.get("path")
            if not isinstance(folder, str) or not folder.strip():
                self._send(400, {"error": "path (string) required"})
                return
            try:
                result = api.add_folder(folder) if path == "/folders" else api.remove_folder(folder)
            except (ConsentError, NotADirectoryError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, result)
        elif path == "/feedback":
            sid = data.get("suggestion_id")
            edited = data.get("edited_text")
            try:
                outcome = FeedbackOutcome(str(data.get("outcome", "")).lower())
            except ValueError:
                self._send(400, {"error": "outcome must be accepted/rejected/edited/ignored"})
                return
            if not isinstance(sid, str) or (edited is not None and not isinstance(edited, str)):
                self._send(400, {"error": "suggestion_id (string) required"})
                return
            if not api.runtime.known_suggestion(sid):
                self._send(404, {"error": "unknown suggestion"})
                return
            api.runtime.feedback(sid, outcome, edited_text=edited)
            self._send(200, {"ok": True, "suggestion_id": sid, "outcome": outcome.value})
        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        api, path = self.server.api, self._path()
        prefix = "/memories/"
        if not path.startswith(prefix) or len(path) <= len(prefix):
            self._send(404, {"error": "not found"})
            return
        item_id = unquote(path[len(prefix):])
        mem = api.memory
        if mem is None:
            self._send(404, {"error": "no memory"})
            return
        if mem.store.get_item(item_id) is None and item_id not in mem.index:
            self._send(404, {"error": "unknown memory"})
            return
        removed = forget(mem.store, mem.index, item_id)
        self._send(200, {"deleted": bool(removed), "verified": verify_deleted(mem.store, mem.index, item_id),
                         "id": item_id})

    def do_PUT(self) -> None:  # noqa: N802
        if self._authorized():
            self._send(405, {"error": "method not allowed"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, addr: tuple[str, int], api: "LocalAPIServer") -> None:
        self.api = api
        if ":" in addr[0]:
            import socket as _s  # stdlib, IPv6 loopback only
            self.address_family = _s.AF_INET6
        super().__init__(addr, _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Never print tracebacks (they could echo request data); drop the connection."""
        return


class LocalAPIServer:
    """Serve the SOMA runtime on loopback with a per-start bearer token.

    Args:
        runtime: a :class:`~soma.runtime.pipeline.SomaRuntime`.
        memory: a :class:`~soma.runtime.factory.MemoryBundle`; defaults to ``runtime.memory``.
        host: loopback host only.
        port: ``0`` picks a free port (see :attr:`port`).
        token_path: optional file that receives the token (owner-only permissions).
        scheduler: optional :class:`InferenceScheduler`; pause/resume go through it when given.
    """

    def __init__(self, runtime: Any, *, memory: Any = None, host: str = "127.0.0.1", port: int = 0,
                 token_path: str | Path | None = None, scheduler: Any = None) -> None:
        if host not in LOOPBACK_HOSTS:
            raise LocalAPIError(f"refusing to bind the SOMA API to non-loopback host {host!r}")
        self.runtime = runtime
        self.memory = memory if memory is not None else getattr(runtime, "memory", None)
        self.scheduler = scheduler
        self.host = "127.0.0.1" if host == "localhost" else host
        self.requested_port = int(port)
        self.token_path = Path(token_path) if token_path else None
        self.token = ""
        self._index_lock = threading.Lock()
        self._indexing: str | None = None
        self._last_report: dict[str, Any] = {}
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def store(self) -> Any:
        return getattr(self.memory, "store", None)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1]) if self._server else 0

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def runtime_pause(self) -> None:
        (self.scheduler or self.runtime).pause()

    def runtime_resume(self) -> None:
        (self.scheduler or self.runtime).resume()

    def trigger(self) -> bool:
        """Look at the screen now: the scheduler runs one forced step (never while paused)."""
        if self.scheduler is not None and not self.runtime.paused:
            self.scheduler.trigger_now()
            return True
        return False

    # -- folders (the user picks them in a UI; nothing is indexed otherwise)
    def folders(self) -> dict[str, Any]:
        consent, store = self.runtime.consent, self.store
        rows = [{"path": f, "memories": len(store.items_under(f)) if store is not None else 0}
                for f in consent.allowed_folders]
        with self._index_lock:
            return {"folders": rows, "indexing": self._indexing, "last_report": dict(self._last_report)}

    def add_folder(self, folder: str) -> dict[str, Any]:
        mem = self.memory
        if mem is None or getattr(mem, "ingestor", None) is None:
            raise ValueError("no memory store")
        p = Path(folder).expanduser().resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"not a folder: {p}")
        consent = self.runtime.consent
        consent.add_folder(p)              # refuses a home directory or drive root
        consent.grant(Scope.FOLDER_INDEXING)
        with self._index_lock:
            self._indexing = str(p)
        threading.Thread(target=self._index, args=(p,), name="soma-index", daemon=True).start()
        return {"path": str(p), "indexing": True}

    def _index(self, folder: Path) -> None:
        report: dict[str, Any] = {"path": str(folder)}
        try:
            r = self.memory.ingestor.ingest_folder(folder)
            report.update(added=len(r.added), updated=len(r.updated), unchanged=len(r.unchanged),
                          skipped=len(r.skipped), errors=len(r.errors))
        except Exception as exc:  # noqa: BLE001 - reported to the UI, never raised on a thread
            report["error"] = type(exc).__name__
        with self._index_lock:
            self._indexing = None
            self._last_report = report

    def remove_folder(self, folder: str) -> dict[str, Any]:
        p = Path(folder).expanduser().resolve()
        self.runtime.consent.remove_folder(p)
        removed = forget_source(self.memory.store, self.memory.index, p) if self.memory is not None else 0
        return {"path": str(p), "forgotten": removed}

    def status(self) -> dict[str, Any]:
        rt = self.runtime
        last = rt.last_result
        consent = rt.consent
        tel = getattr(rt, "telemetry", None)
        return {
            "paused": bool(rt.paused),
            "steps": int(getattr(rt, "steps", 0)),
            "processed": int(getattr(rt, "processed", 0)),
            "frame_version": last.frame_version if last is not None else 0,
            "last_skip_reason": (rt.last_skip_reason or "").split(":", 1)[0] or None,
            "consent": {s.value: consent.allowed(s) for s in Scope},
            "memories": self.store.count_items() if self.store is not None else 0,
            "telemetry": tel.export() if tel is not None and tel.enabled else None,
        }

    def start(self) -> "LocalAPIServer":
        if self.running:
            return self
        self.token = secrets.token_urlsafe(32)
        self._server = _Server((self.host, self.requested_port), self)
        if self.token_path is not None:
            write_token(self.token_path, self.token)
        self._thread = threading.Thread(target=self._server.serve_forever, name="soma-local-api",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(5)
        self._server = None
        self._thread = None
        if self.token_path is not None:
            try:
                self.token_path.unlink()
            except FileNotFoundError:
                pass
        self.token = ""

    def __enter__(self) -> "LocalAPIServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
