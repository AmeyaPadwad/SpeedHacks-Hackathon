"""Pette's brain is SOMA: she shows what SOMA suggests and does the one thing you allow.

SOMA (SOMA/, its own venv) watches the screen with consent, searches the folders you indexed and ranks
next steps with its PyTorch model. `SOMA/scripts/serve.py` serves it on a token-protected localhost API and
writes `{"url", "token"}` to the file named by `PETTE_SOMA_ENDPOINT`. `Suggester` is the hub on Pette's side:

  * polls the latest suggestion (worker thread) and, while Pette is on screen and free, shows it as a
    glass card: Open or Copy / Not now, with a bar for the 15 s it stays;
  * click Pette / the hotkey -> SOMA looks at the screen now; tray -> add or remove indexed folders;
  * "Do it" (or a pinch) -> open the file SOMA pointed at, or copy the text (pet/actions.py). Nothing else;
  * accept / not now / ignore go back to SOMA (its online ranker) and into Pette's fact log (kind + outcome),
    and kinds you keep putting off are muted (db/world_model.py);
  * Pette's windows are excluded from screen capture so SOMA never reads her own bubble back.

All widget work happens on the Qt main thread; network calls run through `spawn` and come back as signals.
"""
import json
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QGuiApplication

from . import actions
from .ui import Proposal

ENDPOINT_VAR = "PETTE_SOMA_ENDPOINT"
POLL_S = 1.0          # SOMA steps every >= 2 s, so a 1 s poll sees every shown suggestion
SHOW_S = 15           # an unanswered card goes away (and counts as "ignored", never as "Not now")
LOOK_TIMEOUT_S = 12   # after "look now", give up waiting for a suggestion
TIMEOUT_S = 2.0
REPORT_AFTER = 20     # consecutive failed polls (~20 s) before saying SOMA is offline

LABELS = {
    "EXPLAIN_ERROR": "Explain error", "OPEN_RELEVANT_FILE": "Open file", "COMPARE_WITH_PRIOR_WORK": "Your files",
    "SUGGEST_NEXT_STEP": "Next step", "ASK_CLARIFYING_QUESTION": "Question",
}


class SomaError(Exception):
    """SOMA unreachable or refused; shown in the bubble or the tray."""


def kind_of(s: dict) -> str:
    """Fact-log kind for a suggestion: its template, else its action (never its text)."""
    return "soma:" + (s.get("template") or s.get("action") or "suggestion").lower()


def opener(s: dict) -> tuple[str, int | None] | None:
    """The file "Do it" opens: the on-screen file SOMA matched, or the indexed file behind a memory."""
    if s.get("file"):
        line = s.get("line")
        return s["file"], line if isinstance(line, int) and line > 0 else None
    if s.get("memory_source"):
        return s["memory_source"], None
    return None


class SomaClient:
    """Tiny client for SOMA's local API (soma/api/local_api.py). `request` is injectable for tests."""

    def __init__(self, url: str, token: str, *, request=None, timeout: float = TIMEOUT_S):
        if not url.startswith(("http://127.0.0.1:", "http://localhost:", "http://[::1]:")):
            raise SomaError("SOMA must be on localhost.")
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self._request = request or self._requests

    @classmethod
    def from_endpoint(cls, path: str | os.PathLike | None = None, **kw) -> "SomaClient":
        p = Path(path or os.environ.get(ENDPOINT_VAR) or "")
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            return cls(info["url"], info["token"], **kw)
        except (OSError, ValueError, KeyError, TypeError):
            raise SomaError("SOMA isn't running.") from None

    def _requests(self, method: str, path: str, body: dict | None):
        import requests
        r = requests.request(method, self.url + path, json=body, timeout=self.timeout,
                             headers={"Authorization": f"Bearer {self.token}"})
        return r.status_code, (r.json() if r.content else {})

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        try:
            code, data = self._request(method, path, body)
        except Exception as e:  # connection refused, timeout, bad JSON
            raise SomaError(f"SOMA unreachable ({type(e).__name__}).") from None
        if code != 200:
            raise SomaError((data or {}).get("error") or f"SOMA returned HTTP {code}.")
        return data

    def suggestion(self) -> dict:
        return self._call("GET", "/suggestion")

    def status(self) -> dict:
        return self._call("GET", "/status")

    def feedback(self, suggestion_id: str, outcome: str) -> dict:
        return self._call("POST", "/feedback", {"suggestion_id": suggestion_id, "outcome": outcome})

    def trigger(self) -> dict:
        return self._call("POST", "/trigger", {})

    def pause(self) -> dict:
        return self._call("POST", "/pause", {})

    def resume(self) -> dict:
        return self._call("POST", "/resume", {})

    def folders(self) -> dict:
        return self._call("GET", "/folders")

    def add_folder(self, path: str) -> dict:
        return self._call("POST", "/folders", {"path": path})

    def remove_folder(self, path: str) -> dict:
        return self._call("POST", "/folders/remove", {"path": path})


def exclude_from_capture(widget) -> bool:
    """Keep a window out of screenshots and screen capture (Windows 10 2004+), so SOMA never OCRs
    Pette's own bubble. No-op elsewhere."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        WDA_EXCLUDEFROMCAPTURE = 0x11
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(int(widget.winId()), WDA_EXCLUDEFROMCAPTURE))
    except Exception:
        return False


def _thread(fn):
    threading.Thread(target=fn, daemon=True).start()


class Suggester(QObject):
    """The hub: SOMA's suggestions in, cards and "Do it" out. Lives on the Qt main thread."""
    status = Signal(str, str)          # (state, detail): idle|looking|suggesting|done|dismissed|indexing|failed
    folders_changed = Signal(list)     # [{"path", "memories"}]
    enabled_changed = Signal(bool)
    toggle_requested = Signal()        # hotkey: safe to emit from any thread
    look_requested = Signal()          # hotkey: safe to emit from any thread
    _arrived = Signal(dict)            # from worker threads (queued)
    _result = Signal(object)           # a callable to run on the main thread
    _offline = Signal(str)

    def __init__(self, overlay, client: SomaClient | None = None, *, connect=SomaClient.from_endpoint,
                 memory=None, world=None, spawn=_thread, poll_s: float = POLL_S, show_s: float = SHOW_S,
                 clipboard=None, open_file=actions.open_file):
        """`client=None` connects lazily through `connect()` (SOMA may still be loading its model)."""
        super().__init__()
        self.overlay, self.client, self.memory, self.world = overlay, client, memory, world
        self._connect, self._spawn, self.poll_s, self.show_s = connect, spawn, poll_s, show_s
        self._open_file = open_file
        self._clipboard = clipboard or (lambda text: QGuiApplication.clipboard().setText(text))
        self.enabled = True
        self.state = "idle"
        self.folders: list[dict] = []
        self.shown: dict[str, dict] = {}       # Proposal.id -> SOMA suggestion
        self._seen: set[str] = set()           # SOMA suggestion ids already handled
        self._card: str | None = None
        self._looking = False
        self._failures = 0
        self._stop = threading.Event()
        self._left = 0.0
        self._tick = QTimer(self, interval=250, timeout=self._count_down)
        self._look_timeout = QTimer(self, singleShot=True, interval=int(LOOK_TIMEOUT_S * 1000),
                                    timeout=self._nothing_found)
        self._indexing = QTimer(self, interval=1500, timeout=self.refresh_folders)
        self._arrived.connect(self._on_suggestion)
        self._result.connect(lambda fn: fn())
        self._offline.connect(self._on_offline)
        self.toggle_requested.connect(self.toggle)
        self.look_requested.connect(self.look_now)
        self.status.connect(overlay.set_status)
        overlay.look_requested.connect(self.look_now)
        overlay.approved.connect(self._accepted)
        overlay.cancelled.connect(self._dismissed)
        overlay.pinch_on_proposal.connect(self._pinched)
        overlay.pause_toggled.connect(self._paused)

    # --- helpers ----------------------------------------------------------
    def _set(self, state: str, detail: str = ""):
        self.state = state
        self.status.emit(state, detail)

    def _async(self, work, on_ok=None, on_err=None):
        """Run `work(client)` on a worker; deliver the result / SomaError message on the main thread."""
        def task():
            try:
                if self.client is None:
                    self.client = self._connect()
                result = work(self.client)
            except SomaError as e:
                if on_err:
                    self._result.emit(lambda msg=str(e): on_err(msg))
                return
            if on_ok:
                self._result.emit(lambda: on_ok(result))
        self._spawn(task)

    # --- lifecycle ----------------------------------------------------------
    def start(self):
        self._spawn(self._loop)
        self.refresh_folders()

    def stop(self):
        self._stop.set()

    def set_enabled(self, on: bool):
        if on == self.enabled:
            return
        self.enabled = on
        if not on:
            self._clear_card(outcome="ignored")
            self._set("idle")
        self._async(lambda c: c.resume() if on else c.pause())
        self.enabled_changed.emit(on)
        self.overlay.say("Suggestions on." if on else "Suggestions off. SOMA stopped looking at the screen.",
                         title="SOMA", seconds=4)

    def toggle(self):
        self.set_enabled(not self.enabled)

    # --- worker thread ----------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.poll_s)

    def poll_once(self):
        """One poll (worker thread): emit a fresh, valid, non-silent suggestion. Never touches widgets."""
        if not self.enabled:
            return
        try:
            if self.client is None:
                self.client = self._connect()
            s = self.client.suggestion()
        except SomaError as e:
            self._failures += 1
            if self._failures == REPORT_AFTER:
                self._offline.emit(str(e))
            return
        self._failures = 0
        if s.get("available") and not s.get("silent") and s.get("valid") and s.get("text") \
                and s.get("id") not in self._seen:
            self._arrived.emit(s)

    # --- main thread: look now ---------------------------------------------
    def _free(self) -> bool:
        o = self.overlay
        return self.enabled and not o.paused and o.context_allowed and o.bubble.proposal is None

    def look_now(self):
        """Click / hotkey: SOMA looks at the screen now; the next suggestion shows, or "nothing" after a while."""
        if self._looking or not self._free():
            return
        self._looking = True
        self._set("looking")
        self.overlay.say("Looking at your screen…", title="SOMA", seconds=None)
        self._look_timeout.start()
        self._async(lambda c: c.trigger(), on_ok=self._triggered, on_err=self._look_failed)

    def _triggered(self, result: dict):
        if not result.get("triggered"):
            self._look_failed("SOMA is paused.")

    def _look_failed(self, msg: str):
        self._looking = False
        self._look_timeout.stop()
        self._set("failed", "SOMA offline")
        self.overlay.say(msg, title="Couldn't look", seconds=6)

    def _nothing_found(self):
        if self._looking:
            self._looking = False
            self._set("idle")
            self.overlay.say("Nothing worth suggesting right now.", title="SOMA", seconds=4)

    # --- main thread: cards -------------------------------------------------
    def _on_suggestion(self, s: dict):
        sid = s["id"]
        if sid in self._seen:
            return
        asked = self._looking
        if asked:                                        # the "Looking…" bubble gives way to the card
            self.overlay.dismiss()
        if not self._free():
            return                                       # busy: SOMA's cooldown decides when to try again
        self._seen.add(sid)
        self._looking = False
        self._look_timeout.stop()
        if self.world is not None and not asked and kind_of(s) in self.world.muted_kinds():
            self._feedback(s, "ignored", log=False)      # you keep saying Not now to this kind
            return
        label = LABELS.get(s.get("action", ""), "Suggestion")
        p = Proposal(title=f"SOMA suggests · {label}", body=s["text"],
                     action_label="Open" if opener(s) else "Copy", cancel_label="Not now", glow=True)
        self.shown[p.id] = s
        self._card = p.id
        self._set("suggesting")
        self.overlay.propose(p)
        self._left = self.show_s
        self.overlay.set_progress(1.0)
        self._tick.start()
        self._log(s, "shown")

    def _count_down(self):
        self._left -= self._tick.interval() / 1000
        if self._left > 0:
            self.overlay.set_progress(self._left / self.show_s)
            return
        self._clear_card(outcome="ignored")
        self._set("idle")

    def _pinched(self, pid: str):
        """Pinch on a card = Do it (single-use, like the button)."""
        if pid in self.shown and self.overlay.bubble.consume() == pid:
            self._accepted(pid)

    def _accepted(self, pid: str):
        s = self.shown.pop(pid, None)
        if s is None:
            return
        self._card = None
        self._tick.stop()
        self._feedback(s, "accepted")
        target = opener(s)
        try:
            msg = self._open_file(*target) if target else actions.copy_text(s["text"], self._clipboard)
        except actions.ActionError as e:
            self._set("failed", "Couldn't open")
            self.overlay.say(str(e), title="Couldn't open", seconds=6)
            return
        self._set("done")
        self.overlay.say(msg, title="Done", seconds=4)

    def _dismissed(self, pid: str):
        s = self.shown.pop(pid, None)
        if s is None:
            return
        self._card = None
        self._tick.stop()
        self._feedback(s, "rejected")
        self.overlay.dismiss()
        self._set("dismissed")

    def _clear_card(self, outcome: str):
        pid, self._card = self._card, None
        self._tick.stop()
        s = self.shown.pop(pid, None) if pid else None
        if s is None:
            return
        if self.overlay.bubble.proposal is not None and self.overlay.bubble.proposal.id == pid:
            self.overlay.dismiss()
        self._feedback(s, outcome)

    def _paused(self, paused: bool):
        if paused:
            self._clear_card(outcome="ignored")
            self._looking = False
            self._look_timeout.stop()
            self._set("idle")
        if self.enabled:
            self._async(lambda c: c.pause() if paused else c.resume())

    def _feedback(self, s: dict, outcome: str, log: bool = True):
        if log:
            self._log(s, {"accepted": "accepted", "rejected": "dismissed"}.get(outcome, outcome))
        self._async(lambda c: c.feedback(s["id"], outcome))

    def _log(self, s: dict, outcome: str):
        if self.memory is not None:   # facts only: which kind of suggestion and what you did
            self.memory.log_offer(kind_of(s), outcome)

    def _on_offline(self, msg: str):
        self._set("failed", "SOMA offline")
        self.overlay.notify("SOMA", f"{msg} Suggestions resume when it's back.")

    # --- folders (tray) -----------------------------------------------------
    def refresh_folders(self):
        self._async(lambda c: c.folders(), on_ok=self._got_folders)

    def _got_folders(self, data: dict):
        self.folders = list(data.get("folders") or [])
        self.folders_changed.emit(self.folders)
        if data.get("indexing"):
            if not self._indexing.isActive():
                self._indexing.start()
            return
        if self._indexing.isActive():                    # just finished
            self._indexing.stop()
            r = data.get("last_report") or {}
            name = Path(r.get("path", "")).name or "the folder"
            if r.get("error"):
                self.overlay.say(f"Couldn't index {name} ({r['error']}).", title="SOMA memory", seconds=8)
            else:
                n = r.get("added", 0) + r.get("updated", 0)
                self.overlay.say(f"Indexed {n} file(s) from {name}. SOMA uses them in suggestions now.",
                                 title="SOMA memory", seconds=6)
            if self.state == "indexing":
                self._set("idle")

    def add_folder(self, path: str):
        """Tray -> folder picker -> here. SOMA records the consent and indexes in the background."""
        if not path:
            return
        self._set("indexing", Path(path).name)
        self.overlay.say(f"Indexing {Path(path).name}…", title="SOMA memory", seconds=None)

        def ok(_):
            self._indexing.start()
            self.refresh_folders()

        def err(msg):
            self._set("failed", "Not indexed")
            self.overlay.say(msg, title="Couldn't add folder", seconds=8)

        self._async(lambda c: c.add_folder(path), on_ok=ok, on_err=err)

    def remove_folder(self, path: str):
        def ok(r):
            self.overlay.say(f"Removed {Path(path).name}; SOMA forgot {r.get('forgotten', 0)} file(s) from it.",
                             title="SOMA memory", seconds=6)
            self.refresh_folders()

        self._async(lambda c: c.remove_folder(path), on_ok=ok,
                    on_err=lambda msg: self.overlay.say(msg, title="Couldn't remove folder", seconds=6))
