"""SOMA companion: on-screen action suggestions from the local SOMA service, shown in Pette's bubble.

SOMA (SOMA/, its own venv) watches the screen with consent and ranks next steps with its PyTorch model;
`SOMA/scripts/serve.py` serves the latest suggestion on a token-protected localhost API and writes
`{"url", "token"}` to the file named by `PETTE_SOMA_ENDPOINT`. Pette polls it and, while she is on screen
and idle, shows the suggestion as a card: "Do it" / "Not now". Nothing runs without that click:

  * share_changes / ask_review / tests_pass / ask_about_error -> a fixed Pette command (no screen text)
    -> the normal draft + countdown, where Cancel still stops the send;
  * search_tools -> "search Darwin for <page topic>" (Darwin search only);
  * anything else (explain an error, open a file, compare with prior work) -> copied to the clipboard.

Accept / dismiss / ignore go back to SOMA as feedback (it trains SOMA's online ranker when SOMA's
feedback storage is granted) and into Pette's fact log as `offers` rows (kind + outcome only).
Pette's windows are excluded from screen capture so SOMA never reads her own bubble back.
"""
import json
import os
import re
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QGuiApplication

from .ui import Proposal

ENDPOINT_VAR = "PETTE_SOMA_ENDPOINT"
POLL_S = 1.0          # SOMA steps every >= 2 s, so a 1 s poll sees every shown suggestion
SHOW_S = 15           # an unanswered card goes away (and counts as "ignored", never as a dismissal)
TIMEOUT_S = 2.0
REPORT_AFTER = 20     # consecutive failed polls (~20 s) before telling you SOMA is unreachable

COMMANDS = {          # SOMA template -> Pette command. Fixed text: screen content never reaches the LLM.
    "share_changes": "Slack the team a summary of what I changed in this file",
    "ask_review": "Slack the team asking someone to review my changes to this file",
    "tests_pass": "Slack the team saying the tests pass",
    "ask_about_error": "Slack the team asking for help with the error I'm seeing in this file",
}
LABELS = {
    "EXPLAIN_ERROR": "Explain error", "OPEN_RELEVANT_FILE": "Open file", "COMPARE_WITH_PRIOR_WORK": "Prior work",
    "SUGGEST_NEXT_STEP": "Next step", "ASK_CLARIFYING_QUESTION": "Question",
}
_TOPIC_RE = re.compile(r"[^\w\s.,+#&'-]")


class SomaError(Exception):
    """SOMA unreachable or refused; Pette keeps working without suggestions."""


def clean_topic(text: str) -> str:
    """A search topic from SOMA's (already sanitized) evidence: plain characters, short."""
    return re.sub(r"\s+", " ", _TOPIC_RE.sub(" ", text or "")).strip()[:60]


def command_for(s: dict) -> str | None:
    """The Pette command an accepted suggestion runs, or None (copy it instead)."""
    template = s.get("template") or ""
    if template in COMMANDS:
        return COMMANDS[template]
    if template == "search_tools":
        topic = clean_topic((s.get("evidence") or [""])[0])
        return f"search Darwin for {topic}" if topic else None
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
            raise SomaError("SOMA isn't running (no endpoint file).") from None

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
            raise SomaError(f"SOMA returned HTTP {code}.")
        return data

    def suggestion(self) -> dict:
        return self._call("GET", "/suggestion")

    def status(self) -> dict:
        return self._call("GET", "/status")

    def feedback(self, suggestion_id: str, outcome: str) -> dict:
        return self._call("POST", "/feedback", {"suggestion_id": suggestion_id, "outcome": outcome})

    def pause(self) -> dict:
        return self._call("POST", "/pause", {})

    def resume(self) -> dict:
        return self._call("POST", "/resume", {})


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
    """Polls SOMA on a worker thread; shows suggestions on the main thread when Pette is free."""
    _arrived = Signal(dict)        # from the poll thread (queued)
    _error = Signal(str)
    enabled_changed = Signal(bool)
    toggle_requested = Signal()    # hotkey: safe to emit from any thread

    def __init__(self, overlay, pipeline, client: SomaClient | None = None, *, connect=SomaClient.from_endpoint,
                 memory=None, spawn=_thread, poll_s: float = POLL_S, show_s: float = SHOW_S, clipboard=None):
        """`client=None` connects lazily through `connect()` (SOMA may still be loading its model)."""
        super().__init__()
        self.overlay, self.pipeline, self.client, self.memory = overlay, pipeline, client, memory
        self._connect = connect
        self._failures = 0
        self._spawn, self.poll_s = spawn, poll_s
        self.enabled = True
        self.shown: dict[str, dict] = {}       # Proposal.id -> SOMA suggestion
        self._seen: set[str] = set()           # SOMA suggestion ids already handled
        self._card: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._clipboard = clipboard or (lambda text: QGuiApplication.clipboard().setText(text))
        self._expire = QTimer(self, singleShot=True, interval=int(show_s * 1000), timeout=self._expired)
        self._arrived.connect(self._on_suggestion)
        self.toggle_requested.connect(self.toggle)
        self._error.connect(self._on_error)
        overlay.approved.connect(self._accepted)
        overlay.cancelled.connect(self._dismissed)
        overlay.pause_toggled.connect(self._paused)

    # --- lifecycle ---------------------------------------------------------
    def start(self):
        self._spawn(self._loop)

    def stop(self):
        self._stop.set()
        self._wake.set()

    def set_enabled(self, on: bool):
        if on == self.enabled:
            return
        self.enabled = on
        if not on:
            self._clear_card(outcome="ignored")
        self._spawn(lambda: self._safe("resume" if on else "pause"))
        self.enabled_changed.emit(on)
        self.overlay.say("Suggestions on." if on else "Suggestions off. SOMA stopped watching the screen.",
                         title="SOMA", seconds=4)

    def toggle(self):
        self.set_enabled(not self.enabled)

    # --- worker thread ----------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            self.poll_once()
            self._wake.wait(self.poll_s)
            self._wake.clear()

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
                self._error.emit(str(e))
            return
        self._failures = 0
        if s.get("available") and not s.get("silent") and s.get("valid") and s.get("text") \
                and s.get("id") not in self._seen:
            self._arrived.emit(s)

    def _safe(self, method: str, *args):
        if self.client is None:
            return
        try:
            getattr(self.client, method)(*args)
        except SomaError:
            pass

    # --- main thread ------------------------------------------------------
    def _free(self) -> bool:
        o, b = self.overlay, self.overlay.bubble
        return (self.enabled and not o.paused and o.context_allowed and not self.pipeline.busy
                and not b.asking and b.proposal is None)

    def _on_suggestion(self, s: dict):
        sid = s["id"]
        if sid in self._seen or not self._free():
            return                                       # busy: SOMA's cooldown decides when to try again
        self._seen.add(sid)
        label = LABELS.get(s.get("action", ""), "Suggestion")
        acting = command_for(s) is not None
        p = Proposal(title=f"SOMA suggests · {label}", body=s["text"],
                     action_label="Do it" if acting else "Copy", cancel_label="Not now")
        self.shown[p.id] = s
        self._card = p.id
        self.overlay.propose(p)
        self._expire.start()
        self._log(s, "shown")

    def _accepted(self, pid: str):
        s = self.shown.pop(pid, None)
        if s is None:
            return
        self._card = None
        self._expire.stop()
        self._feedback(s, "accepted")
        cmd = command_for(s)
        if cmd and self.pipeline.run_command(cmd):
            return
        self._clipboard(s["text"])
        self.overlay.say("Copied. Paste it to your coding agent.", title="SOMA", seconds=4)

    def _dismissed(self, pid: str):
        s = self.shown.pop(pid, None)
        if s is None:
            return
        self._card = None
        self._expire.stop()
        self._feedback(s, "rejected")
        self.overlay.dismiss()

    def _expired(self):
        self._clear_card(outcome="ignored")

    def _clear_card(self, outcome: str):
        pid, self._card = self._card, None
        self._expire.stop()
        s = self.shown.pop(pid, None) if pid else None
        if s is None:
            return
        if self.overlay.bubble.proposal is not None and self.overlay.bubble.proposal.id == pid:
            self.overlay.dismiss()
        self._feedback(s, outcome)

    def _paused(self, paused: bool):
        if paused:
            self._clear_card(outcome="ignored")
        if self.enabled:
            self._spawn(lambda: self._safe("pause" if paused else "resume"))

    def _feedback(self, s: dict, outcome: str):
        self._log(s, {"accepted": "accepted", "rejected": "dismissed"}.get(outcome, outcome))
        self._spawn(lambda: self._safe("feedback", s["id"], outcome))

    def _log(self, s: dict, outcome: str):
        if self.memory is not None:   # facts only: which kind of suggestion and what you did
            self.memory.log_offer(f"soma:{s.get('template') or s.get('action', '').lower()}", outcome)

    def _on_error(self, msg: str):
        self.overlay.notify("SOMA", f"{msg} Suggestions resume when it's back.")
