"""The one workflow: click Pette -> listen (ElevenLabs) -> your words in the bubble -> Groq -> dispatch.

    idle --click/hotkey--> listening --transcript--> thinking --draft--> pending (countdown, Cancel)
         --timer or "Send now"--> sending --> sent | failed;  Cancel / pinch during pending --> cancelled

Lives on the Qt main thread; slow steps run in threads and come back through `_deliver`.
`status` drives the camera ring above Pette and the world-model window.
"""
import threading
import time
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer, Signal

from . import context, darwin, dispatch, voice
from .brain import PLATFORM_WORDS, Brain, BrainError, Draft
from .dispatch import DispatchError
from .ui import Overlay, Proposal
from .voice import VoiceError

MIN_INTERVAL_S = 3
MAX_REQUESTS = 50
DEFAULT_SEND_DELAY_S = 4.0


def _thread(fn):
    threading.Thread(target=fn, daemon=True).start()


def _whisper_listen(on_level=None, register_cancel=None):
    return voice.transcribe(voice.record_until_silence())


class Pipeline(QObject):
    _deliver = Signal(int, object)  # (generation, callable to run on the main thread)
    hotkey_pressed = Signal()       # safe to emit from the keyboard library's thread
    status = Signal(str, str)       # (state, detail): idle|listening|thinking|pending|sending|sent|cancelled|failed
    level = Signal(float)           # mic loudness 0..1 while listening (from the listener thread)

    def __init__(self, overlay: Overlay, brain: Brain | None = None, *, capture=context.capture,
                 listen=_whisper_listen, send=dispatch.dispatch, configured=dispatch.configured,
                 use_voice=True, spawn=_thread, memory=None, world=None, send_delay_s=None):
        """`listen(on_level, register_cancel)` blocks in a worker and returns the transcript or raises
        VoiceError. A listener that can stop early passes its cancel function to `register_cancel`."""
        super().__init__()
        self.overlay = overlay
        self.memory, self.world = memory, world   # db.Store / db.WorldModel, optional
        self.brain = brain or Brain()
        self._capture, self._listen_fn = capture, listen
        self._send, self._configured = send, configured
        self.use_voice = use_voice
        self._spawn = spawn
        self._send_delay_s = send_delay_s
        self.gen = 0
        self.state = "idle"
        self.busy = False
        self.ctx: context.Context | None = None
        self.pending: dict[str, Draft] = {}
        self._command = ""
        self._last_request = 0.0
        self._requests = 0
        self.cancel_listen = None        # set by a listener that supports early stop
        self._countdown = QTimer(self, interval=1000, timeout=self._tick)
        self._pending_id: str | None = None
        self._left = 0

        self._deliver.connect(self._on_deliver)
        self.hotkey_pressed.connect(self.trigger)
        self.level.connect(overlay.set_level)
        self.status.connect(overlay.set_status)
        overlay.ask_requested.connect(self.trigger)
        overlay.text_submitted.connect(self._heard)
        overlay.input_cancelled.connect(lambda: self._set("idle"))
        overlay.approved.connect(self._approved)
        overlay.cancelled.connect(self._cancelled)
        overlay.pinch_on_proposal.connect(self._pinch_send)  # pinch = send the pending message now
        overlay.pause_toggled.connect(self._paused)

    # --- state ------------------------------------------------------------
    def _set(self, state: str, detail: str = ""):
        self.state = state
        self.busy = state in ("listening", "thinking", "pending", "sending")
        self.status.emit(state, detail)

    def _can_start(self) -> bool:
        return not (self.overlay.paused or self.busy) and self.overlay.context_allowed

    def delay_s(self) -> float:
        if self._send_delay_s is not None:
            return self._send_delay_s
        return self.world.send_delay_s() if self.world is not None else DEFAULT_SEND_DELAY_S

    # --- 1. listen --------------------------------------------------------
    def trigger(self):
        if self.state == "listening" and self.cancel_listen:
            self.cancel_listen()          # click again while listening: stop
            return
        if not self._can_start():
            return
        if not self.use_voice:
            self._set("listening", "typing")
            self.overlay.ask_text("What should I do?")
            return
        self._set("listening")
        self.overlay.say("Listening…", title="Speak now", seconds=None)
        self._run(lambda: self._listen_fn(self.level.emit, lambda c: setattr(self, "cancel_listen", c)),
                  self._heard, self._listen_failed, VoiceError)

    def _listen_failed(self, e: VoiceError):
        self.cancel_listen = None
        if str(e) == "Listening cancelled.":
            self._set("idle")
            self.overlay.dismiss()
            return
        self._set("listening", "typing")   # typed fallback keeps the same workflow
        self.overlay.ask_text(f"{e} Type your command instead:")

    def run_command(self, command: str) -> bool:
        """Run a command as if it had been said (e.g. an accepted SOMA suggestion). False if Pette is busy,
        paused or hidden. The draft still goes through the countdown, so Cancel can stop it."""
        if not self._can_start():
            return False
        self._heard(command)
        return True

    # --- 2. transcript -> Groq --------------------------------------------
    def _rate_ok(self) -> bool:
        now = time.monotonic()
        if self._requests >= MAX_REQUESTS:
            self._fail("Session request limit reached.")
            return False
        if now - self._last_request < MIN_INTERVAL_S:
            self._fail("Slow down a moment, then try again.")
            return False
        self._last_request, self._requests = now, self._requests + 1
        return True

    def _heard(self, command: str):
        self.cancel_listen = None
        if self.overlay.paused or not self._rate_ok():
            return
        self._command = command
        if self.world is not None and hasattr(self.brain, "contacts"):
            self.brain.contacts = dict(self.world.known_contacts())
        self._set("thinking")
        self.overlay.say(f"“{command}”", title="You said", seconds=None)

        def work():
            if darwin.wants_darwin(command):   # "search Darwin for …" -> results as a Slack draft
                return darwin.draft(command)
            self.ctx = self._capture()    # file + git diff only; no screenshot
            return self.brain.draft(command, self.ctx)

        self._run(work, self._drafted, self._fail, (BrainError, darwin.DarwinError))

    # --- 3. draft -> countdown -> dispatch --------------------------------
    def _drafted(self, d: Draft):
        label = getattr(self.brain, "label", "")  # demo brains must say so on every draft
        self.ctx = None
        if d.platform == "none":
            self._set("idle")
            self.overlay.say(d.message, title=label or "Pette", seconds=15)
            return
        learned = ""
        if self.world is not None and not PLATFORM_WORDS.search(self._command):
            usual = self.world.preferred_platform(d.recipient)
            if usual and usual != d.platform:   # you didn't name a platform: use the one you usually use
                d = replace(d, platform=usual)
            learned = " (usual)" if usual else ""
        self._remember(d, "drafted")
        where = f"{d.platform.capitalize()}{learned} to {d.recipient}"
        if not self._configured(d.platform):
            self._set("failed", f"{d.platform} not configured")
            self.overlay.say(f"{d.message}\n\n{d.platform.capitalize()} isn't configured, so nothing was sent.",
                             title=f"{where} · not sent", seconds=12)
            self._remember(d, "failed")
            return
        body = f"Subject: {d.subject}\n\n{d.message}" if d.platform == "email" and d.subject else d.message
        if d.code:
            body += f"\n\n＋ {d.code_label} ({d.code.count(chr(10)) + 1} lines attached)"
        p = Proposal(title=f"{label + ' · ' if label else ''}{where}", body=body, action_label="Send now",
                     countdown=True)
        self.pending = {p.id: d}
        self._pending_id = p.id
        self._left = max(1, round(self.delay_s()))
        self.overlay.propose(p)
        self._set("pending", str(self._left))
        self.overlay.set_countdown(self._left)
        self._countdown.start()

    def _tick(self):
        self._left -= 1
        if self._left > 0:
            self.overlay.set_countdown(self._left)
            self._set("pending", str(self._left))
            return
        self._countdown.stop()
        if self._pending_id:
            self.overlay.bubble.consume()     # the Cancel button can't fire after the send has started
            self._approved(self._pending_id)

    def _approved(self, pid: str):
        self._countdown.stop()
        d = self.pending.pop(pid, None)
        self._pending_id = None
        if d is None:
            return
        self._set("sending")
        self.overlay.say(f"Sending to {d.recipient} on {d.platform.capitalize()}…", seconds=None)

        def sent(msg):
            self._remember(d, "sent")
            self._set("sent", d.recipient)
            self.overlay.say(msg, title="Sent ✓", seconds=5)

        def failed(e):
            self._remember(d, "failed")
            self._set("failed", str(e))
            self.overlay.say(str(e), title="Not sent", seconds=8)

        self._run(lambda: self._send(d), sent, failed, DispatchError)

    def _pinch_send(self, pid: str):
        """The one gesture: pinch on a pending draft sends it (same as Send now). Single-use via consume()."""
        if pid in self.pending and self.overlay.bubble.consume() == pid:
            self._approved(pid)

    def _cancelled(self, pid: str):
        self._countdown.stop()
        d = self.pending.pop(pid, None)
        if d is None:
            return
        self._pending_id = None
        self._remember(d, "cancelled")
        self.overlay.dismiss()
        self._set("cancelled", d.recipient)
        self.overlay.say(f"Cancelled. Nothing was sent to {d.recipient}.", title="Not sent", seconds=4)

    # --- memory (facts only: no message text, code or audio) -------------
    def _remember(self, d: Draft, outcome: str):
        if self.memory is not None and d.platform in ("slack", "discord", "email"):
            self.memory.log_message(d.recipient, d.platform, outcome)

    def _fail(self, msg):
        self.cancel_listen = None
        self.ctx = None
        self._set("failed", str(msg))
        self.overlay.say(str(msg), title="Couldn't finish", seconds=6)

    def _paused(self, paused: bool):
        if paused:
            if self.cancel_listen:
                self.cancel_listen()
            self._countdown.stop()
            self.gen += 1  # late results from in-flight work are ignored
            self.pending.clear()
            self._pending_id = None
            self.ctx = None
            self._set("idle")

    # --- threading --------------------------------------------------------
    def _run(self, work, on_ok, on_err, expected):
        gen = self.gen

        def task():
            # Bind the exception as a default: Python unbinds `e` when the except block ends, and the
            # lambda runs later on the main thread (queued signal).
            try:
                result = work()
                self._deliver.emit(gen, lambda: on_ok(result))
            except expected as e:
                self._deliver.emit(gen, lambda e=e: on_err(e))
            except Exception as e:  # never let a worker die silently
                self._deliver.emit(gen, lambda e=e: self._fail(f"Unexpected error: {type(e).__name__}"))

        self._spawn(task)

    def _on_deliver(self, gen: int, fn):
        if gen == self.gen and not self.overlay.paused:
            fn()
