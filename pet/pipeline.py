"""Orchestrates Context -> Voice -> AI -> UI -> Dispatch. Lives on the Qt main thread; slow steps run in threads."""
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal

from . import context, dispatch, voice
from .brain import Brain, BrainError, Draft
from .dispatch import DispatchError
from .ui import Overlay, Proposal
from .voice import VoiceError

MIN_INTERVAL_S = 3
MAX_REQUESTS = 50
CAPTURE_DELAY_MS = 150  # let the overlay disappear before the screenshot


def _thread(fn):
    threading.Thread(target=fn, daemon=True).start()


class Pipeline(QObject):
    _deliver = Signal(int, object)  # (generation, callable to run on the main thread)
    hotkey_pressed = Signal()       # safe to emit from the keyboard library's thread

    def __init__(self, overlay: Overlay, brain: Brain | None = None, *, capture=context.capture,
                 record=voice.record_until_silence, transcribe=voice.transcribe, send=dispatch.dispatch,
                 configured=dispatch.configured, use_voice=True, spawn=_thread, capture_delay_ms=CAPTURE_DELAY_MS):
        super().__init__()
        self.overlay = overlay
        self.brain = brain or Brain()
        self._capture, self._record, self._transcribe = capture, record, transcribe
        self._send, self._configured = send, configured
        self.use_voice = use_voice
        self._spawn = spawn
        self._delay = capture_delay_ms
        self.gen = 0
        self.busy = False
        self.ctx: context.Context | None = None
        self.pending: dict[str, Draft] = {}
        self._last_request = 0.0
        self._requests = 0

        self._deliver.connect(self._on_deliver)
        self.hotkey_pressed.connect(self.trigger)
        overlay.ask_requested.connect(self.trigger)
        overlay.text_submitted.connect(self._heard)
        overlay.input_cancelled.connect(self._reset)
        overlay.approved.connect(self._approved)
        overlay.cancelled.connect(lambda pid: self.pending.pop(pid, None))
        overlay.pause_toggled.connect(self._paused)

    # --- flow -------------------------------------------------------------
    def trigger(self):
        if self.overlay.paused or self.busy:
            return
        self.busy = True
        gen = self.gen
        was_visible = self.overlay.isVisible()
        self.overlay.hide()

        def after_capture():
            if gen != self.gen:
                return
            self.ctx = self._capture()
            if was_visible:
                self.overlay.show()
            self._listen()

        if self._delay:
            QTimer.singleShot(self._delay, after_capture)
        else:
            after_capture()

    def _listen(self):
        if not self.use_voice:
            self.overlay.ask_text("What should I do?")
            return
        self.overlay.say("Listening...", seconds=None)
        self.overlay.set_state("listening")
        self._run(lambda: self._transcribe(self._record()), self._heard,
                  lambda e: self.overlay.ask_text(f"{e} Type your command instead:"), VoiceError)

    def _heard(self, command: str):
        if self.overlay.paused:
            return
        now = time.monotonic()
        if self._requests >= MAX_REQUESTS:
            return self._fail("Session request limit reached.")
        if now - self._last_request < MIN_INTERVAL_S:
            return self._fail("Slow down a moment, then try again.")
        self._last_request, self._requests = now, self._requests + 1
        self.busy = True
        self.overlay.say(f"“{command}”", title="Heard", seconds=None)
        self.overlay.set_state("thinking")
        ctx = self.ctx or context.Context(None, None)
        self._run(lambda: self.brain.draft(command, ctx), self._drafted, self._fail, BrainError)

    def _drafted(self, d: Draft):
        self._reset()
        label = getattr(self.brain, "label", "")  # demo brains must say so on every draft
        if d.platform == "none":
            self.overlay.say(d.message, title=label or "FlowState", seconds=15)
            return
        title = f"{d.platform.capitalize()} to {d.recipient}"
        if not self._configured(d.platform):
            title += " (not configured)"
        if label:
            title = f"{label} · {title}"
        body = f"Subject: {d.subject}\n\n{d.message}" if d.platform == "email" and d.subject else d.message
        p = Proposal(title=title, body=body, action_label="Send")
        self.pending[p.id] = d
        self.overlay.propose(p)

    def _approved(self, pid: str):
        d = self.pending.pop(pid, None)
        if d is None:
            return
        self.overlay.say("Sending...", seconds=None)
        self._run(lambda: self._send(d), lambda msg: self.overlay.say(msg, seconds=4), self._fail, DispatchError)

    def _fail(self, msg):
        self._reset()
        self.overlay.say(str(msg), title="Couldn't finish", seconds=6)

    def _reset(self):
        self.busy = False
        self.ctx = None  # drop the screenshot as soon as it's no longer needed

    def _paused(self, paused: bool):
        if paused:
            self.gen += 1  # late results from in-flight work are ignored
            self.pending.clear()
            self._reset()

    # --- threading --------------------------------------------------------
    def _run(self, work, on_ok, on_err, expected):
        gen = self.gen

        def task():
            try:
                result = work()
                self._deliver.emit(gen, lambda: on_ok(result))
            except expected as e:
                self._deliver.emit(gen, lambda: on_err(e))
            except Exception as e:  # never let a worker die silently
                self._deliver.emit(gen, lambda: self._fail(f"Unexpected error: {type(e).__name__}"))

        self._spawn(task)

    def _on_deliver(self, gen: int, fn):
        if gen == self.gen and not self.overlay.paused:
            fn()
