"""Orchestrates Context -> Voice -> AI -> UI -> Dispatch, plus Screen CV -> Analysis -> Coding agent.

Lives on the Qt main thread; slow steps run in threads and come back through `_deliver`.
"""
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal

from . import agents, context, dispatch, voice
from .agents import AgentError, AgentTask
from .analyst import Analysis, Analyst
from .brain import Brain, BrainError, Draft
from .dispatch import DispatchError
from .ui import Overlay, Proposal
from .voice import VoiceError

MIN_INTERVAL_S = 3
MAX_REQUESTS = 50
CAPTURE_DELAY_MS = 150  # let the overlay disappear before the screenshot
OFFER = "offer"          # pending marker for a proactive "Investigate?" question


def _thread(fn):
    threading.Thread(target=fn, daemon=True).start()


class Pipeline(QObject):
    _deliver = Signal(int, object)  # (generation, callable to run on the main thread)
    hotkey_pressed = Signal()       # safe to emit from the keyboard library's thread
    investigate_hotkey = Signal()

    def __init__(self, overlay: Overlay, brain: Brain | None = None, *, capture=context.capture,
                 record=voice.record_until_silence, transcribe=voice.transcribe, send=dispatch.dispatch,
                 configured=dispatch.configured, analyst: Analyst | None = None, pick_agent=agents.pick_agent,
                 copy_text=None, show_result=None, use_voice=True, spawn=_thread,
                 capture_delay_ms=CAPTURE_DELAY_MS):
        super().__init__()
        self.overlay = overlay
        self.brain = brain or Brain()
        self.analyst = analyst or Analyst()
        self._capture, self._record, self._transcribe = capture, record, transcribe
        self._send, self._configured = send, configured
        self._pick_agent = pick_agent
        self._copy_text = copy_text or _clipboard
        self._show_result = show_result or (lambda title, text: None)
        self.agent_name: str | None = None   # tray choice; None = auto
        self.use_voice = use_voice
        self._spawn = spawn
        self._delay = capture_delay_ms
        self.gen = 0
        self.busy = False
        self.ctx: context.Context | None = None
        self.pending: dict[str, Draft | AgentTask | str] = {}
        self._last_request = 0.0
        self._requests = 0

        self._deliver.connect(self._on_deliver)
        self.hotkey_pressed.connect(self.trigger)
        self.investigate_hotkey.connect(self.investigate)
        overlay.ask_requested.connect(self.trigger)
        overlay.investigate_requested.connect(self.investigate)
        overlay.text_submitted.connect(self._heard)
        overlay.input_cancelled.connect(self._reset)
        overlay.approved.connect(self._approved)
        overlay.cancelled.connect(lambda pid: self.pending.pop(pid, None))
        overlay.pause_toggled.connect(self._paused)

    @property
    def agent(self):
        return self._pick_agent(self.agent_name)

    # --- capture ----------------------------------------------------------
    def _capture_then(self, next_step):
        """Hide Pette, take one screenshot (+ local screen CV), show Pette, continue on the main thread."""
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
            next_step()

        if self._delay:
            QTimer.singleShot(self._delay, after_capture)
        else:
            after_capture()

    # --- voice / message flow ---------------------------------------------
    def trigger(self):
        if self.overlay.paused or self.busy:
            return
        self._capture_then(self._listen)

    def _listen(self):
        if not self.use_voice:
            self.overlay.ask_text("What should I do?")
            return
        self.overlay.say("Listening...", seconds=None)
        self.overlay.set_state("listening")
        self._run(lambda: self._transcribe(self._record()), self._heard,
                  lambda e: self.overlay.ask_text(f"{e} Type your command instead:"), VoiceError)

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
        if self.overlay.paused or not self._rate_ok():
            return
        self.busy = True
        self.overlay.say(f"“{command}”", title="Heard", seconds=None)
        self.overlay.set_state("thinking")
        ctx = self.ctx or context.Context(None, None)
        self._run(lambda: self.brain.draft(command, ctx), self._drafted, self._fail, BrainError)

    def _drafted(self, d: Draft):
        label = getattr(self.brain, "label", "")  # demo brains must say so on every draft
        if d.platform == "agent":
            self._analyze(request=d.message)        # reuse the screenshot taken when the hotkey fired
            return
        self._reset()
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

    # --- screen CV -> analysis -> coding agent ----------------------------
    def investigate(self):
        """Explicit request (tray/hotkey/approved offer): capture the screen and analyse it."""
        if self.overlay.paused or self.busy:
            return
        self._capture_then(self._analyze)

    def offer_investigation(self, reason: str):
        """Proactive: ask first. Nothing is captured or uploaded unless the user clicks Investigate."""
        if self.overlay.paused or self.busy or self.overlay.bubble.isVisible():
            return
        p = Proposal(title="Looks like you're stuck", body=f"{reason}\n\nWant me to look at your screen and "
                     f"prepare a task for {self.agent.label}?", action_label="Investigate")
        self.pending[p.id] = OFFER
        self.overlay.propose(p)

    def _analyze(self, request: str = ""):
        self.busy = True
        ctx = self.ctx or context.Context(None, None)
        n = len(ctx.screen.regions) if ctx.screen else 0
        self.overlay.say(f"Looking at your screen... {n} error region(s) found locally." if ctx.screen
                         else "Looking at your screen...", seconds=None)
        self.overlay.set_state("thinking")
        self._run(lambda: self.analyst.analyze(ctx), lambda a: self._analyzed(a, ctx, request), self._fail, BrainError)

    def _analyzed(self, a: Analysis, ctx: context.Context, request: str):
        self._reset()
        agent = self.agent
        task = AgentTask.from_context(a, ctx, request)
        tag = "" if a.source == "ai" else " · local CV only"
        body = f"{a.summary()}\n\nNext: {a.next_step}"
        if a.source != "ai":
            body += f"\n\n({a.source})"
        action = agent.label if agent.name == "clipboard" else f"Ask {agent.label}"
        p = Proposal(title=f"{a.confidence:.0%} confident{tag}", body=body, action_label=action)
        self.pending[p.id] = task
        self.overlay.propose(p)

    def _run_agent(self, task: AgentTask):
        agent = self.agent
        self.busy = True
        if agent.name != "clipboard":
            self.overlay.say(f"{agent.label} is investigating (read-only)...", seconds=None)
            self.overlay.set_state("thinking")

        def done(res: agents.AgentResult):
            self._reset()
            if res.agent == "clipboard":
                self._copy_text(res.text)
                self.overlay.say("Task copied. Paste it into Claude Code, Hermes or OpenCode.", seconds=5)
            else:
                self._show_result(f"{agent.label}: proposed fix", res.text)
                self.overlay.say("Proposal ready for review. Nothing was changed.", seconds=6)

        self._run(lambda: agent.run(task), done, self._fail, AgentError)

    # --- approval ---------------------------------------------------------
    def _approved(self, pid: str):
        item = self.pending.pop(pid, None)
        if item is None:
            return
        if item == OFFER:
            self.investigate()
        elif isinstance(item, AgentTask):
            self._run_agent(item)
        elif isinstance(item, Draft):
            self.overlay.say("Sending...", seconds=None)
            self._run(lambda: self._send(item), lambda msg: self.overlay.say(msg, seconds=4), self._fail, DispatchError)

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


def _clipboard(text: str):
    from PySide6.QtWidgets import QApplication
    QApplication.clipboard().setText(text)
