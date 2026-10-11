"""Orchestrates Context -> Voice (ElevenLabs call or Whisper) -> AI -> UI -> Dispatch,
plus Active file -> Analysis -> Coding agent.

Lives on the Qt main thread; slow steps run in threads and come back through `_deliver`.
"""
import os
import threading
import time
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer, Signal

from . import agents, context, dispatch, eleven, voice
from .agents import AgentError, AgentTask
from .analyst import Analysis, Analyst
from .brain import PLATFORM_WORDS, Brain, BrainError, Draft
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
                 read_file=context.read_active_file,
                 record=voice.record_until_silence, transcribe=voice.transcribe, send=dispatch.dispatch,
                 configured=dispatch.configured, analyst: Analyst | None = None, pick_agent=agents.pick_agent,
                 copy_text=None, show_result=None, use_voice=True, spawn=_thread,
                 capture_delay_ms=CAPTURE_DELAY_MS, memory=None, world=None, voice_call=None):
        super().__init__()
        self.overlay = overlay
        self.memory, self.world = memory, world   # db.Store / db.WorldModel, optional
        self._command = ""
        self.brain = brain or Brain()
        self.analyst = analyst or Analyst()
        self._capture, self._record, self._transcribe = capture, record, transcribe
        self._read_file = read_file
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
        # ElevenLabs voice agent: factory(on_tool=, on_text=, on_state=, on_end=) -> eleven.Call-like object.
        self._make_call = voice_call
        self.call = None

        self._deliver.connect(self._on_deliver)
        self.hotkey_pressed.connect(self.trigger)
        self.investigate_hotkey.connect(self.investigate)
        overlay.ask_requested.connect(self.trigger)
        overlay.investigate_requested.connect(self.investigate)
        overlay.text_submitted.connect(self._heard)
        overlay.input_cancelled.connect(self._reset)
        overlay.approved.connect(self._approved)
        overlay.cancelled.connect(self._cancelled)
        overlay.pinch_on_proposal.connect(self._pinch_on_proposal)
        overlay.pause_toggled.connect(self._paused)

    @property
    def agent(self):
        return self._pick_agent(self.agent_name)

    # --- capture ----------------------------------------------------------
    def _capture_then(self, next_step):
        """Message flow: hide Pette, take one screenshot + the active file, show Pette, continue."""
        self.busy = True
        gen = self.gen
        was_visible = self.overlay.isVisible()
        self.overlay.hide()

        def after_capture():
            if gen != self.gen:
                return
            self.ctx = self._capture()
            if was_visible:
                self.overlay.reveal()
            next_step()

        if self._delay:
            QTimer.singleShot(self._delay, after_capture)
        else:
            after_capture()

    # --- voice / message flow ---------------------------------------------
    def _can_start(self) -> bool:
        return not (self.overlay.paused or self.busy) and self.overlay.context_allowed

    def trigger(self):
        if self._in_call():
            self.call.stop()   # the hotkey toggles the voice call: second press hangs up
            return
        if not self._can_start():
            return
        if self._make_call is not None and self.use_voice:
            self._start_call()
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
        self._command = command
        if self.world is not None and hasattr(self.brain, "contacts"):
            self.brain.contacts = dict(self.world.known_contacts())
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
        learned = ""
        if self.world is not None and not PLATFORM_WORDS.search(self._command):
            usual = self.world.preferred_platform(d.recipient)
            if usual and usual != d.platform:   # you didn't name a platform: use the one you usually use
                d = replace(d, platform=usual)
            learned = " (usual)" if usual else ""
        self._remember_message(d, "drafted")
        title = f"{d.platform.capitalize()}{learned} to {d.recipient}"
        if not self._configured(d.platform):
            title += " (not configured)"
        if label:
            title = f"{label} · {title}"
        body = f"Subject: {d.subject}\n\n{d.message}" if d.platform == "email" and d.subject else d.message
        p = Proposal(title=title, body=body, action_label="Send")
        self.pending[p.id] = d
        self.overlay.propose(p)

    # --- ElevenLabs voice call (Flow-chan) ----------------------------------
    def _in_call(self) -> bool:
        return self.call is not None and self.call.active

    def _bubble_free(self) -> bool:
        """Agent chatter never covers a pending approval or the typed-input box."""
        return self.overlay.bubble.proposal is None and not self.overlay.bubble.asking

    def _start_call(self):
        gen = self.gen
        self.busy = True   # proactive offers wait while you're talking

        def deliver(fn):
            self._deliver.emit(gen, fn)

        self.call = self._make_call(
            on_tool=lambda name, params: self._call_tool(gen, name, params),
            on_text=lambda text: deliver(lambda: self._call_said(text)),
            on_state=lambda state: deliver(lambda: self._call_state(state)),
            on_end=lambda err: deliver(lambda: self._call_ended(err)))
        self.overlay.say("Calling Flow-chan... press the hotkey again to hang up.", title="Voice", seconds=None)
        self.overlay.set_state("thinking")
        self.call.start(eleven.context_update(self._read_file()))

    def _call_tool(self, gen: int, name: str, params: dict) -> str:
        """Runs on the call's reader thread: validate here, show UI via _deliver. Never sends anything."""
        if name == eleven.MESSAGE_TOOL:
            d = eleven.tool_draft(params)        # BrainError -> the agent is told to fix its arguments
            self._deliver.emit(gen, lambda: self._call_drafted(d))
            return eleven.PROPOSED
        if name == eleven.INVESTIGATE_TOOL:
            request = str(params.get("request", "")).strip()[:500]
            self._deliver.emit(gen, lambda: self._call_investigate(request))
            return ("Pette is analysing the open file and will show a proposal for the coding agent. "
                    "Nothing runs until the user approves.")
        raise BrainError(f"Unknown tool {name!r}.")

    def _call_drafted(self, d: Draft):
        self._command = d.platform   # the agent named the platform: no "usual platform" override
        self._drafted(d)

    def _call_investigate(self, request: str):
        self.ctx = context.Context(None, self._read_file())
        self._analyze(request=request)

    def _call_said(self, text: str):
        if self._bubble_free():
            self.overlay.say(text, title="Flow-chan", seconds=None)

    def _call_state(self, state: str):
        if self._bubble_free():
            self.overlay.set_state(state)

    def _call_ended(self, err):
        self.call = None
        self.busy = False
        if not self._bubble_free():
            return
        if err:
            self.overlay.ask_text(f"{err} Type your command instead:")
        else:
            self.overlay.say("Call ended.", title="Voice", seconds=3)

    def _tell_call(self, text: str):
        if self._in_call():
            self.call.tell(text)

    # --- active file -> analysis -> coding agent ---------------------------
    def investigate(self):
        """Explicit request (tray/hotkey/approved offer): analyse the active editor file. No screenshot."""
        if not self._can_start():
            return
        self.ctx = context.Context(None, self._read_file())
        self._analyze()

    def offer_investigation(self, reason: str):
        """Proactive: ask first. Nothing is uploaded or sent to an agent unless the user clicks Investigate."""
        if not self._can_start() or self.overlay.bubble.isVisible():
            return
        p = Proposal(title="Looks like you're stuck", body=f"{reason}\n\nWant me to look at this file and "
                     f"prepare a task for {self.agent.label}?", action_label="Investigate")
        self.pending[p.id] = OFFER
        self._remember_offer("proactive", "shown")
        self.overlay.propose(p)

    def _analyze(self, request: str = ""):
        self.busy = True
        ctx = self.ctx or context.Context(None, None)
        af = ctx.active_file
        self.overlay.say(f"Looking at {os.path.basename(af.path)}..." if af
                         else "No active file from VS Code; analysing without one...", seconds=None)
        self.overlay.set_state("thinking")
        self._run(lambda: self.analyst.analyze(ctx), lambda a: self._analyzed(a, ctx, request), self._fail, BrainError)

    def _analyzed(self, a: Analysis, ctx: context.Context, request: str):
        self._reset()
        agent = self.agent
        task = AgentTask.from_context(a, ctx, request)
        tag = "" if a.source == "ai" else " · diagnostics only"
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
            self._remember_offer("proactive", "accepted")
            self.investigate()
        elif isinstance(item, AgentTask):
            self._remember_offer("agent", "accepted")
            self._run_agent(item)
        elif isinstance(item, Draft):
            self.overlay.say("Sending...", seconds=None)

            def sent(msg):
                self._remember_message(item, "sent")
                self._tell_call(f"The user clicked Send. {msg}")
                self.overlay.say(msg, seconds=4)

            def failed(e):
                self._remember_message(item, "failed")
                self._tell_call(f"The user clicked Send but it failed: {e}")
                self._fail(e)

            self._run(lambda: self._send(item), sent, failed, DispatchError)

    def _cancelled(self, pid: str):
        item = self.pending.pop(pid, None)
        if item == OFFER:
            self._remember_offer("proactive", "declined")
        elif isinstance(item, AgentTask):
            self._remember_offer("agent", "declined")
        elif isinstance(item, Draft):
            self._remember_message(item, "cancelled")
            self._tell_call(f"The user cancelled the {item.platform} draft to {item.recipient}. Nothing was sent.")

    def _pinch_on_proposal(self, pid: str):
        """Pinch on a pending message draft = discard it and dictate again. Never sends anything."""
        item = self.pending.get(pid)
        if not isinstance(item, Draft):
            return  # agent tasks and offers are not touched by gestures
        self.pending.pop(pid)
        self._remember_message(item, "cancelled")
        self.overlay.dismiss()
        if self._in_call():   # already talking: let Flow-chan redo it instead of starting a new call
            self.call.tell("The user discarded that draft with a gesture. Ask what to change. Nothing was sent.")
            return
        self.trigger()

    # --- memory (facts only: no message text, screenshots or code) --------
    def _remember_message(self, d: Draft, outcome: str):
        if self.memory is not None and d.platform in ("slack", "discord", "email"):
            self.memory.log_message(d.recipient, d.platform, outcome)

    def _remember_offer(self, kind: str, outcome: str):
        if self.memory is not None:
            self.memory.log_offer(kind, outcome)

    def _fail(self, msg):
        self._reset()
        self.overlay.say(str(msg), title="Couldn't finish", seconds=6)

    def _reset(self):
        self.busy = self._in_call()   # an open voice call still counts as busy
        self.ctx = None  # drop the screenshot as soon as it's no longer needed

    def _paused(self, paused: bool):
        if paused:
            if self.call is not None:
                self.call.stop()   # pause hangs up; the mic is released at once
                self.call = None
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
