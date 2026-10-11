import pytest

from pet.agents import AgentError, AgentResult
from pet.analyst import Analysis
from pet.brain import Draft, OfflineBrain
from pet.context import ActiveFile, Context, Diagnostic
from pet.pipeline import Pipeline
from pet.screen_cv import Region, ScreenFindings
from pet.ui import Overlay
from pet.vision import EYES_CLOSED, FACE_AWAY, FACE_BACK, PINCH
from pet.watcher import InterventionPolicy, Signals, Watcher, diag_signal

ERR = Diagnostic(12, "error", "'NoneType' object is not subscriptable", "pytest")
AF = ActiveFile("C:/repo/dashboard.py", "python", "x = 1\n" * 30, workspace="C:/repo", diagnostics=(ERR,))
SCREEN = ScreenFindings([Region(5, 5, 120, 14, 90)], 0.001, b"\xff\xd8crop", 1920, 1080)
ANALYSIS = Analysis("Crash on unknown user", ("line 12 error",), "user is None", "guard None", 0.8, "ai")


class FakeAnalyst:
    def __init__(self):
        self.calls = []

    def analyze(self, ctx):
        self.calls.append(ctx)
        return ANALYSIS


class FakeAgent:
    def __init__(self, name="claude", label="Claude Code", exc=None):
        self.name, self.label, self.exc, self.tasks = name, label, exc, []

    def run(self, task):
        self.tasks.append(task)
        if self.exc:
            raise self.exc
        return AgentResult(self.name, "diff --git a/dashboard.py ...")


@pytest.fixture
def make(qapp, monkeypatch):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)

    def build(agent=None, brain=None):
        o = Overlay(tray=False)
        out = {"copied": [], "shown": []}
        agent = agent or FakeAgent()
        p = Pipeline(o, brain or OfflineBrain(), capture=lambda: Context(b"jpg", AF, SCREEN),
                     analyst=FakeAnalyst(), pick_agent=lambda name=None: agent,
                     copy_text=out["copied"].append, show_result=lambda t, x: out["shown"].append((t, x)),
                     use_voice=False, spawn=lambda f: f(), capture_delay_ms=0)
        return o, p, agent, out
    return build


def test_investigate_requires_approval_then_runs_agent(make):
    o, p, agent, out = make()
    p.investigate()
    ctx = p.analyst.calls[0]
    assert ctx.screen is SCREEN and ctx.active_file is AF          # CV findings + diagnostics reach analysis
    assert o.bubble.proposal.action_label == "Ask Claude Code" and agent.tasks == []
    assert "80% confident" in o.bubble.proposal.title
    o.bubble.primary.click()
    task = agent.tasks[0]
    assert task.screen_crop == b"\xff\xd8crop" and task.workspace == "C:/repo"
    assert out["shown"][0][1].startswith("diff --git") and "Nothing was changed" in o.bubble.body.text()
    o.bubble.primary.click()
    assert len(agent.tasks) == 1                                     # single-use approval


def test_cancel_never_reaches_agent(make):
    o, p, agent, out = make()
    p.investigate()
    o.bubble.cancel.click()
    assert agent.tasks == [] and p.pending == {}


def test_voice_request_routes_to_agent_with_request(make):
    o, p, agent, out = make()
    p.trigger()
    o.bubble.input.setText("why is this test failing, ask claude")
    o.bubble.primary.click()
    assert len(p.analyst.calls) == 1                                 # reused the hotkey-time screenshot
    o.bubble.primary.click()
    assert agent.tasks[0].user_request == "why is this test failing, ask claude"


def test_clipboard_agent_copies_task(make):
    o, p, agent, out = make(agent=FakeAgent("clipboard", "Copy task"))
    p.investigate()
    assert o.bubble.proposal.action_label == "Copy task"
    o.bubble.primary.click()
    assert out["copied"] and "Task copied" in o.bubble.body.text()


def test_agent_error_shown(make):
    o, p, agent, out = make(agent=FakeAgent(exc=AgentError("Claude Code timed out after 240s.")))
    p.investigate()
    o.bubble.primary.click()
    assert "timed out" in o.bubble.body.text() and not p.busy


def test_proactive_offer_asks_before_capturing(make):
    o, p, agent, out = make()
    p.offer_investigation("dashboard.py line 12: NoneType")
    assert p.analyst.calls == [] and o.bubble.proposal.action_label == "Investigate"
    o.bubble.primary.click()                                         # approve offer -> capture + analyse
    assert len(p.analyst.calls) == 1 and o.bubble.proposal.action_label == "Ask Claude Code"


def test_pause_during_agent_run_drops_result(make):
    o, p, agent, out = make()
    p.investigate()
    deferred = []
    p._spawn = deferred.append
    o.bubble.primary.click()
    o.set_paused(True)
    deferred.pop()()
    assert out["shown"] == [] and not o.bubble.isVisible()


# --- intervention policy -------------------------------------------------

def sig(t, diag="f|err", screen=None, **kw):
    return Signals(t, diag, "summary", screen, **kw)


def test_policy_needs_persistence_then_cools_down():
    pol = InterventionPolicy()
    assert pol.update(sig(0)) is None
    assert pol.update(sig(15)) is None
    assert pol.update(sig(21)) == ("diag:f|err", "summary")
    assert pol.update(sig(60)) is None                           # cooldown
    assert pol.update(sig(21 + 601)) is not None                 # after cooldown, same issue may be offered again


def test_policy_resets_when_issue_resolves_and_defers_when_away():
    pol = InterventionPolicy()
    pol.update(sig(0))
    pol.update(sig(10, diag=None))                               # fixed -> timer resets
    assert pol.update(sig(25)) is None
    assert pol.update(sig(46, user_present=False)) is None       # deferred, not consumed
    assert pol.update(sig(47, eyes_closed=True)) is None
    assert pol.update(sig(48, busy=True)) is None
    assert pol.update(sig(49)) is not None                       # offered once the user is back


def test_policy_screen_only_waits_longer_and_daily_cap():
    pol = InterventionPolicy()
    pol.update(sig(0, diag=None, screen="3:4"))
    assert pol.update(sig(30, diag=None, screen="3:4")) is None
    assert pol.update(sig(46, diag=None, screen="3:4"))[0] == "screen:3:4"
    pol = InterventionPolicy()
    hits = 0
    for i in range(20):
        pol.update(sig(i * 100, diag=f"f|e{i}"))
        hits += pol.update(sig(i * 100 + 30, diag=f"f|e{i}")) is not None
    assert hits == InterventionPolicy.DAILY_CAP


def test_diag_signal():
    fp, summary = diag_signal(AF)
    assert fp.startswith("C:/repo/dashboard.py|") and summary.startswith("dashboard.py line 12")
    assert diag_signal(None) == (None, "")


# --- webcam reactions ----------------------------------------------------

def test_vision_events_drive_presence_and_pinch(qapp):
    o = Overlay(tray=False)
    asks = []
    o.ask_requested.connect(lambda: asks.append(1))
    w = Watcher(o, lambda: False, read_file=lambda: None)
    w.on_vision_event(FACE_AWAY)
    assert o.pet.state == "away" and not w.present
    w.on_vision_event(FACE_BACK)
    w.on_vision_event(EYES_CLOSED)
    assert o.pet.state == "blink"
    w.on_vision_event(PINCH)
    assert asks == [1]                                            # pinch opens Pette


def test_pinch_never_touches_a_pending_proposal(make):
    o, p, agent, out = make()
    p.investigate()
    pid = o.bubble.proposal.id
    o.pinch()
    assert o.bubble.proposal.id == pid and agent.tasks == [] and pid in p.pending


def test_watcher_offers_from_diagnostics(qapp, monkeypatch):
    o = Overlay(tray=False)
    offers = []
    w = Watcher(o, lambda: False, read_file=lambda: AF)
    w.offer.connect(offers.append)
    clock = iter([1000.0, 1030.0])
    monkeypatch.setattr("pet.watcher.time.time", lambda: next(clock))
    w.poll()
    w.poll()
    assert offers and offers[0].startswith("dashboard.py line 12")
