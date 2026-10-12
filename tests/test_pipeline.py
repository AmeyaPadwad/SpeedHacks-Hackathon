import pytest

from pet.brain import BrainError, Draft
from pet.context import ActiveFile, Context
from pet.dispatch import DispatchError
from pet.pipeline import Pipeline
from pet.ui import Overlay
from pet.voice import VoiceError

AF = ActiveFile("C:/repo/auth.py", "python", "def login(): ...\n")
CTX = Context(AF, "--- a/auth.py\n+++ b/auth.py\n+    check_token()")


class FakeBrain:
    def __init__(self, result):
        self.result, self.calls = result, []

    def draft(self, command, ctx):
        self.calls.append((command, ctx))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


SLACK = Draft("slack", "Sarah", "", "I added a token check to login.", code=CTX.changes,
              code_label="auth.py · uncommitted changes", code_lang="diff")


@pytest.fixture
def make(qapp, monkeypatch):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)

    def build(result=SLACK, heard="slack sarah what I changed", sent=None, send_exc=None, listen=None,
              configured=True, delay=3):
        o = Overlay(tray=False)
        sent = [] if sent is None else sent
        states = []

        def send(d):
            if send_exc:
                raise send_exc
            sent.append(d)
            return f"Sent to {d.recipient} on Slack."

        def default_listen(on_level=None, register_cancel=None):
            on_level(0.5)
            return heard

        p = Pipeline(o, FakeBrain(result), capture=lambda: CTX, listen=listen or default_listen, send=send,
                     configured=lambda _: configured, spawn=lambda f: f(), send_delay_s=delay)
        p.status.connect(lambda s, d: states.append(s))
        return o, p, sent, states
    return build


def finish_countdown(p):
    while p._countdown.isActive():
        p._tick()


def test_one_workflow_listen_transcribe_draft_countdown_send(make):
    o, p, sent, states = make()
    p.trigger()                                          # = clicking Pette
    assert p.brain.calls[0] == ("slack sarah what I changed", CTX)   # file + diff, no screenshot
    assert o.bubble.proposal.countdown and "Slack to Sarah" in o.bubble.proposal.title
    assert "auth.py · uncommitted changes" in o.bubble.body.text() and sent == []
    assert p.state == "pending" and o.orb.label() == "Sending in 3s" and "Cancel · 3s" in o.bubble.cancel.text()
    finish_countdown(p)
    assert sent == [SLACK] and sent[0].code.startswith("--- a/auth.py")
    assert states == ["listening", "thinking", "pending", "pending", "pending", "sending", "sent"]
    assert o.orb.label() == "Sent ✓" and not p.busy


def test_send_now_skips_the_wait_and_is_single_use(make):
    o, p, sent, _ = make()
    p.trigger()
    o.bubble.primary.click()                             # "Send now"
    o.bubble.primary.click()
    finish_countdown(p)
    assert sent == [SLACK]


def test_cancel_during_countdown_sends_nothing(make):
    o, p, sent, states = make()
    p.trigger()
    o.bubble.cancel.click()
    finish_countdown(p)
    assert sent == [] and p.pending == {} and states[-1] == "cancelled" and o.orb.label() == "Not sent"


def test_pinch_cancels_a_pending_send(make):
    o, p, sent, _ = make()
    p.trigger()
    o.pinch()
    finish_countdown(p)
    assert sent == [] and p.state == "cancelled"


def test_transcript_shows_in_bubble_before_groq(make):
    o, p, sent, _ = make()
    seen = []
    p.brain = type("B", (), {"draft": lambda self, c, ctx: (seen.append(o.bubble.body.text()), SLACK)[1]})()
    p.trigger()
    assert seen == ["“slack sarah what I changed”"]


def test_answer_platform_none_shows_text(make):
    o, p, sent, _ = make(result=Draft("none", "", "", "It's a null check."))
    p.trigger()
    assert o.bubble.proposal is None and "null check" in o.bubble.body.text() and p.state == "idle"


def test_unconfigured_platform_is_not_sent(make):
    o, p, sent, states = make(configured=False)
    p.trigger()
    assert sent == [] and states[-1] == "failed" and "isn't configured" in o.bubble.body.text()


def test_voice_failure_falls_back_to_typing(make):
    def no_mic(on_level=None, register_cancel=None):
        raise VoiceError("Microphone unavailable.")
    o, p, sent, _ = make(listen=no_mic)
    p.trigger()
    assert o.bubble.asking and "Microphone unavailable" in o.bubble.body.text()
    o.bubble.input.setText("email bob")
    o.bubble.primary.click()
    assert p.brain.calls[0][0] == "email bob" and o.bubble.proposal is not None


def test_click_while_listening_stops(make):
    cancelled = []

    def slow(on_level=None, register_cancel=None):
        register_cancel(lambda: cancelled.append(1))
        return "x"
    o, p, sent, _ = make(listen=slow)
    p._spawn = lambda f: None            # listener "running"
    p.trigger()
    p.cancel_listen = lambda: cancelled.append(1)
    p.trigger()                          # second click
    assert cancelled == [1]


def test_brain_error_shown(make):
    o, p, sent, states = make(result=BrainError("AI request failed: Timeout"))
    p.trigger()
    assert "Timeout" in o.bubble.body.text() and not p.busy and states[-1] == "failed"


def test_dispatch_error_shown(make):
    o, p, sent, states = make(send_exc=DispatchError("Webhook returned HTTP 403"))
    p.trigger()
    finish_countdown(p)
    assert "403" in o.bubble.body.text() and states[-1] == "failed" and o.orb.label() == "Not sent"


def test_pause_drops_late_results_and_stops_countdown(make):
    o, p, sent, _ = make()
    deferred = []
    p._spawn = deferred.append
    p.trigger()                 # listening; voice work is queued, not run
    o.set_paused(True)
    deferred.pop()()            # late voice result arrives after pause
    assert p.brain.calls == [] and not o.bubble.isVisible()
    o.set_paused(False)
    p._spawn = lambda f: f()
    p.trigger()
    assert o.bubble.proposal is not None
    o.set_paused(True)
    finish_countdown(p)
    assert sent == []


def test_busy_ignores_retrigger(make):
    o, p, sent, _ = make()
    p._spawn = lambda f: None
    p.trigger()
    p.trigger()
    assert p.busy and p.state == "listening"


def test_errors_from_a_real_worker_thread_reach_the_bubble(make):
    """Regression: the error lambda ran after `except` unbound `e` (NameError), so failures never showed."""
    import threading
    import time

    from PySide6.QtWidgets import QApplication

    def no_mic(on_level=None, register_cancel=None):
        raise VoiceError("Microphone unavailable.")
    o, p, sent, _ = make(listen=no_mic)
    p._spawn = lambda f: threading.Thread(target=f, daemon=True).start()   # queued delivery, like the app
    p.trigger()
    deadline = time.time() + 3
    while not o.bubble.asking and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert o.bubble.asking and "Microphone unavailable" in o.bubble.body.text()
