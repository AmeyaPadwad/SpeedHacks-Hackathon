import pytest

from pet.brain import BrainError, Draft
from pet.context import Context
from pet.dispatch import DispatchError
from pet.pipeline import Pipeline
from pet.ui import Overlay
from pet.voice import VoiceError


class FakeBrain:
    def __init__(self, result):
        self.result, self.calls = result, []

    def draft(self, command, ctx):
        self.calls.append((command, ctx))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


SLACK = Draft("slack", "Sarah", "", "please review auth.py")


@pytest.fixture
def make(qapp, monkeypatch):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)

    def build(result=SLACK, record=lambda: b"wav", transcribe=lambda w: "slack sarah", sent=None, send_exc=None):
        o = Overlay(tray=False)
        sent = [] if sent is None else sent

        def send(d):
            if send_exc:
                raise send_exc
            sent.append(d)
            return "Sent."

        p = Pipeline(o, FakeBrain(result), capture=lambda: Context(b"jpg", None), record=record,
                     transcribe=transcribe, send=send, configured=lambda _: True,
                     spawn=lambda f: f(), capture_delay_ms=0)
        return o, p, sent
    return build


def test_happy_path_requires_approval(make):
    o, p, sent = make()
    p.trigger()
    assert p.brain.calls[0] == ("slack sarah", Context(b"jpg", None))
    assert o.bubble.proposal and o.bubble.proposal.title == "Slack to Sarah"
    assert sent == [] and p.ctx is None and not p.busy
    o.bubble.primary.click()
    o.bubble.primary.click()
    assert sent == [SLACK]


def test_cancel_sends_nothing(make):
    o, p, sent = make()
    p.trigger()
    o.bubble.cancel.click()
    assert sent == [] and p.pending == {}


def test_answer_platform_none_shows_text(make):
    o, p, sent = make(result=Draft("none", "", "", "It's a null check."))
    p.trigger()
    assert o.bubble.proposal is None and "null check" in o.bubble.body.text()


def test_voice_failure_falls_back_to_typing(make):
    def no_mic():
        raise VoiceError("Microphone unavailable.")
    o, p, sent = make(record=no_mic)
    p.trigger()
    assert o.bubble.asking and "Microphone unavailable" in o.bubble.body.text()
    o.bubble.input.setText("email bob")
    o.bubble.primary.click()
    assert p.brain.calls[0][0] == "email bob"
    assert o.bubble.proposal is not None


def test_brain_error_shown(make):
    o, p, sent = make(result=BrainError("AI request failed: Timeout"))
    p.trigger()
    assert "Timeout" in o.bubble.body.text() and not p.busy


def test_dispatch_error_shown(make):
    o, p, sent = make(send_exc=DispatchError("Webhook returned HTTP 403"))
    p.trigger()
    o.bubble.primary.click()
    assert "403" in o.bubble.delivery_status.text()
    assert o.bubble.proposal and p.pending and not p.busy


def test_pause_drops_late_results(make):
    o, p, sent = make()
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


def test_busy_ignores_retrigger(make):
    o, p, sent = make()
    p._spawn = lambda f: None
    p.trigger()
    p.trigger()
    assert p.busy
