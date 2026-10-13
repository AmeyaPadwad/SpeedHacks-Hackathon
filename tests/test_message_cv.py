"""Webcam signals mapped onto the one workflow, plus world-model learning from it."""
import pytest

from db import Store, WorldModel
from pet.brain import Brain, Draft, OfflineBrain, build_messages
from pet.context import Context
from pet.pipeline import Pipeline
from pet.ui import Overlay
from pet.vision import (EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH, BaselineMeter, Observation,
                        VisionPolicy)
from pet.watcher import Watcher


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def rig(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    store = Store(tmp_path / "c.sqlite3")
    world = WorldModel(store)
    o = Overlay(tray=False)
    sent = []
    p = Pipeline(o, OfflineBrain(), capture=lambda: Context(None), use_voice=False,
                 send=lambda d: sent.append(d) or "Sent.", configured=lambda _: True,
                 spawn=lambda f: f(), memory=store, world=world, send_delay_s=3)
    clock = Clock()
    w = Watcher(o, memory=store, world=world, clock=clock)
    yield o, p, w, store, world, sent, clock
    store.close()


def type_command(o, text):
    o.bubble.input.setText(text)
    o.bubble.primary.click()


def finish(p):
    while p._countdown.isActive():
        p._tick()


def test_pinch_while_idle_is_ignored(rig):
    o, p, w, *_ = rig
    w.on_vision_event(PINCH)
    assert not p.busy and not o.bubble.asking              # only clicking Pette / the hotkey starts listening


def test_pinch_sends_pending_now_and_cancel_still_stops(rig):
    o, p, w, store, world, sent, _ = rig
    p.trigger()
    type_command(o, "Slack Sarah asking her to review the PR")
    o.bubble.cancel.click()                                # Cancel during the countdown
    finish(p)
    assert sent == [] and p.state == "cancelled"
    p.trigger()
    type_command(o, "Slack Sarah saying the PR is merged")
    w.on_vision_event(PINCH)                               # pinch = Send now
    finish(p)                                              # countdown runs out -> dispatched
    assert [d.message for d in sent] == ["Hi Sarah, the PR is merged. Thanks!"]
    assert store.query("SELECT outcome FROM messages ORDER BY ts") == [("drafted",), ("cancelled",), ("drafted",),
                                                                      ("sent",)]


def test_pinch_dismisses_plain_message(rig):
    o, p, w, *_ = rig
    o.say("hello", seconds=None)
    w.on_vision_event(PINCH)
    assert not o.bubble.isVisible()


def test_away_and_back_nudges_pending_send(rig):
    o, p, w, store, world, sent, clock = rig
    p.trigger()
    type_command(o, "Discord Ameya saying standup moved to 10")
    pid = o.bubble.proposal.id
    w.on_vision_event(FACE_AWAY)
    clock.t = 42.0
    w.on_vision_event(FACE_BACK)
    assert o.bubble.proposal.id == pid and o.bubble.title.text().startswith("Still sending?")
    assert store.query("SELECT duration FROM events WHERE kind = 'away_ended'") == [(42.0,)]


def test_eye_closure_duration_logged(rig):
    o, p, w, store, world, sent, clock = rig
    w.on_vision_event(EYES_CLOSED)
    clock.t = 1.25
    w.on_vision_event(EYES_OPEN)
    assert store.query("SELECT kind, duration FROM events WHERE kind LIKE 'eyes_closed%' ORDER BY ts") == [
        ("eyes_closed", None), ("eyes_closed_ended", 1.25)]


def test_learned_platform_routes_message_without_platform_word(rig):
    o, p, w, store, world, sent, _ = rig
    for _ in range(2):
        store.log_message("sarah", "discord", "sent")
    p.trigger()
    type_command(o, "message Sarah asking her to join the call")
    assert o.bubble.proposal.title.endswith("Discord (usual) to Sarah")


def test_named_platform_beats_learned_one(rig):
    o, p, w, store, world, sent, _ = rig
    store.log_message("sarah", "discord", "sent")
    p.trigger()
    type_command(o, "Slack Sarah asking her to join the call")
    assert o.bubble.proposal.title.endswith(" Slack to Sarah")


def test_model_draft_is_rerouted_to_usual_platform(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    store = Store(tmp_path / "c.sqlite3")
    store.log_message("sarah", "email", "sent")

    class StubBrain:
        contacts = {}

        def draft(self, command, ctx):
            return Draft("slack", "Sarah", "", "hi")             # model guessed slack

    o = Overlay(tray=False)
    p = Pipeline(o, StubBrain(), capture=lambda: Context(None), use_voice=False, configured=lambda _: True,
                 spawn=lambda f: f(), memory=store, world=WorldModel(store))
    p.trigger()
    type_command(o, "tell Sarah hi")
    assert p.brain.contacts == {"sarah": "email"}                # contacts fed to the brain
    assert o.bubble.proposal.title == "Email (usual) to Sarah"
    store.close()


def test_contacts_reach_prompt():
    text = build_messages("tell sarah hi", Context(None), {"sarah": "slack"})[1]["content"]
    assert "Sarah (slack)" in text
    assert Brain().contacts == {}


def test_world_model_sets_the_send_countdown(rig):
    o, p, w, store, world, sent, _ = rig
    p._send_delay_s = None                                      # use the world model
    assert p.delay_s() == 4.0
    for outcome in ["cancelled"] * 3 + ["sent"] * 3:
        store.log_message("sarah", "slack", outcome)
    assert p.delay_s() == 6.0                                   # you cancel a lot: more time to catch it
    for _ in range(60):
        store.log_message("sarah", "slack", "sent")
    assert p.delay_s() == 2.5                                   # you almost never cancel: snappier


def test_calibrated_policy_and_baseline_meter():
    pol = VisionPolicy(blink_on=0.75)
    assert (pol.BLINK_ON, pol.BLINK_OFF) == (0.75, 0.55)
    frames = [Observation(t / 10, True, 0.7, 0.7) for t in range(10)]
    assert not any(pol.update(o) for o in frames)              # 0.7 no longer counts as closed
    meter = BaselineMeter()
    out = [meter.add(Observation(t / 10, True, 0.1 + (t % 10) / 100, 0.1), False) for t in range(305)]
    summary = [s for s in out if s][0]
    assert summary["samples"] >= 290 and 0.1 <= summary["mean"] <= 0.2 and summary["p95"] <= 0.19


def test_memory_off_logs_nothing(rig):
    o, p, w, store, *_ = rig
    store.enabled = False
    w.on_vision_event(PINCH)
    type_command(o, "Slack Sarah saying hi")
    finish(p)
    assert sum(store.counts().values()) == 0
