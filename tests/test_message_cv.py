"""Webcam signals mapped onto the message workflow, plus world-model learning from that workflow."""
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
    p = Pipeline(o, OfflineBrain(), capture=lambda: Context(None, None), use_voice=False,
                 send=lambda d: sent.append(d) or "Sent.", configured=lambda _: True,
                 spawn=lambda f: f(), capture_delay_ms=0, memory=store, world=world)
    clock = Clock()
    w = Watcher(o, lambda: p.busy, read_file=lambda: None, memory=store, world=world, clock=clock)
    yield o, p, w, store, world, sent, clock
    store.close()


def type_command(o, text):
    o.bubble.input.setText(text)
    o.bubble.primary.click()


def test_pinch_is_push_to_talk(rig):
    o, p, w, *_ = rig
    w.on_vision_event(PINCH)
    assert p.busy and o.bubble.asking                      # same as pressing the hotkey


def test_pinch_on_draft_redictates_and_never_sends(rig):
    o, p, w, store, world, sent, _ = rig
    w.on_vision_event(PINCH)
    type_command(o, "Slack Sarah asking her to review the PR")
    first = o.bubble.proposal.id
    w.on_vision_event(PINCH)                               # re-dictate
    assert sent == [] and first not in p.pending and o.bubble.asking
    type_command(o, "Slack Sarah saying the PR is merged")
    o.bubble.primary.click()                               # only a click sends
    assert [d.message for d in sent] == ["Hi Sarah, the PR is merged. Thanks!"]
    assert store.query("SELECT outcome FROM messages ORDER BY ts") == [("drafted",), ("cancelled",), ("drafted",),
                                                                      ("sent",)]


def test_pinch_dismisses_plain_message(rig):
    o, p, w, *_ = rig
    o.say("hello", seconds=None)
    w.on_vision_event(PINCH)
    assert not o.bubble.isVisible()


def test_away_keeps_draft_and_nudges_on_return(rig):
    o, p, w, store, world, sent, clock = rig
    p.trigger()
    type_command(o, "Discord Ameya saying standup moved to 10")
    pid = o.bubble.proposal.id
    w.on_vision_event(FACE_AWAY)
    assert o.pet.state != "away" or o.bubble.proposal      # draft still showing while away
    clock.t = 42.0
    w.on_vision_event(FACE_BACK)
    assert o.bubble.proposal.id == pid and pid in p.pending and sent == []
    assert o.bubble.title.text().startswith("Still send?")
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


def test_gpt_draft_is_rerouted_to_usual_platform(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    store = Store(tmp_path / "c.sqlite3")
    store.log_message("sarah", "email", "sent")

    class StubBrain:
        contacts = {}

        def draft(self, command, ctx):
            return Draft("slack", "Sarah", "", "hi")             # model guessed slack

    o = Overlay(tray=False)
    p = Pipeline(o, StubBrain(), capture=lambda: Context(None, None), use_voice=False, configured=lambda _: True,
                 spawn=lambda f: f(), capture_delay_ms=0, memory=store, world=WorldModel(store))
    p.trigger()
    type_command(o, "tell Sarah hi")
    assert p.brain.contacts == {"sarah": "email"}                # contacts fed to the brain
    assert o.bubble.proposal.title == "Email (usual) to Sarah"
    store.close()


def test_contacts_reach_gpt_prompt():
    parts = build_messages("tell sarah hi", Context(None, None), {"sarah": "slack"})[1]["content"]
    assert any("Sarah (slack)" in part.get("text", "") for part in parts)
    assert Brain().contacts == {}


def test_offer_outcomes_feed_cooldown(rig):
    o, p, w, store, world, sent, _ = rig
    for _ in range(6):
        p.offer_investigation("x")
        o.bubble.cancel.click()
    assert world.offer_cooldown_s() == 1800
    w.poll()
    assert w.policy.COOLDOWN_S == 1800                          # watcher applies the learned back-off


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
    o.bubble.primary.click()
    assert sum(store.counts().values()) == 0
