"""RuntimeBridge: worker results reach the main thread; stale ones are dropped."""

import threading

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from soma.ui.desktop.bridge import RuntimeBridge, RuntimeLike


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class FakeRuntime:
    def __init__(self):
        self._paused = False
        self.calls = []

    def step(self, force=False):
        self.calls.append(("step", force))
        return {"step": len(self.calls)}

    def feedback(self, suggestion_id, outcome, edited_text=None):
        self.calls.append(("feedback", suggestion_id, outcome, edited_text))

    def pause(self):
        self._paused = True
        self.calls.append(("pause",))

    def resume(self):
        self._paused = False
        self.calls.append(("resume",))

    def current_suggestion_valid(self, current_version):
        return True

    @property
    def paused(self):
        return self._paused


class FakeScheduler:
    def __init__(self, runtime, interval_s, on_result):
        self.runtime, self.interval_s, self.on_result = runtime, interval_s, on_result
        self.events = []

    def start(self): self.events.append("start")
    def stop(self): self.events.append("stop")
    def pause(self): self.events.append("pause")
    def resume(self): self.events.append("resume")
    def trigger_now(self): self.events.append("trigger")


def make_bridge():
    scheds = []

    def factory(rt, interval, cb):
        s = FakeScheduler(rt, interval, cb)
        scheds.append(s)
        return s

    rt = FakeRuntime()
    return RuntimeBridge(rt, 0.5, scheduler_factory=factory), rt, scheds


def emit_from_thread(bridge, payload):
    t = threading.Thread(target=bridge.on_worker_result, args=(payload,))
    t.start()
    t.join()


def test_fake_runtime_matches_protocol():
    assert isinstance(FakeRuntime(), RuntimeLike)


def test_off_until_started(qapp):
    bridge, _, scheds = make_bridge()
    assert bridge.state == "Off" and not scheds


def test_worker_result_delivered_on_main_thread(qapp):
    bridge, _, scheds = make_bridge()
    got = []
    bridge.result.connect(lambda r: got.append((r, threading.current_thread() is threading.main_thread())))
    bridge.start()
    assert scheds[0].events == ["start"] and bridge.state == "Capturing"
    emit_from_thread(bridge, "r1")
    assert got == []  # queued, not delivered synchronously from the worker
    QCoreApplication.processEvents()
    assert got == [("r1", True)]


def test_results_after_pause_are_dropped(qapp):
    bridge, rt, scheds = make_bridge()
    got = []
    bridge.result.connect(got.append)
    bridge.start()
    emit_from_thread(bridge, "before")   # in flight when the user pauses
    bridge.pause()
    emit_from_thread(bridge, "after")
    QCoreApplication.processEvents()
    assert got == []
    assert bridge.dropped == 2
    assert ("pause",) in rt.calls and "pause" in scheds[0].events
    assert bridge.state == "Paused"


def test_resume_delivers_new_results_only(qapp):
    bridge, rt, _ = make_bridge()
    got = []
    bridge.result.connect(got.append)
    bridge.start()
    emit_from_thread(bridge, "stale")
    bridge.pause()
    bridge.resume()
    emit_from_thread(bridge, "fresh")
    QCoreApplication.processEvents()
    assert got == ["fresh"]
    assert ("resume",) in rt.calls


def test_toggle_and_state_signal(qapp):
    bridge, _, _ = make_bridge()
    states = []
    bridge.state_changed.connect(states.append)
    bridge.start()
    bridge.toggle()
    bridge.toggle()
    bridge.stop()
    assert states == ["Capturing", "Paused", "Capturing", "Off"]


def test_step_sync_skipped_when_paused(qapp):
    bridge, rt, _ = make_bridge()
    bridge.start()
    assert bridge.step_sync() == {"step": 1}
    bridge.pause()
    assert bridge.step_sync() is None
