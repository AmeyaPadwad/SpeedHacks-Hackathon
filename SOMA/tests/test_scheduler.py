"""InferenceScheduler lifecycle and adaptive interval, plus a real synthetic runtime run."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from soma.config import SomaConfig, small_model_config
from soma.consent import ConsentManager, Scope
from soma.runtime.factory import build_runtime
from soma.runtime.scheduler import InferenceScheduler
from soma.types import Suggestion


def wait_for(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class Result:
    def __init__(self, silent: bool, skipped: str | None = None) -> None:
        self.skipped_reason = skipped
        self.suggestion = Suggestion(None if silent else object(), 0.9, 1)  # type: ignore[arg-type]
        self.timings_ms = {"total": 1.0}


class ScriptedRuntime:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls = 0
        self.forced = []
        self.paused = False
        self.lock = threading.Lock()

    def step(self, force=False):
        with self.lock:
            self.calls += 1
            self.forced.append(force)
            return self.results[min(self.calls - 1, len(self.results) - 1)]

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


def test_start_stop_idempotent_and_thread_is_daemon():
    s = InferenceScheduler(ScriptedRuntime([Result(True)]), interval_s=10.0)
    s.start()
    s.start()
    assert s.running and s._thread.daemon
    s.stop()
    s.stop()
    assert not s.running


def test_trigger_now_forces_and_periodic_steps_do_not():
    rt = ScriptedRuntime([Result(True)])
    s = InferenceScheduler(rt, interval_s=0.05)
    s.start()
    try:
        assert wait_for(lambda: rt.calls >= 2)
        s.trigger_now()
        assert wait_for(lambda: True in rt.forced)
    finally:
        s.stop()
    assert rt.forced[0] is False


def test_interval_backs_off_on_unchanged_and_resets_on_suggestion():
    rt = ScriptedRuntime([Result(True, "unchanged")] * 6 + [Result(False)])
    got = []
    s = InferenceScheduler(rt, interval_s=0.01, max_interval_s=0.05, backoff=2.0, on_result=got.append)
    s.start()
    try:
        assert wait_for(lambda: len(got) >= 7)
    finally:
        s.stop()
    assert s.interval_s == 0.01   # last result was a shown suggestion


def test_pause_stops_stepping_and_resume_restarts():
    rt = ScriptedRuntime([Result(True)])
    s = InferenceScheduler(rt, interval_s=0.02)
    s.start()
    try:
        assert wait_for(lambda: rt.calls >= 1)
        s.pause()
        assert rt.paused and s.paused
        time.sleep(0.05)
        n = rt.calls
        time.sleep(0.15)
        assert rt.calls == n
        s.resume()
        assert not rt.paused
        assert wait_for(lambda: rt.calls > n)
    finally:
        s.stop()


def test_scheduler_drives_real_synthetic_runtime(tmp_path: Path):
    cfg = SomaConfig(device="cpu", model=small_model_config())
    cfg.capture.min_interval_s = 0.0
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    rt = build_runtime(cfg, consent, data_dir=tmp_path / "data", synthetic=True, telemetry=True,
                       synthetic_dwell_s=0.05)
    got = []
    s = InferenceScheduler(rt, interval_s=0.02, max_interval_s=0.05, on_result=got.append)
    s.start()
    try:
        assert wait_for(lambda: sum(r.skipped_reason is None for r in got) >= 3, 30)
        s.pause()
        assert rt.paused
        assert rt.step().skipped_reason == "paused"
    finally:
        s.stop()
        rt.memory.close()
    processed = [r for r in got if r.skipped_reason is None]
    assert len({r.window.app_name for r in processed}) >= 2       # scenarios cycle
    assert rt.telemetry.export()["processed"] >= 3
