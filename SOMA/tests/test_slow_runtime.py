"""Scheduler backpressure: slow OCR/model work never blocks the caller (plan section 9)."""

from __future__ import annotations

import threading
import time

from soma.config import SomaConfig, small_model_config
from soma.consent import ConsentManager, Scope
from soma.perception.synthetic import render_scenario
from soma.runtime.pipeline import SomaRuntime
from soma.runtime.scheduler import InferenceScheduler
from soma.types import ScreenFrame, Suggestion, WindowInfo


class SlowRuntime:
    """Fake runtime whose step takes ``delay`` seconds and tracks concurrency."""

    def __init__(self, delay: float = 0.2, silent: bool = False) -> None:
        self.delay = delay
        self.silent = silent
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self.paused = False
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self._lock = threading.Lock()

    def step(self, force: bool = False):
        with self._lock:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.set()
        try:
            self.release.wait(5)
            time.sleep(self.delay)
            from soma.runtime.pipeline import StepResult
            from soma.types import Candidate, ActionType, ScreenObservation
            win = WindowInfo("Code.exe", "")
            cand = None if self.silent else Candidate(ActionType.EXPLAIN_ERROR, "Explain")
            return StepResult(frame_version=self.calls, window=win,
                              observation=ScreenObservation(win, [], self.calls), candidates=[],
                              suggestion=Suggestion(cand, 0.9, self.calls), retrieved=[],
                              timings_ms={"ocr": self.delay * 1000, "total": self.delay * 1000})
        finally:
            with self._lock:
                self.concurrent -= 1

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


def wait_for(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_slow_step_does_not_block_caller():
    rt = SlowRuntime(delay=0.5)
    got = []
    sched = InferenceScheduler(rt, interval_s=10.0, on_result=got.append)
    sched.start()
    try:
        t0 = time.perf_counter()
        for _ in range(20):
            sched.trigger_now()
        assert time.perf_counter() - t0 < 0.1          # triggers return immediately
        assert wait_for(lambda: got)
    finally:
        sched.stop()


def test_at_most_one_step_in_flight_and_triggers_coalesce():
    rt = SlowRuntime(delay=0.05)
    rt.release.clear()
    got = []
    sched = InferenceScheduler(rt, interval_s=10.0, on_result=got.append)
    sched.start()
    try:
        sched.trigger_now()
        assert rt.started.wait(2)
        for _ in range(50):                              # pile up while busy
            sched.trigger_now()
        assert sched.in_flight == 1
        rt.release.set()
        assert wait_for(lambda: len(got) >= 2)
        time.sleep(0.2)
        assert rt.max_concurrent == 1
        assert sched.max_in_flight == 1
        assert rt.calls <= 3                             # 50 triggers coalesced
    finally:
        sched.stop()


def test_pause_drops_stale_in_flight_result():
    rt = SlowRuntime(delay=0.0)
    rt.release.clear()
    got = []
    sched = InferenceScheduler(rt, interval_s=10.0, on_result=got.append)
    sched.start()
    try:
        sched.trigger_now()
        assert rt.started.wait(2)
        sched.pause()                                    # result in flight becomes stale
        assert rt.paused
        rt.release.set()
        assert wait_for(lambda: sched.dropped_stale == 1)
        assert got == []
        sched.trigger_now()                              # paused: no new steps
        time.sleep(0.1)
        assert rt.calls == 1
        sched.resume()
        sched.trigger_now()
        assert wait_for(lambda: len(got) == 1)
    finally:
        sched.stop()


def test_latency_stats_recorded_and_interval_backs_off_when_silent():
    rt = SlowRuntime(delay=0.01, silent=True)
    got = []
    sched = InferenceScheduler(rt, interval_s=0.02, max_interval_s=0.08, on_result=got.append)
    sched.start()
    try:
        assert wait_for(lambda: len(got) >= 4)
    finally:
        sched.stop()
    summary = sched.stats.summary()
    assert summary["step"]["n"] >= 4
    assert summary["step"]["p95_ms"] >= summary["step"]["p50_ms"] >= 5.0
    assert "ocr" in summary
    assert sched.interval_s > 0.02                      # backed off
    assert sched.interval_s <= 0.08


def test_errors_do_not_kill_worker():
    class Boom(SlowRuntime):
        def step(self, force=False):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("ocr crashed")
            return super().step(force)

    rt = Boom(delay=0.0)
    got, errs = [], []
    sched = InferenceScheduler(rt, interval_s=10.0, on_result=got.append, on_error=errs.append)
    sched.start()
    try:
        sched.trigger_now()
        assert wait_for(lambda: sched.errors == 1)
        sched.trigger_now()
        assert wait_for(lambda: got)
        assert len(errs) == 1
    finally:
        sched.stop()


class SlowOCR:
    name = "slow"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    def recognize(self, image, window_id="", timestamp_ms=0):
        self.calls += 1
        time.sleep(self.delay)
        return []


def test_real_runtime_with_slow_ocr_runs_off_thread_and_records_latency():
    cfg = SomaConfig(device="cpu", model=small_model_config())
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    image, obs, _ = render_scenario("terminal_traceback", seed=1, size=(320, 200))

    class Capturer:
        version = 0
        last_skip_reason = ""
        last_window = obs.window

        def capture(self, force=False):
            self.version += 1
            return ScreenFrame(image=image, window=obs.window, version=self.version)

        def reset(self):
            pass

    from soma.runtime.telemetry import Telemetry
    ocr = SlowOCR(0.3)
    rt = SomaRuntime(cfg, consent, capturer=Capturer(), ocr=ocr, telemetry=Telemetry(enabled=True))
    got = []
    sched = InferenceScheduler(rt, interval_s=10.0, on_result=got.append)
    sched.start()
    try:
        t0 = time.perf_counter()
        sched.trigger_now()
        sched.trigger_now()
        assert time.perf_counter() - t0 < 0.1
        assert wait_for(lambda: got, 10)
    finally:
        sched.stop()
    assert got[0].timings_ms["ocr"] >= 250
    lat = rt.telemetry.export()["latency_ms"]
    assert lat["ocr"]["p50_ms"] >= 250 and lat["total"]["p95_ms"] >= lat["ocr"]["p50_ms"]


def test_telemetry_rates_percentiles_and_opt_in():
    from soma.runtime.telemetry import Telemetry, percentile

    assert percentile([], 50) == 0.0 and percentile([1, 2, 3, 4], 50) == 2.5
    t = [0.0]
    off = Telemetry(enabled=False, clock=lambda: t[0])
    off.record_step({"total": 5.0}, silent=False)
    assert off.export()["steps"] == 0                           # opt-in: nothing recorded
    tel = Telemetry(enabled=True, idle_gap_s=120.0, clock=lambda: t[0])
    for i in range(10):
        t[0] += 60.0
        tel.record_step({"ocr": 10.0 + i, "total": 20.0 + i, "secret_text": 1.0}, silent=i % 5 != 0)
    tel.record_step({}, silent=True, skipped_reason="excluded:auth-prompt:sign in to bank")
    out = tel.export()
    assert out["processed"] == 10 and out["silent"] == 8 and out["suggestions"] == 2
    assert out["silent_rate"] == 0.8
    assert out["active_s"] == 540.0                             # 9 gaps of 60 s
    assert abs(out["suggestions_per_active_hour"] - 2 / (540 / 3600)) < 1e-9
    assert out["skipped"] == {"excluded": 1}                    # no title detail kept
    assert "secret_text" not in out["latency_ms"] and out["latency_ms"]["ocr"]["n"] == 10
    t[0] += 10_000                                              # long idle gap is capped
    tel.record_step({"total": 1.0}, silent=True)
    assert tel.export()["active_s"] == 660.0
    tel.disable()
    assert tel.export()["steps"] == 0 and tel.export()["latency_ms"] == {}
