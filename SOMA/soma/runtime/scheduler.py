"""Background inference scheduler: keeps OCR/model work off the UI thread.

* One dedicated daemon worker thread runs ``runtime.step()``; callers never
  block (``trigger_now`` only sets an event).
* **Backpressure:** at most one step is in flight. Triggers arriving while a
  step runs are coalesced into a single follow-up step.
* **Stale results are dropped:** ``pause()`` (and ``stop()``) bump a
  generation counter; a result computed under an older generation is never
  delivered.
* **Adaptive interval:** unchanged / skipped / silent steps back the interval
  off geometrically up to ``max_interval_s``; a shown suggestion resets it.
  ``trigger_now`` (e.g. on a user event) runs immediately with ``force=True``,
  which bypasses rate limit and change detection but never consent gates.

Pure Python threading, no Qt: ``on_result`` is called **on the worker
thread**, so a UI must marshal it to its own thread (e.g. a Qt signal).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Protocol

from soma.runtime.telemetry import LatencyStats

log = logging.getLogger(__name__)

#: Skip reasons that mean "nothing new on screen" (back off).
IDLE_SKIPS = ("unchanged", "rate-limited")


class _Runtime(Protocol):
    def step(self, force: bool = False): ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


class InferenceScheduler:
    """Run ``runtime.step`` periodically on a worker thread.

    Args:
        runtime: a :class:`~soma.runtime.pipeline.SomaRuntime` (or anything with
            ``step(force)``, ``pause()``, ``resume()``).
        interval_s: base interval between steps.
        on_result: callback ``(StepResult) -> None`` (worker thread).
        max_interval_s: back-off ceiling (default ``4 * interval_s``).
        backoff: multiplier applied after an idle/silent step.
        on_error: optional callback ``(Exception) -> None``; errors never kill the worker.
    """

    def __init__(self, runtime: _Runtime, interval_s: float = 1.0,
                 on_result: Callable[[object], None] | None = None, *,
                 max_interval_s: float | None = None, backoff: float = 1.5,
                 on_error: Callable[[BaseException], None] | None = None,
                 name: str = "soma-inference") -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.runtime = runtime
        self.base_interval_s = float(interval_s)
        self.max_interval_s = float(max_interval_s if max_interval_s is not None else 4 * interval_s)
        self.backoff = max(1.0, float(backoff))
        self.on_result = on_result
        self.on_error = on_error
        self.name = name
        self.stats = LatencyStats()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gen = 0
        self._paused = False
        self._forced = False
        self._in_flight = 0
        self._interval = self.base_interval_s
        # counters
        self.steps = 0
        self.delivered = 0
        self.dropped_stale = 0
        self.errors = 0
        self.max_in_flight = 0

    # ------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def generation(self) -> int:
        return self._gen

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def interval_s(self) -> float:
        """Current (adaptive) interval."""
        return self._interval

    # ------------------------------------------------------------ control
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        """Stop the worker; an in-flight result is discarded. Waits up to ``timeout``."""
        with self._lock:
            self._gen += 1
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)
        self._thread = None

    def pause(self) -> None:
        """Pause instantly: the runtime stops capturing and in-flight results go stale."""
        with self._lock:
            self._paused = True
            self._gen += 1
        self.runtime.pause()

    def resume(self) -> None:
        self.runtime.resume()
        with self._lock:
            self._paused = False
            self._gen += 1
            self._interval = self.base_interval_s
        self._wake.set()

    def trigger_now(self) -> None:
        """Request an immediate (forced) step. Never blocks; coalesces while busy."""
        with self._lock:
            self._forced = True
        self._wake.set()

    # ------------------------------------------------------------ worker
    def _adapt(self, result: object) -> None:
        reason = getattr(result, "skipped_reason", None)
        suggestion = getattr(result, "suggestion", None)
        silent = suggestion is None or getattr(suggestion, "is_silent", True)
        idle = (reason is not None and reason.split(":", 1)[0] in IDLE_SKIPS) or (reason is None and silent)
        with self._lock:
            if reason is None and not silent:
                self._interval = self.base_interval_s
            elif idle:
                self._interval = min(self.max_interval_s, self._interval * self.backoff)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._interval)
            if self._stop.is_set():
                break
            with self._lock:
                self._wake.clear()
                forced, self._forced = self._forced, False
                if self._paused:
                    continue
                gen = self._gen
                self._in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self._in_flight)
            t0 = time.perf_counter()
            result = None
            try:
                result = self.runtime.step(force=forced)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                self.errors += 1
                log.warning("SOMA step failed: %s", type(exc).__name__)
                if self.on_error is not None:
                    try:
                        self.on_error(exc)
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                with self._lock:
                    self._in_flight -= 1
                self.steps += 1
                self.stats.add("step", (time.perf_counter() - t0) * 1000.0)
            if result is None:
                continue
            with self._lock:
                stale = gen != self._gen or self._paused or self._stop.is_set()
            if stale:
                self.dropped_stale += 1
                continue
            timings = getattr(result, "timings_ms", None) or {}
            if getattr(result, "skipped_reason", None) is None:
                self.stats.add_all(timings)
            self._adapt(result)
            self.delivered += 1
            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception as exc:  # noqa: BLE001 - a UI bug must not stop inference
                    self.errors += 1
                    log.warning("on_result callback failed: %s", type(exc).__name__)
