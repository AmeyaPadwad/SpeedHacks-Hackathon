"""Opt-in, local-only runtime metrics (plan sections 9 and 12).

Records per-stage latencies (P50/P95), step counts, the silent rate and
suggestions per active hour. It stores numbers and stage names only: never
screen text, window titles, suggestion text or images. Nothing leaves the
process; :meth:`Telemetry.export` returns a plain dict the caller may show or
save. When ``enabled`` is False every ``record_*`` call is a no-op.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Iterable

STAGES: tuple[str, ...] = ("capture", "ocr", "parse", "encode", "retrieve", "policy", "gate", "total")


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]); 0.0 for no values."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * min(100.0, max(0.0, q)) / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class LatencyStats:
    """Thread-safe bounded sample windows per named stage."""

    def __init__(self, window: int = 512) -> None:
        self.window = int(window)
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = {}

    def add(self, stage: str, ms: float) -> None:
        with self._lock:
            self._samples.setdefault(stage, deque(maxlen=self.window)).append(float(ms))

    def add_all(self, timings: dict[str, float]) -> None:
        for k, v in timings.items():
            self.add(k, v)

    def count(self, stage: str) -> int:
        with self._lock:
            return len(self._samples.get(stage, ()))

    def summary(self) -> dict[str, dict[str, float]]:
        """``{stage: {"n", "p50_ms", "p95_ms", "mean_ms", "max_ms"}}``."""
        with self._lock:
            snap = {k: list(v) for k, v in self._samples.items()}
        out: dict[str, dict[str, float]] = {}
        for k, xs in snap.items():
            if not xs:
                continue
            out[k] = {"n": float(len(xs)), "p50_ms": percentile(xs, 50), "p95_ms": percentile(xs, 95),
                      "mean_ms": sum(xs) / len(xs), "max_ms": max(xs)}
        return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


class Telemetry:
    """Opt-in counters + latency windows for the runtime loop.

    ``idle_gap_s`` caps how much time between two processed frames counts as
    "active" (a long gap means the user was away or SOMA was paused).
    """

    def __init__(self, enabled: bool = False, *, window: int = 512, idle_gap_s: float = 120.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = bool(enabled)
        self.idle_gap_s = float(idle_gap_s)
        self.clock = clock
        self.latency = LatencyStats(window)
        self._lock = threading.Lock()
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.steps = 0
        self.processed = 0
        self.skipped: dict[str, int] = {}
        self.silent = 0
        self.suggestions = 0
        self.feedback: dict[str, int] = {}
        self.active_s = 0.0
        self._last_active_t: float | None = None

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        """Stop recording and drop everything recorded so far."""
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._reset_counters()
        self.latency.reset()

    # -- recording
    def record_step(self, timings: dict[str, float], *, silent: bool,
                    skipped_reason: str | None = None) -> None:
        if not self.enabled:
            return
        now = self.clock()
        with self._lock:
            self.steps += 1
            if skipped_reason:
                key = skipped_reason.split(":", 1)[0]   # never keep detail (e.g. titles)
                self.skipped[key] = self.skipped.get(key, 0) + 1
                return
            self.processed += 1
            if self._last_active_t is not None:
                self.active_s += min(max(0.0, now - self._last_active_t), self.idle_gap_s)
            self._last_active_t = now
            if silent:
                self.silent += 1
            else:
                self.suggestions += 1
        self.latency.add_all({k: v for k, v in timings.items() if k in STAGES})

    def record_feedback(self, outcome: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.feedback[outcome] = self.feedback.get(outcome, 0) + 1

    def mark_inactive(self) -> None:
        """Break the active-time chain (pause): the next frame starts a new span."""
        with self._lock:
            self._last_active_t = None

    # -- reporting
    @property
    def silent_rate(self) -> float:
        with self._lock:
            return self.silent / self.processed if self.processed else 0.0

    @property
    def suggestions_per_active_hour(self) -> float:
        with self._lock:
            hours = self.active_s / 3600.0
            return self.suggestions / hours if hours > 0 else 0.0

    def export(self) -> dict[str, Any]:
        """Plain-number snapshot (no content)."""
        with self._lock:
            counters = {"steps": self.steps, "processed": self.processed,
                        "skipped": dict(self.skipped), "silent": self.silent,
                        "suggestions": self.suggestions, "feedback": dict(self.feedback),
                        "active_s": round(self.active_s, 3)}
        return {"enabled": self.enabled, **counters, "silent_rate": self.silent_rate,
                "suggestions_per_active_hour": self.suggestions_per_active_hour,
                "latency_ms": self.latency.summary()}
