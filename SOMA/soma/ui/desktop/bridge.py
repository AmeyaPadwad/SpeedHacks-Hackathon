"""Thread bridge between the runtime's worker thread and the Qt main thread.

``InferenceScheduler`` runs ``runtime.step()`` on a plain Python thread and
calls ``on_result`` there. :class:`RuntimeBridge` turns that callback into a
queued Qt signal so widgets are only ever touched on the main thread, and
drops any result that arrives after a pause using a generation counter.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Protocol, runtime_checkable

from PySide6.QtCore import QObject, Qt, Signal, Slot

from soma.types import FeedbackOutcome


@runtime_checkable
class RuntimeLike(Protocol):
    """The slice of ``soma.runtime.pipeline.SomaRuntime`` the UI depends on."""

    def step(self, force: bool = False) -> Any: ...
    def feedback(self, suggestion_id: str, outcome: FeedbackOutcome,
                 edited_text: str | None = None) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def current_suggestion_valid(self, current_version: int) -> bool: ...

    @property
    def paused(self) -> bool: ...


class SchedulerLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def trigger_now(self) -> None: ...


SchedulerFactory = Callable[[Any, float, Callable[[Any], None]], SchedulerLike]


def default_scheduler_factory(runtime: Any, interval_s: float,
                              on_result: Callable[[Any], None]) -> SchedulerLike:
    from soma.runtime.scheduler import InferenceScheduler
    return InferenceScheduler(runtime, interval_s, on_result)


class RuntimeBridge(QObject):
    """Owns the scheduler; re-emits worker results on the main thread.

    Signals:
        result(object): a ``StepResult`` that is current (not from before a pause).
        state_changed(str): "Capturing", "Paused" or "Off".
        error(str): a short message when a step raised.
    """

    result = Signal(object)
    state_changed = Signal(str)
    error = Signal(str)
    _raw = Signal(object, int)
    _raw_error = Signal(str, int)

    def __init__(self, runtime: RuntimeLike | None = None, interval_s: float = 1.0,
                 scheduler_factory: SchedulerFactory | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self._interval = float(interval_s)
        self._factory = scheduler_factory or default_scheduler_factory
        self._scheduler: SchedulerLike | None = None
        self._gen = 0
        self._gen_lock = threading.Lock()
        self._paused = False
        self.dropped = 0
        self._raw.connect(self._on_raw, Qt.ConnectionType.QueuedConnection)
        self._raw_error.connect(self._on_raw_error, Qt.ConnectionType.QueuedConnection)

    # -- properties
    @property
    def runtime(self) -> RuntimeLike | None:
        return self._runtime

    @property
    def generation(self) -> int:
        with self._gen_lock:
            return self._gen

    @property
    def running(self) -> bool:
        return self._scheduler is not None

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def state(self) -> str:
        if self._scheduler is None or self._runtime is None:
            return "Off"
        return "Paused" if self._paused else "Capturing"

    # -- lifecycle
    def set_runtime(self, runtime: RuntimeLike) -> None:
        self.stop()
        self._runtime = runtime

    def start(self) -> None:
        """Start the worker. Callers must have checked SCREEN_CAPTURE consent."""
        if self._runtime is None or self._scheduler is not None:
            return
        self._paused = False
        self._scheduler = self._factory(self._runtime, self._interval, self.on_worker_result)
        self._scheduler.start()
        self.state_changed.emit(self.state)

    def stop(self) -> None:
        self._bump()
        sched, self._scheduler = self._scheduler, None
        if sched is not None:
            sched.stop()
        self.state_changed.emit(self.state)

    def pause(self) -> None:
        """Instant pause: later results from the worker are dropped."""
        self._bump()
        self._paused = True
        if self._runtime is not None:
            self._runtime.pause()
        if self._scheduler is not None:
            self._scheduler.pause()
        self.state_changed.emit(self.state)

    def resume(self) -> None:
        self._paused = False
        if self._runtime is not None:
            self._runtime.resume()
        if self._scheduler is not None:
            self._scheduler.resume()
        self.state_changed.emit(self.state)

    def toggle(self) -> None:
        self.resume() if self._paused else self.pause()

    def trigger_now(self) -> None:
        if self._scheduler is not None and not self._paused:
            self._scheduler.trigger_now()

    def step_sync(self, force: bool = True) -> Any:
        """Run one step on the calling (main) thread; used by ``--smoke``."""
        if self._runtime is None or self._paused:
            return None
        res = self._runtime.step(force=force)
        self.result.emit(res)
        return res

    # -- worker side (any thread)
    def on_worker_result(self, res: Any) -> None:
        """Called by the scheduler on its worker thread. Never touches widgets."""
        self._raw.emit(res, self.generation)

    def on_worker_error(self, message: str) -> None:
        self._raw_error.emit(str(message), self.generation)

    def _bump(self) -> None:
        with self._gen_lock:
            self._gen += 1

    # -- main thread
    @Slot(object, int)
    def _on_raw(self, res: Any, gen: int) -> None:
        if gen != self.generation or self._paused:
            self.dropped += 1
            return
        self.result.emit(res)

    @Slot(str, int)
    def _on_raw_error(self, message: str, gen: int) -> None:
        if gen == self.generation and not self._paused:
            self.error.emit(message)
