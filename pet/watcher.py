"""Webcam presence: owns the optional camera worker and turns its events into Pette's reactions.

Face away -> Pette dozes; back -> a pending send is brought to attention; eyes closed -> she blinks;
pinch -> the same as clicking Pette (or cancels a pending send). Preview frames go straight to the orb.
Nothing is uploaded or stored apart from event names and timings in the local db.
"""
import time

from PySide6.QtCore import QObject, Signal

from .vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH, CameraWorker


class Watcher(QObject):
    camera_status = Signal(str)

    def __init__(self, overlay, *, memory=None, world=None, clock=time.monotonic, worker=CameraWorker):
        super().__init__()
        self.overlay = overlay
        self.memory, self.world = memory, world   # db.Store / db.WorldModel, optional
        self._clock = clock
        self._worker = worker
        self._away_since: float | None = None
        self._closed_since: float | None = None
        self.present, self.eyes_closed = True, False
        self.camera: CameraWorker | None = None

    # --- lifecycle ----------------------------------------------------------
    def stop(self):
        self.set_camera(False)

    def set_camera(self, on: bool):
        if on and self.camera is None:
            self.camera = self._worker(blink_on=self.world.blink_threshold() if self.world else None)
            self.camera.event.connect(self.on_vision_event)
            self.camera.status.connect(self.camera_status)
            self.camera.baseline.connect(self.on_baseline)
            self.camera.frame.connect(self.on_frame)
            self.camera.finished.connect(self._camera_finished)
            self.camera.start()
        elif not on and self.camera is not None:
            cam, self.camera = self.camera, None
            cam.stop()
            self._camera_finished()
        self.overlay.set_camera_indicator(self.camera is not None)

    def _camera_finished(self):
        if self.camera is not None and not self.camera.isRunning():
            self.camera = None
            self.overlay.set_camera_indicator(False)
        self.present, self.eyes_closed = True, False
        self.overlay.set_presence(away=False, eyes_closed=False)

    # --- inputs -------------------------------------------------------------
    def on_frame(self, frame: dict):
        if self.camera is not None and not self.overlay.paused:
            self.overlay.orb.set_frame(frame)

    def on_vision_event(self, ev: str):
        if self.overlay.paused:   # paused means no reactions and no logging
            return
        now = self._clock()
        self._log(ev)
        if ev == FACE_AWAY:
            self.present = False
            self._away_since = now
        elif ev == FACE_BACK:
            self.present = True
            if self._away_since is not None:
                self._log("away_ended", duration=round(now - self._away_since, 1))
                self._away_since = None
            self.overlay.nudge()                 # a send that counted down while you were away comes forward
        elif ev == EYES_CLOSED:
            self.eyes_closed = True
            self._closed_since = now
        elif ev == EYES_OPEN:
            self.eyes_closed = False
            if self._closed_since is not None:
                self._log("eyes_closed_ended", duration=round(now - self._closed_since, 2))
                self._closed_since = None
        elif ev == PINCH:
            self.overlay.pinch()                 # talk / cancel a pending send / dismiss; never sends
            return
        self.overlay.set_presence(away=not self.present, eyes_closed=self.eyes_closed)

    def on_baseline(self, summary: dict):
        if self.memory:
            self.memory.log_blink_baseline(summary["mean"], summary["p95"], summary["samples"])

    def _log(self, kind: str, duration: float | None = None):
        if self.memory:
            self.memory.log_event(kind, duration)
