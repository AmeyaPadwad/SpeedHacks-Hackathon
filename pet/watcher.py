"""Context/state engine: fuses IDE diagnostics and webcam presence to decide
when (rarely) to offer help, and drives Pette's presence reactions.

Interruption policy (VISUAL_CODING_COMPANION.md): offer only on persistent, strong evidence;
dedupe by issue fingerprint; per-issue cooldown; daily cap; never while the user is away,
has their eyes closed, or is mid-interaction. An offer is just a question; nothing is uploaded
or sent to an agent until the user approves.
"""
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal

from . import context
from .vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH, CameraWorker

POLL_S = 5


@dataclass
class Signals:
    t: float
    diag_fingerprint: str | None       # active file + first error, from the VS Code bridge
    diag_summary: str
    user_present: bool = True
    eyes_closed: bool = False
    busy: bool = False


class InterventionPolicy:
    DIAG_PERSIST_S = 20        # an IDE error must survive this long (not just mid-typing)
    COOLDOWN_S = 10 * 60       # per issue, after any offer
    DAILY_CAP = 8

    def __init__(self):
        self._first_seen: dict[str, float] = {}
        self._offered: dict[str, float] = {}
        self._day, self._count = None, 0

    def update(self, s: Signals) -> tuple[str, str] | None:
        """Return (fingerprint, reason) when an offer should be made now, else None."""
        candidates = []
        if s.diag_fingerprint:
            candidates.append(("diag:" + s.diag_fingerprint, self.DIAG_PERSIST_S, s.diag_summary))
        live = {fp for fp, _, _ in candidates}
        self._first_seen = {fp: t for fp, t in self._first_seen.items() if fp in live}  # resolved issues reset

        day = int(s.t // 86400)
        if day != self._day:
            self._day, self._count = day, 0

        for fp, persist, reason in candidates:
            first = self._first_seen.setdefault(fp, s.t)
            if s.t - first < persist:
                continue
            if s.t - self._offered.get(fp, -1e18) < self.COOLDOWN_S or self._count >= self.DAILY_CAP:
                continue
            if not s.user_present or s.eyes_closed or s.busy:
                return None   # defer: offer when the user is back and free
            self._offered[fp] = s.t
            self._count += 1
            return fp, reason
        return None


def diag_signal(af: context.ActiveFile | None) -> tuple[str | None, str]:
    if not af or not af.errors:
        return None, ""
    e = af.errors[0]
    name = af.path.replace("\\", "/").rsplit("/", 1)[-1]
    more = f" (+{len(af.errors) - 1} more)" if len(af.errors) > 1 else ""
    return f"{af.path}|{e.message[:120]}", f"{name} line {e.line}: {e.message[:140]}{more}"


class Watcher(QObject):
    """Owns the optional camera worker and the poll timer. All UI calls happen on the main thread."""
    offer = Signal(str)            # reason text; the pipeline turns it into an approval bubble
    camera_status = Signal(str)

    def __init__(self, overlay, is_busy, *, read_file=context.read_active_file,
                 memory=None, world=None, clock=time.monotonic):
        super().__init__()
        self.overlay = overlay
        self._is_busy = is_busy
        self._read_file = read_file
        self.memory, self.world = memory, world   # db.Store / db.WorldModel, optional
        self._clock = clock
        self._away_since: float | None = None
        self._closed_since: float | None = None
        self.policy = InterventionPolicy()
        self.proactive = True
        self.present, self.eyes_closed = True, False
        self.camera: CameraWorker | None = None
        self._timer = QTimer(self, interval=POLL_S * 1000, timeout=self.poll)

    # --- lifecycle ----------------------------------------------------------
    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.set_camera(False)

    def set_camera(self, on: bool):
        if on and self.camera is None:
            self.camera = CameraWorker(blink_on=self.world.blink_threshold() if self.world else None)
            self.camera.event.connect(self.on_vision_event)
            self.camera.status.connect(self.camera_status)
            self.camera.baseline.connect(self.on_baseline)
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
            self.overlay.nudge()                 # a draft that waited while you were away comes forward
            QTimer.singleShot(1500, self.poll)   # offer anything deferred while away, once settled
        elif ev == EYES_CLOSED:
            self.eyes_closed = True
            self._closed_since = now
        elif ev == EYES_OPEN:
            self.eyes_closed = False
            if self._closed_since is not None:
                self._log("eyes_closed_ended", duration=round(now - self._closed_since, 2))
                self._closed_since = None
        elif ev == PINCH:
            self.overlay.pinch()                 # push-to-talk / re-dictate / dismiss; never approves
            return
        self.overlay.set_presence(away=not self.present, eyes_closed=self.eyes_closed)

    def on_baseline(self, summary: dict):
        if self.memory:
            self.memory.log_blink_baseline(summary["mean"], summary["p95"], summary["samples"])

    def _log(self, kind: str, duration: float | None = None):
        if self.memory:
            self.memory.log_event(kind, duration)

    def poll(self):
        if self.overlay.paused or not self.proactive:
            return
        if self.world:  # learned back-off: fewer offers if you usually decline them
            self.policy.COOLDOWN_S = self.world.offer_cooldown_s(InterventionPolicy.COOLDOWN_S)
        fp, summary = diag_signal(self._read_file())
        # Outside VS Code / terminals counts as busy, so the offer is deferred rather than lost.
        busy = self._is_busy() or self.overlay.bubble.isVisible() or not self.overlay.context_allowed
        hit = self.policy.update(Signals(time.time(), fp, summary, self.present, self.eyes_closed, busy))
        if hit:
            self.offer.emit(hit[1])

