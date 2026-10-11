"""Context/state engine: fuses IDE diagnostics, local screen CV and webcam presence to decide
when (rarely) to offer help, and drives Pette's presence reactions.

Interruption policy (VISUAL_CODING_COMPANION.md): offer only on persistent, strong evidence;
dedupe by issue fingerprint; per-issue cooldown; daily cap; never while the user is away,
has their eyes closed, or is mid-interaction. An offer is just a question; nothing is captured
for the cloud or sent to an agent until the user approves.
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
    screen_fingerprint: str | None     # local CV error regions (only when screen watching is on)
    user_present: bool = True
    eyes_closed: bool = False
    busy: bool = False


class InterventionPolicy:
    DIAG_PERSIST_S = 20        # an IDE error must survive this long (not just mid-typing)
    SCREEN_PERSIST_S = 45      # screen-only evidence is weaker, so wait longer
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
        if s.screen_fingerprint:
            candidates.append(("screen:" + s.screen_fingerprint, self.SCREEN_PERSIST_S,
                               "Error-coloured text has stayed on screen for a while."))
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

    def __init__(self, overlay, is_busy, *, read_file=context.read_active_file, grab_findings=None):
        super().__init__()
        self.overlay = overlay
        self._is_busy = is_busy
        self._read_file = read_file
        self._grab_findings = grab_findings or _grab_findings
        self.policy = InterventionPolicy()
        self.proactive = True
        self.watch_screen = False
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
            self.camera = CameraWorker()
            self.camera.event.connect(self.on_vision_event)
            self.camera.status.connect(self.camera_status)
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
        if ev == FACE_AWAY:
            self.present = False
        elif ev == FACE_BACK:
            self.present = True
            QTimer.singleShot(1500, self.poll)   # offer anything deferred while away, once settled
        elif ev == EYES_CLOSED:
            self.eyes_closed = True
        elif ev == EYES_OPEN:
            self.eyes_closed = False
        elif ev == PINCH:
            self.overlay.pinch()                 # open/dismiss only; never approves
            return
        self.overlay.set_presence(away=not self.present, eyes_closed=self.eyes_closed)

    def poll(self):
        if self.overlay.paused or not self.proactive:
            return
        fp, summary = diag_signal(self._read_file())
        screen_fp = None
        if self.watch_screen and self.present and not self.overlay.isHidden():
            findings = self._grab_findings(self.overlay)
            screen_fp = findings.fingerprint() if findings and findings.has_errors else None
        busy = self._is_busy() or self.overlay.bubble.isVisible()
        hit = self.policy.update(Signals(time.time(), fp, summary, screen_fp, self.present, self.eyes_closed, busy))
        if hit:
            self.offer.emit(hit[1])


def _grab_findings(overlay):
    """Local-only screen CV for watching: the frame is analysed and dropped, never uploaded or stored."""
    from PySide6.QtGui import QGuiApplication
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return None
    pm = screen.grabWindow(0)
    if pm.isNull():
        return None
    findings = context.find_errors(pm, screen)
    if findings is not None:
        g, r = overlay.frameGeometry(), pm.devicePixelRatio() or 1
        sg = screen.geometry()
        ox, oy = int((g.x() - sg.x()) * r), int((g.y() - sg.y()) * r)
        ow, oh = int(g.width() * r), int(g.height() * r)
        # Drop regions that overlap Pette itself (pink ears, bubble text).
        findings.regions = [x for x in findings.regions
                            if not (x.x < ox + ow and x.x + x.w > ox and x.y < oy + oh and x.y + x.h > oy)]
    return findings
