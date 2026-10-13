"""Webcam CV (local only): face presence, eye closure and hand gestures -> debounced events, plus a small
preview (mirrored 168px crop, hand points, gesture name) for the orb.

Gestures (all pure logic in `VisionPolicy`, so they're unit-tested without a camera):
  * pinch (thumb to index) held still ~0.5 s       -> PINCH ("Do it" on a SOMA card)
  * pinch, then move the hand                      -> DRAG_START, scroll amounts, DRAG_END (browser scroll)
  * index finger pointing to your left / right     -> POINT_LEFT / POINT_RIGHT (previous / next browser tab)
Left and right are from your point of view, the same as in the mirrored preview on the orb.

Nothing leaves this PC: the preview is only painted on screen, never stored, logged or uploaded.
Eye closure is a heuristic, not gaze, attention or emotion measurement.
"""
import math
import time
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from .paths import asset_path

FACE_MODEL = asset_path("face_landmarker.task", folder="models")
HAND_MODEL = asset_path("hand_landmarker.task", folder="models")
CAMERA_INDEX = 0
MAX_FPS = 15

# Event names emitted by VisionPolicy / CameraWorker.
FACE_AWAY, FACE_BACK = "face_away", "face_back"
EYES_CLOSED, EYES_OPEN = "eyes_closed", "eyes_open"
PINCH = "pinch"
POINT_LEFT, POINT_RIGHT = "point_left", "point_right"
DRAG_START, DRAG_END = "drag_start", "drag_end"


@dataclass
class Observation:
    """One processed frame. Hand fields are None when no usable hand was found."""
    t: float                      # monotonic seconds
    face: bool
    blink_left: float = 0.0
    blink_right: float = 0.0
    pinch_ratio: float | None = None
    pinch_y: float | None = None  # vertical position of the pinch (0 top .. 1 bottom of the camera image)
    point: str | None = None      # "left" / "right" (your point of view) while the index finger points sideways


class VisionPolicy:
    """Deterministic debounce rules from BUILD.md, separated from the camera so they can be tested."""
    AWAY_AFTER_S, BACK_AFTER_S = 3.0, 1.0
    BLINK_ON, BLINK_OFF, BLINK_HOLD_S = 0.6, 0.4, 0.3
    PINCH_ON, PINCH_REARM, PINCH_HOLD_S, PINCH_COOLDOWN_S = 0.25, 0.4, 0.5, 2.0
    DRAG_START_DY = 0.04          # pinch moves this far (fraction of image height) -> it's a drag, not a pinch
    DRAG_SMOOTH = 0.5             # EMA on the pinch position; less jitter in the scroll
    POINT_HOLD_S, POINT_COOLDOWN_S = 0.25, 0.6

    def __init__(self, blink_on: float | None = None):
        if blink_on is not None:  # calibrated per user by the world model; reopen stays 0.2 below
            self.BLINK_ON = blink_on
            self.BLINK_OFF = max(0.2, blink_on - 0.2)
        self.present = True
        self.eyes_closed = False
        self._face_change_since: float | None = None
        self._blink_since: float | None = None
        self._pinch_since: float | None = None
        self._pinch_armed = True
        self._last_pinch = -math.inf
        self.dragging = False
        self._pinch_start_y: float | None = None   # where this pinch began (smoothed)
        self._pinch_y: float | None = None         # smoothed current pinch position
        self._scroll = 0.0                         # pending scroll (image heights; + = hand moved down)
        self._point: str | None = None
        self._point_since: float | None = None
        self._point_armed = True
        self._last_point = -math.inf

    def take_scroll(self) -> float:
        """Scroll accumulated since the last call, in image heights (+ = the hand moved down)."""
        dy, self._scroll = self._scroll, 0.0
        return dy

    def update(self, o: Observation) -> list[str]:
        events: list[str] = []

        # Face presence with hysteresis in time.
        if o.face != self.present:
            if self._face_change_since is None:
                self._face_change_since = o.t
            need = self.BACK_AFTER_S if o.face else self.AWAY_AFTER_S
            if o.t - self._face_change_since >= need:
                self.present = o.face
                self._face_change_since = None
                events.append(FACE_BACK if o.face else FACE_AWAY)
                if not o.face and self.eyes_closed:
                    self.eyes_closed = False
                    self._blink_since = None
        else:
            self._face_change_since = None

        # Eye closure: both blendshapes high for a hold time; reopen below a lower threshold.
        if o.face:
            if not self.eyes_closed:
                if min(o.blink_left, o.blink_right) > self.BLINK_ON:
                    self._blink_since = self._blink_since if self._blink_since is not None else o.t
                    if o.t - self._blink_since >= self.BLINK_HOLD_S:
                        self.eyes_closed = True
                        events.append(EYES_CLOSED)
                else:
                    self._blink_since = None
            elif max(o.blink_left, o.blink_right) < self.BLINK_OFF:
                self.eyes_closed = False
                self._blink_since = None
                events.append(EYES_OPEN)

        events += self._pinch_and_drag(o)
        events += self._pointing(o)
        return events

    def _end_drag(self) -> list[str]:
        was = self.dragging
        self.dragging = False
        self._pinch_start_y = self._pinch_y = None
        return [DRAG_END] if was else []

    def _pinch_and_drag(self, o: Observation) -> list[str]:
        """Pinch held still -> PINCH (needs rearm + cooldown). Pinch that moves -> a drag that scrolls."""
        r = o.pinch_ratio
        if r is None or r > self.PINCH_REARM:          # hand gone or fingers apart: pinch / drag over
            if r is not None:
                self._pinch_armed = True
            self._pinch_since = None
            return self._end_drag()
        if r >= self.PINCH_ON:                         # between thresholds: keep whatever is going on
            return []
        events: list[str] = []
        if o.pinch_y is not None:
            prev = self._pinch_y
            self._pinch_y = o.pinch_y if prev is None else prev + self.DRAG_SMOOTH * (o.pinch_y - prev)
            if self._pinch_start_y is None:
                self._pinch_start_y = self._pinch_y
            if self.dragging:
                self._scroll += self._pinch_y - prev
            elif abs(self._pinch_y - self._pinch_start_y) >= self.DRAG_START_DY:
                self.dragging = True
                self._pinch_since = None               # a drag is never also a pinch
                self._scroll += self._pinch_y - self._pinch_start_y
                events.append(DRAG_START)
        if self.dragging or not self._pinch_armed:
            return events
        self._pinch_since = self._pinch_since if self._pinch_since is not None else o.t
        if o.t - self._pinch_since >= self.PINCH_HOLD_S and o.t - self._last_pinch >= self.PINCH_COOLDOWN_S:
            events.append(PINCH)
            self._last_pinch = o.t
            self._pinch_armed = False
            self._pinch_since = None
        return events

    def _pointing(self, o: Observation) -> list[str]:
        """Point held briefly -> one POINT_LEFT / POINT_RIGHT; relax (or switch side) to fire again."""
        d = None if self.dragging else o.point
        if d is None:
            self._point, self._point_since, self._point_armed = None, None, True
            return []
        if d != self._point:
            self._point, self._point_since, self._point_armed = d, o.t, True
        if (self._point_armed and o.t - self._point_since >= self.POINT_HOLD_S
                and o.t - self._last_point >= self.POINT_COOLDOWN_S):
            self._point_armed = False
            self._last_point = o.t
            return [POINT_LEFT if d == "left" else POINT_RIGHT]
        return []


def pinch_ratio(landmarks) -> float | None:
    """Thumb tip (4) to index tip (8) distance, normalised by wrist (0) to middle knuckle (9)."""
    if landmarks is None or len(landmarks) < 10:
        return None
    d = lambda a, b: math.hypot(a.x - b.x, a.y - b.y)
    scale = d(landmarks[0], landmarks[9])
    if scale < 1e-3:
        return None
    return d(landmarks[4], landmarks[8]) / scale


def pinch_y(landmarks) -> float | None:
    """Vertical position of the pinch: the midpoint of the thumb and index tips (0 top .. 1 bottom)."""
    if landmarks is None or len(landmarks) < 9:
        return None
    return (landmarks[4].y + landmarks[8].y) / 2


def pointing(landmarks) -> str | None:
    """"left" / "right" (your point of view) when only the index finger is out and it points sideways.

    The camera sees you face to face, so pointing to your left moves the fingertip to the image's right
    (+x). Other fingers must be curled (tips close to the wrist) so an open hand never counts.
    """
    if landmarks is None or len(landmarks) < 21:
        return None
    d = lambda a, b: math.hypot(a.x - b.x, a.y - b.y)
    wrist, scale = landmarks[0], d(landmarks[0], landmarks[9])
    if scale < 1e-3:
        return None
    vx, vy = landmarks[8].x - landmarks[5].x, landmarks[8].y - landmarks[5].y
    if math.hypot(vx, vy) < 0.6 * scale or abs(vx) < 1.6 * abs(vy):
        return None                                   # index not stretched out sideways
    for tip, mcp in ((12, 9), (16, 13), (20, 17)):    # middle, ring, pinky curled toward the palm
        if d(landmarks[tip], wrist) > d(landmarks[mcp], wrist) * 1.25:
            return None
    return "left" if vx > 0 else "right"


def gesture_name(landmarks) -> str:
    """Orb label: "pinch", "point left", "point right", or "" for anything else."""
    if landmarks is None or len(landmarks) < 21:
        return ""
    r = pinch_ratio(landmarks)
    if r is not None and r < VisionPolicy.PINCH_ON:
        return "pinch"
    p = pointing(landmarks)
    return f"point {p}" if p else ""


def preview(rgb, hand_landmarks, size: int = 168) -> dict:
    """Small mirrored square crop + hand points (0..1, mirrored to match) for the orb. Local only."""
    import cv2
    from PySide6.QtGui import QImage
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = cv2.resize(rgb[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)
    crop = cv2.flip(crop, 1).copy()
    img = QImage(crop.data, size, size, 3 * size, QImage.Format_RGB888).copy()   # own the pixels
    pts = []
    if hand_landmarks:
        for lm in hand_landmarks:
            px, py = lm.x * w, lm.y * h
            pts.append((1 - (px - x0) / side, (py - y0) / side))
    return {"image": img, "hand": pts, "gesture": gesture_name(hand_landmarks)}


def blink_scores(blendshapes) -> tuple[float, float]:
    scores = {c.category_name: c.score for c in blendshapes or []}
    return scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0)


class BaselineMeter:
    """Summarises open-eye blink scores over a window (numbers only) for threshold calibration."""
    WINDOW_S = 30

    def __init__(self):
        self._samples: list[float] = []
        self._start: float | None = None

    def add(self, o: Observation, eyes_closed: bool) -> dict | None:
        if self._start is None:
            self._start = o.t
        if o.face and not eyes_closed:
            self._samples.append(max(o.blink_left, o.blink_right))
        if o.t - self._start < self.WINDOW_S:
            return None
        s, self._samples, self._start = sorted(self._samples), [], o.t
        if len(s) < 20:
            return None
        return {"mean": round(sum(s) / len(s), 3), "p95": round(s[int(len(s) * 0.95) - 1], 3), "samples": len(s)}


class CameraWorker(QThread):
    """Owns the camera and MediaPipe tasks on its own thread. Emits event names and blink summaries only."""
    event = Signal(str)
    status = Signal(str)     # human-readable camera status for the UI
    baseline = Signal(dict)  # {"mean", "p95", "samples"} every ~30 s, for the world model
    frame = Signal(object)   # preview() dict for the orb; display only
    scroll = Signal(float)   # pinch-drag: image heights moved since the last frame (+ = hand down)

    def __init__(self, camera_index: int = CAMERA_INDEX, blink_on: float | None = None):
        super().__init__()
        self.camera_index = camera_index
        self.blink_on = blink_on
        self._running = False

    def stop(self):
        self._running = False
        self.wait(2000)  # BUILD.md: release the camera within 2 seconds

    def run(self):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError:
            self.status.emit("Camera unavailable: install opencv-contrib-python and mediapipe.")
            return
        if not (FACE_MODEL.exists() and HAND_MODEL.exists()):
            self.status.emit("Camera unavailable: run tools/get_models.py first.")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status.emit("Camera unavailable: denied or in use. Check Windows camera privacy settings.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        mode = vision.RunningMode.VIDEO
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=mode,
            num_faces=1, output_face_blendshapes=True))
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=mode, num_hands=1))
        policy = VisionPolicy(self.blink_on)
        meter = BaselineMeter()
        self._running = True
        self.status.emit("Camera on (processed locally)")
        last_ts, period = -1, 1.0 / MAX_FPS
        try:
            while self._running:
                start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    self.status.emit("Camera disconnected.")
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = max(int(start * 1000), last_ts + 1)  # strictly increasing for VIDEO mode
                last_ts = ts
                fr = face.detect_for_video(image, ts)
                hr = hand.detect_for_video(image, ts)
                has_face = bool(fr.face_landmarks)
                bl, br = blink_scores(fr.face_blendshapes[0]) if has_face and fr.face_blendshapes else (0.0, 0.0)
                lm = hr.hand_landmarks[0] if hr.hand_landmarks else None
                obs = Observation(start, has_face, bl, br, pinch_ratio(lm), pinch_y(lm), pointing(lm))
                for ev in policy.update(obs):
                    self.event.emit(ev)
                if (dy := policy.take_scroll()) != 0.0:
                    self.scroll.emit(dy)
                if (summary := meter.add(obs, policy.eyes_closed)) is not None:
                    self.baseline.emit(summary)
                self.frame.emit(preview(rgb, hr.hand_landmarks[0] if hr.hand_landmarks else None))
                del frame, rgb, image  # frames are never kept
                time.sleep(max(0.0, period - (time.monotonic() - start)))
        finally:
            cap.release()
            face.close()
            hand.close()
            self.status.emit("Camera off")
