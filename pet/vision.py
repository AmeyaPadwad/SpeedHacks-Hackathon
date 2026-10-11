"""Webcam CV (local only): face presence, eye closure and pinch -> debounced events.

Frames, landmarks and blendshapes never leave this module; only event names are emitted.
Gestures are explicit UI controls and never approve actions. Eye closure is a heuristic,
not gaze, attention or emotion measurement.
"""
import math
import time
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from .sprites import asset_path

FACE_MODEL = asset_path("face_landmarker.task", folder="models")
HAND_MODEL = asset_path("hand_landmarker.task", folder="models")
CAMERA_INDEX = 0
MAX_FPS = 10

# Event names emitted by VisionPolicy / CameraWorker.
FACE_AWAY, FACE_BACK = "face_away", "face_back"
EYES_CLOSED, EYES_OPEN = "eyes_closed", "eyes_open"
PINCH = "pinch"


@dataclass
class Observation:
    """One processed frame. `pinch_ratio` is None when no usable hand was found."""
    t: float                      # monotonic seconds
    face: bool
    blink_left: float = 0.0
    blink_right: float = 0.0
    pinch_ratio: float | None = None


class VisionPolicy:
    """Deterministic debounce rules from BUILD.md, separated from the camera so they can be tested."""
    AWAY_AFTER_S, BACK_AFTER_S = 3.0, 1.0
    BLINK_ON, BLINK_OFF, BLINK_HOLD_S = 0.6, 0.4, 0.3
    PINCH_ON, PINCH_REARM, PINCH_HOLD_S, PINCH_COOLDOWN_S = 0.25, 0.4, 0.5, 2.0

    def __init__(self):
        self.present = True
        self.eyes_closed = False
        self._face_change_since: float | None = None
        self._blink_since: float | None = None
        self._pinch_since: float | None = None
        self._pinch_armed = True
        self._last_pinch = -math.inf

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

        # Pinch: held below threshold, must rearm above a higher one, plus a cooldown.
        r = o.pinch_ratio
        if r is None:
            self._pinch_since = None
        elif r > self.PINCH_REARM:
            self._pinch_armed = True
            self._pinch_since = None
        elif r < self.PINCH_ON and self._pinch_armed:
            self._pinch_since = self._pinch_since if self._pinch_since is not None else o.t
            if o.t - self._pinch_since >= self.PINCH_HOLD_S and o.t - self._last_pinch >= self.PINCH_COOLDOWN_S:
                events.append(PINCH)
                self._last_pinch = o.t
                self._pinch_armed = False
                self._pinch_since = None
        return events


def pinch_ratio(landmarks) -> float | None:
    """Thumb tip (4) to index tip (8) distance, normalised by wrist (0) to middle knuckle (9)."""
    if landmarks is None or len(landmarks) < 10:
        return None
    d = lambda a, b: math.hypot(a.x - b.x, a.y - b.y)
    scale = d(landmarks[0], landmarks[9])
    if scale < 1e-3:
        return None
    return d(landmarks[4], landmarks[8]) / scale


def blink_scores(blendshapes) -> tuple[float, float]:
    scores = {c.category_name: c.score for c in blendshapes or []}
    return scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0)


class CameraWorker(QThread):
    """Owns the camera and MediaPipe tasks on its own thread. Emits event names only."""
    event = Signal(str)
    status = Signal(str)   # human-readable camera status for the UI

    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__()
        self.camera_index = camera_index
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
        policy = VisionPolicy()
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
                pr = pinch_ratio(hr.hand_landmarks[0]) if hr.hand_landmarks else None
                for ev in policy.update(Observation(start, has_face, bl, br, pr)):
                    self.event.emit(ev)
                del frame, rgb, image  # frames are never kept
                time.sleep(max(0.0, period - (time.monotonic() - start)))
        finally:
            cap.release()
            face.close()
            hand.close()
            self.status.emit("Camera off")
