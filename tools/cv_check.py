"""Live webcam check for Pette's CV signals: preview, raw scores vs thresholds, events, and Pette's reaction.

Local only: frames are shown in this window and never saved or sent anywhere.
Run: py tools/cv_check.py   (close the window or press Esc to stop and release the camera)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from pet.ui import Orb
from pet.vision import (EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, FACE_MODEL, HAND_MODEL, PINCH, Observation,
                        VisionPolicy, blink_scores, pinch_ratio)


class Loop(QThread):
    frame = Signal(QImage, object, list)   # preview, Observation, events

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=vision.RunningMode.VIDEO,
            num_faces=1, output_face_blendshapes=True))
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=vision.RunningMode.VIDEO,
            num_hands=1))
        policy, last = VisionPolicy(), -1
        self.running = True
        try:
            while self.running:
                start = time.monotonic()
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ts = max(int(start * 1000), last + 1)
                last = ts
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                fr, hr = face.detect_for_video(img, ts), hand.detect_for_video(img, ts)
                has_face = bool(fr.face_landmarks)
                bl, br = blink_scores(fr.face_blendshapes[0]) if has_face and fr.face_blendshapes else (0.0, 0.0)
                hl = hr.hand_landmarks[0] if hr.hand_landmarks else None
                o = Observation(start, has_face, bl, br, pinch_ratio(hl))
                events = policy.update(o)
                h, w = rgb.shape[:2]
                if hl:  # draw thumb and index tips so the pinch is visible
                    for i in (4, 8):
                        cv2.circle(rgb, (int(hl[i].x * w), int(hl[i].y * h)), 7, (124, 242, 196), -1)
                rgb = cv2.flip(rgb, 1)   # mirror, like a selfie view
                q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                self.frame.emit(q, o, events)
                time.sleep(max(0.0, 0.1 - (time.monotonic() - start)))
        finally:
            cap.release()
            face.close()
            hand.close()


def bar(value: float, on: float, off: float | None = None, invert=False) -> str:
    filled = int(min(1.0, value) * 20)
    hit = value < on if invert else value > on
    return f"{'#' * filled:<20} {value:.2f} {'<<' if hit else '  '}"


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pette CV check (local only, nothing saved)")
        self.preview = QLabel(alignment=Qt.AlignCenter)
        self.preview.setFixedSize(640, 480)
        self.stats = QLabel()
        self.stats.setFont(QFont("Consolas", 10))
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(QFont("Consolas", 10))
        self.log.setMaximumHeight(150)
        self.pet = Orb()            # Pette's body: mirrors you (dozes / shuts eyes)
        self.pet.set_camera(True)
        self.status = QLabel("Pette: idle")
        self.status.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        right = QVBoxLayout()
        right.addWidget(self.status)
        right.addWidget(self.pet, 0, Qt.AlignHCenter)
        right.addWidget(self.stats)
        right.addStretch()
        top = QHBoxLayout()
        top.addWidget(self.preview)
        top.addLayout(right)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(QLabel("Events (what the app reacts to):"))
        lay.addWidget(self.log)
        self.away = self.closed = False
        self.t0 = time.monotonic()
        self.loop = Loop()
        self.loop.frame.connect(self.on_frame)
        self.loop.start()
        self.log.appendPlainText("Try: look away/leave for 3s · close both eyes ~0.5s · pinch thumb+index ~0.5s")

    def on_frame(self, q: QImage, o: Observation, events: list):
        self.preview.setPixmap(QPixmap.fromImage(q))
        p = VisionPolicy
        pr = "no hand" if o.pinch_ratio is None else bar(o.pinch_ratio, p.PINCH_ON, invert=True)
        self.stats.setText(
            f"face        {'YES' if o.face else 'no'}\n"
            f"blink L     {bar(o.blink_left, p.BLINK_ON)}\n"
            f"blink R     {bar(o.blink_right, p.BLINK_ON)}\n"
            f"pinch ratio {pr}\n\n"
            f"closed > {p.BLINK_ON} (both, {p.BLINK_HOLD_S}s) · pinch < {p.PINCH_ON} ({p.PINCH_HOLD_S}s)\n"
            f"away after {p.AWAY_AFTER_S:.0f}s without a face")
        for ev in events:
            self.log.appendPlainText(f"{o.t - self.t0:7.1f}s  {ev}")
            if ev in (FACE_AWAY, FACE_BACK):
                self.away = ev == FACE_AWAY
            elif ev in (EYES_CLOSED, EYES_OPEN):
                self.closed = ev == EYES_CLOSED
            elif ev == PINCH:
                self.status.setText("Pette: pinch! (would open Pette / dismiss a message)")
        state = "away" if self.away else "blink" if self.closed else "idle"
        self.pet.set_mood(state)
        if PINCH not in events and not self.status.text().startswith("Pette: pinch"):
            self.status.setText({"away": "Pette: dozing (you're away)", "blink": "Pette: eyes closed (mirroring you)",
                                 "idle": "Pette: idle"}[state])
        elif PINCH not in events and self.status.text().startswith("Pette: pinch") and int(o.t * 2) % 6 == 0:
            self.status.setText("Pette: idle")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, e):
        self.loop.running = False
        self.loop.wait(2000)
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())
