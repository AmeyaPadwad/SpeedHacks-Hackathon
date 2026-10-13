"""Run every Pette check and write test_results/REPORT.md (+ junit.xml and UI screenshots).

    py tools/run_checks.py                 # everything (SOMA's own suite too, if SOMA/.venv exists)
    py tools/run_checks.py --no-camera     # skip the live webcam check
    py tools/run_checks.py --no-soma-tests # skip SOMA's pytest suite (a few minutes on CPU)

Live checks report numbers only: no camera frames, screen images or file contents are saved.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_results"
SHOTS = OUT / "screenshots"
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
results: list[tuple[str, str, str]] = []   # (check, PASS/FAIL/SKIP/INFO, detail)


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status:4}] {name}: {detail}")


def check(name):
    def wrap(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
            except Exception as e:  # a crashing check is a failed check, not a crashed report
                record(name, "FAIL", f"{type(e).__name__}: {e}"[:300])
        return run
    return wrap


# --- checks ------------------------------------------------------------------
@check("Environment")
def env_check():
    import cv2
    import mediapipe
    import PySide6
    record("Environment", "INFO", f"Windows {platform.release()} ({platform.version()}), Python {platform.python_version()}, "
           f"PySide6 {PySide6.__version__}, OpenCV {cv2.__version__}, MediaPipe {mediapipe.__version__}")


@check("Unit + integration tests")
def pytest_check():
    xml = OUT / "junit.xml"
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    suite = ET.parse(xml).getroot()
    suite = suite if suite.tag == "testsuite" else suite[0]
    n, fail, err, skip = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    by_file: dict[str, list[int]] = {}
    for case in suite.iter("testcase"):
        f = case.get("classname", "").split(".")[1] if "." in case.get("classname", "") else case.get("classname")
        ok = not any(c.tag in ("failure", "error") for c in case)
        by_file.setdefault(f, [0, 0])[0 if ok else 1] += 1
    global test_breakdown
    test_breakdown = by_file
    status = "PASS" if p.returncode == 0 else "FAIL"
    record("Unit + integration tests", status,
           f"{n - fail - err - skip}/{n} passed, {fail} failed, {err} errors, {skip} skipped in {time.time() - t:.1f}s")


test_breakdown: dict[str, list[int]] = {}


@check("MediaPipe models")
def models_check():
    from pet.vision import FACE_MODEL, HAND_MODEL
    ok = FACE_MODEL.exists() and HAND_MODEL.exists()
    record("MediaPipe models", "PASS" if ok else "FAIL",
           "face + hand landmarker present" if ok else "missing: run py tools/get_models.py")


@check("Webcam: face / eyes / hands (live)")
def camera_check():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from pet.vision import FACE_MODEL, HAND_MODEL, blink_scores
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        record("Webcam: face / eyes / hands (live)", "FAIL", "camera 0 could not be opened")
        return
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=vision.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=vision.RunningMode.VIDEO, num_hands=1))
    frames = faces = hands = 0
    blinks, last, t0 = [], -1, time.monotonic()
    try:
        while time.monotonic() - t0 < 4:
            ok, bgr = cap.read()
            if not ok:
                break
            ts = max(int(time.monotonic() * 1000), last + 1)
            last = ts
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            fr, hr = face.detect_for_video(img, ts), hand.detect_for_video(img, ts)
            frames += 1
            if fr.face_landmarks:
                faces += 1
                blinks.append(max(blink_scores(fr.face_blendshapes[0])))
            hands += bool(hr.hand_landmarks)
            del bgr, img
    finally:
        cap.release()
        face.close()
        hand.close()
    detail = (f"{frames} frames in 4s, face in {faces}/{frames}, hand in {hands}/{frames}"
              + (f", open-eye blink score avg {sum(blinks) / len(blinks):.2f} (closed threshold 0.6)" if blinks else ""))
    status = "PASS" if frames and faces else ("INFO" if frames else "FAIL")
    record("Webcam: face / eyes / hands (live)", status,
           detail + ("" if faces else " - no face seen (is anyone in front of the camera?)") + "; camera released")


@check("Foreground app detection")
def focus_check():
    from pet.focus import allowed_apps, classify, foreground_info
    exe, cls, pid = foreground_info()
    verdict = classify(exe, cls, pid, os.getpid(), allowed_apps())
    record("Foreground app detection", "PASS" if exe else "FAIL",
           f"foreground = {Path(exe).name if exe else '?'} -> "
           f"{'show Pette' if verdict else 'hide Pette' if verdict is False else 'neutral'}")


@check("db cache + world model")
def db_check():
    from db import Store, WorldModel
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "c.sqlite3")
        w = WorldModel(s)
        for _ in range(4):
            s.log_offer("soma:write_tests", "dismissed")
        s.log_event("away_ended", 5.0)
        ok = w.muted_kinds() == {"soma:write_tests"} and w.presence_today()["away_episodes"] == 1
        s.close()
    record("db cache + world model", "PASS" if ok else "FAIL", "write -> learn -> read round trip in a temp db")


SOMA_DIR = ROOT / "SOMA"
SOMA_PY = SOMA_DIR / ".venv" / "Scripts" / "python.exe"


@check("SOMA service (synthetic screen)")
def soma_check():
    """One SOMA step on a synthetic screen through the same serve script Pette's launcher uses."""
    name = "SOMA service (synthetic screen)"
    if not SOMA_PY.exists():
        record(name, "SKIP", "SOMA/.venv missing: run ./run_pette_soma.sh once")
        return
    with tempfile.TemporaryDirectory() as d:   # temp data dir: no consent or memory touches SOMA/data
        t = time.time()
        p = subprocess.run([str(SOMA_PY), "-m", "scripts.serve", "--config", "configs/pette.yaml", "--synthetic",
                            "--once", "--grant-screen", "--data-dir", d], cwd=SOMA_DIR, capture_output=True,
                           text=True, timeout=300, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    decision = next((ln for ln in p.stdout.splitlines() if ln.startswith("decision:")), "")
    ok = p.returncode == 0 and decision
    record(name, "PASS" if ok else "FAIL",
           f"{time.time() - t:.1f}s, {decision[:120]}" if ok else (p.stderr or p.stdout)[-300:])


@check("SOMA tests")
def soma_tests_check():
    name = "SOMA tests"
    if not SOMA_PY.exists():
        record(name, "SKIP", "SOMA/.venv missing")
        return
    t = time.time()
    p = subprocess.run([str(SOMA_PY), "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider"], cwd=SOMA_DIR,
                       capture_output=True, text=True, timeout=1800)
    last = (p.stdout.strip().splitlines() or ["?"])[-1]
    record(name, "PASS" if p.returncode == 0 else "FAIL", f"{last} ({time.time() - t:.0f}s)")


@check("UI states (screenshots)")
def ui_check(app):
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap

    from db import Store, WorldModel
    from pet import soma
    from pet.ui import Overlay
    from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
    from pet.watcher import Watcher
    from pet.worldview import WorldWindow

    SHOTS.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    store = Store(Path(tmp) / "c.sqlite3")
    world = WorldModel(store)
    for outcome in ("shown", "accepted", "shown", "accepted", "shown", "dismissed"):
        store.log_offer("soma:open_relevant_file", outcome)
    for _ in range(4):
        store.log_offer("soma:take_notes", "dismissed")
    store.log_event("away_ended", 18.0)

    class Fake:                                   # SOMA's API, scripted; nothing leaves this process
        queue: list = []

        def suggestion(self):
            return self.queue.pop(0) if self.queue else {"available": True, "silent": True}

        def trigger(self):
            return {"triggered": True}

        def folders(self):
            return {"folders": [{"path": "C:/Users/you/notes", "memories": 42}], "indexing": None}

        def __getattr__(self, name):              # feedback / pause / resume
            return lambda *a, **k: {}

    def card(**kw):
        Fake.queue.append({"available": True, "silent": False, "valid": True, "id": str(len(Fake.queue)) + kw["text"],
                           "evidence": [], "template": "", "file": None, "line": None, "memory_source": None, **kw})
        sug.poll_once()
        app.processEvents()

    o = Overlay(tray=False)
    sug = soma.Suggester(o, Fake(), memory=store, world=world, spawn=lambda f: f(), clipboard=lambda t: None,
                         open_file=lambda path, line=None: f"Opened calc.py at line {line} in VS Code.")
    w = Watcher(o, memory=store, world=world)
    win = WorldWindow(o, world, sug, w)
    sug.refresh_folders()
    o.place_default()
    o.show()
    shots = []
    # synthetic camera frame (no real camera in checks): a face blob and an open hand
    img = QImage(168, 168, QImage.Format_RGB888)
    img.fill(QColor("#c7b8a6"))
    pt = QPainter(img)
    pt.setBrush(QColor("#8d6e63"))
    pt.drawEllipse(54, 30, 60, 70)
    pt.end()
    hand = [(0.70, 0.95), (0.62, 0.85), (0.57, 0.76), (0.53, 0.68), (0.50, 0.62), (0.66, 0.66), (0.66, 0.54),
            (0.66, 0.46), (0.66, 0.40), (0.73, 0.65), (0.74, 0.52), (0.75, 0.44), (0.75, 0.37), (0.80, 0.67),
            (0.82, 0.56), (0.83, 0.48), (0.84, 0.42), (0.86, 0.71), (0.89, 0.63), (0.91, 0.57), (0.92, 0.52)]

    def camera(gesture):
        o.set_camera_indicator(True)
        o.orb.set_frame({"image": img, "hand": hand, "gesture": gesture})

    steps = [
        ("idle, camera off", lambda: None),
        ("camera: you + hand", lambda: camera("")),
        ("away", lambda: w.on_vision_event(FACE_AWAY)),
        ("eyes closed", lambda: (w.on_vision_event(FACE_BACK), w.on_vision_event(EYES_CLOSED))),
        ("click = look now", lambda: (w.on_vision_event(EYES_OPEN), o.orb.clicked.emit())),
        ("card: open file", lambda: card(action="OPEN_RELEVANT_FILE", text="Open calc.py at line 40",
                                         file="C:/repo/calc.py", line=40)),
        ("pinch = Do it", lambda: (camera("pinch"), w.on_vision_event(PINCH))),
        ("card: next step", lambda: (camera(""), card(action="SUGGEST_NEXT_STEP", template="review_changes",
                                                      text='Review your changes to "calc.py" before you commit'))),
        ("not now", lambda: o.bubble.cancel.click()),
        ("camera hidden (Ctrl+Alt+C)", lambda: (o.dismiss(), o.set_camera_visible(False))),
        ("paused", lambda: (o.set_camera_visible(True), o.set_paused(True))),
    ]
    done = {"v": False}

    def run(i=0):
        if i == len(steps):
            done["v"] = True
            return
        name, action = steps[i]
        action()
        QTimer.singleShot(700, lambda: (app.processEvents(), shots.append((name, o.grab())), run(i + 1)))

    QTimer.singleShot(300, run)
    deadline = time.time() + 40
    while not done["v"] and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    o.set_paused(False)
    win.show_near()
    t_end = time.time() + 3
    while time.time() < t_end:
        app.processEvents()
        time.sleep(0.02)
    shots.append(("world model window", win.grab()))
    o.hide()
    win.hide()
    store.close()

    for i, (name, pm) in enumerate(shots, 1):
        pm.save(str(SHOTS / f"{i:02d}_{re.sub(r'\W', '',name.split(' ')[0]).lower()}.png"))
    pad, cap = 14, 26
    cols = 5
    rows = [shots[i:i + cols] for i in range(0, len(shots), cols)]
    cw = max(pm.width() for _, pm in shots) + pad
    rh = max(pm.height() for _, pm in shots) + cap + pad
    sheet = QPixmap(cw * cols + pad, rh * len(rows) + pad)
    sheet.fill(QColor("#27272a"))
    painter = QPainter(sheet)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
    for r, row in enumerate(rows):
        for c, (name, pm) in enumerate(row):
            x, y = pad + c * cw, pad + r * rh
            painter.drawText(x, y + 16, f"{r * cols + c + 1}. {name}")
            painter.drawPixmap(x, y + rh - pad - pm.height(), pm)
    painter.end()
    sheet.save(str(OUT / "ui_states.png"))
    expected = len(steps) + 1
    record("UI states (screenshots)", "PASS" if len(shots) == expected else "FAIL",
           f"{len(shots)}/{expected} states rendered on the real display -> test_results/ui_states.png")


# --- report ------------------------------------------------------------------
def write_report(started: float):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    icon = {"PASS": "PASS", "FAIL": "**FAIL**", "SKIP": "SKIP", "INFO": "INFO"}
    tally = {k: sum(1 for _, s, _ in results if s == k) for k in ("PASS", "FAIL", "SKIP", "INFO")}
    lines = [
        "# Pette check report",
        "",
        f"- **When:** {datetime.now():%Y-%m-%d %H:%M} local, took {time.time() - started:.0f}s",
        f"- **Code:** `{branch}` @ `{commit}` (checks run on the working tree before this report was committed)",
        f"- **Result:** {tally['PASS']} pass, {tally['FAIL']} fail, {tally['SKIP']} skipped, {tally['INFO']} info",
        "- **Regenerate:** `py tools/run_checks.py` (add `--webhooks` to also post a test message)",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {icon[s]} | {d.replace('|', '/')} |" for n, s, d in results]
    if test_breakdown:
        lines += ["", "## Tests by file", "", "| File | Passed | Failed |", "|---|---|---|"]
        lines += [f"| `{f}` | {ok} | {bad} |" for f, (ok, bad) in sorted(test_breakdown.items())]
    if (OUT / "ui_states.png").exists():
        lines += ["", "## UI states", "", "![Pette UI states](ui_states.png)"]
    lines += ["", "Privacy: live checks record counts and timings only. No camera frames, screen images, file "
              "contents or suggestion text were saved. SOMA's check runs on a synthetic screen in a temp data dir."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps([{"check": n, "status": s, "detail": d} for n, s, d in results],
                                                 indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-soma-tests", action="store_true", help="skip SOMA's own pytest suite")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    started = time.time()

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    env_check()
    pytest_check()
    models_check()
    if args.no_camera:
        record("Webcam: face / eyes / hands (live)", "SKIP", "--no-camera")
    else:
        camera_check()
    focus_check()
    db_check()
    soma_check()
    if args.no_soma_tests:
        record("SOMA tests", "SKIP", "--no-soma-tests")
    else:
        soma_tests_check()
    ui_check(app)
    write_report(started)
    print(f"\nwrote {OUT / 'REPORT.md'}")
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
