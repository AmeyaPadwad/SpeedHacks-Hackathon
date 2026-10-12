"""Run every Pette check and write test_results/REPORT.md (+ junit.xml and UI screenshots).

    py tools/run_checks.py                 # all checks except ones with outside side effects
    py tools/run_checks.py --webhooks      # also post a labelled test message to Slack/Discord
    py tools/run_checks.py --no-camera

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
    import openai
    import PySide6
    record("Environment", "INFO", f"Windows {platform.release()} ({platform.version()}), Python {platform.python_version()}, "
           f"PySide6 {PySide6.__version__}, OpenCV {cv2.__version__}, MediaPipe {mediapipe.__version__}, "
           f"openai {openai.__version__}")


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


@check("VS Code bridge snapshot")
def bridge_check():
    from pet.context import ACTIVE_FILE, read_active_file
    af = read_active_file()
    if af is None:
        record("VS Code bridge snapshot", "SKIP", f"no snapshot at ~/{ACTIVE_FILE.name} (extension not installed or no file open)")
        return
    age = time.time() - ACTIVE_FILE.stat().st_mtime
    record("VS Code bridge snapshot", "PASS", f"{Path(af.path).name} ({af.language}), {len(af.diagnostics)} diagnostic(s), "
           f"updated {age:.0f}s ago")


@check("db cache + world model")
def db_check():
    from db import Store, WorldModel
    with tempfile.TemporaryDirectory() as d:
        s = Store(Path(d) / "c.sqlite3")
        w = WorldModel(s)
        s.log_message("sarah", "slack", "sent")
        s.log_event("away_ended", 5.0)
        ok = w.preferred_platform("Sarah") == "slack" and w.presence_today()["away_episodes"] == 1
        s.close()
    record("db cache + world model", "PASS" if ok else "FAIL", "write -> learn -> read round trip in a temp db")


@check("Groq (chat + Whisper)")
def openai_check():
    from pet import llm
    name = f"Groq ({llm.chat_model()} / {llm.stt_model()})"
    client = llm.client(20)
    if client is None:
        record(name, "SKIP", f"{llm.KEY_VAR} not set")
        return
    try:
        client.chat.completions.create(model=llm.chat_model(), max_tokens=1, messages=[{"role": "user", "content": "hi"}])
        record(name, "PASS", "1-token request succeeded")
    except Exception as e:
        record(name, "FAIL", llm.describe_error(e) + " (app falls back to labelled local analysis)")


@check("ElevenLabs transcription")
def eleven_check():
    """Speak a known sentence with Windows' built-in TTS, stream it as the 'mic', and compare the transcript."""
    import wave

    from pet import eleven
    name = "ElevenLabs transcription"
    settings = eleven.Settings.from_env()
    if settings is None:
        record(name, "SKIP", "ELEVENLABS_AGENT_ID / ELEVENLABS_API_KEY not set (or FLOWSTATE_VOICE=whisper)")
        return
    said = "Send a Slack message to Sarah saying the dashboard bug is fixed."
    wav_path = Path(tempfile.mkdtemp()) / "said.wav"
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, "
          "([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{wav_path}', $f); $s.Speak('{said}'); $s.Dispose()")
    import shutil
    shell = shutil.which("powershell") or str(
        Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    subprocess.run([shell, "-NoProfile", "-Command", ps], check=True, capture_output=True, timeout=60)
    speech = wave.open(str(wav_path)).readframes(10 ** 9)
    t = time.time()
    if settings.mode == "scribe":
        import io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(speech)
        text = eleven.scribe(buf.getvalue(), settings.api_key)
    else:
        class WavMic:
            def __init__(self):
                self.buf = b"\0\0" * 16000 + speech + b"\0\0" * 16000 * 20

            def open(self):
                pass

            def read(self, frames):
                time.sleep(frames / 16000)
                chunk, self.buf = self.buf[:frames * 2], self.buf[frames * 2:]
                return chunk or b"\0\0" * frames

            def close(self):
                pass
        text = eleven.LiveTranscriber(settings, audio=WavMic()).run()
    words = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    recall = len(words(said) & words(text)) / len(words(said))
    detail = f"{settings.mode} mode, {time.time() - t:.1f}s: “{text}” ({recall:.0%} of words)"
    if recall >= 0.8:
        record(name, "PASS", detail)
    elif settings.mode == "live":
        record(name, "FAIL", detail + ". The agent's spoken greeting swallows the start of the command: clear "
                                      "'First message' in the ElevenLabs agent, or set ELEVENLABS_API_KEY for Scribe")
    else:
        record(name, "FAIL", detail)


@check("Slack / Discord webhooks")
def webhook_check(enabled):
    from pet.brain import Draft
    from pet.dispatch import configured, dispatch
    for p in ("slack", "discord"):
        if not configured(p):
            record(f"{p.title()} webhook", "SKIP", "not configured")
        elif not enabled:
            record(f"{p.title()} webhook", "SKIP", "configured; live post skipped (run with --webhooks)")
        else:
            msg = dispatch(Draft(p, "team", "", "[Pette check] Automated webhook test. Ignore."), {})
            record(f"{p.title()} webhook", "PASS", msg)


@check("UI states (screenshots)")
def ui_check(app):
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap

    from db import Store, WorldModel
    from pet.brain import OfflineBrain
    from pet.context import ActiveFile, Context
    from pet.pipeline import Pipeline
    from pet.ui import Overlay
    from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
    from pet.watcher import Watcher
    from pet.worldview import WorldWindow
    import pet.pipeline as pp
    pp.MIN_INTERVAL_S = 0

    SHOTS.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    store = Store(Path(tmp) / "c.sqlite3")
    world = WorldModel(store)
    for _ in range(3):
        store.log_message("sarah", "slack", "sent")
    store.log_event("away_ended", 18.0)
    af = ActiveFile("C:/repo/dashboard.py", "python", "x = 1\n")
    diff = "--- a/dashboard.py\n+++ b/dashboard.py\n+    if user is None:\n+        return 'Unknown user'"

    o = Overlay(tray=False)
    p = Pipeline(o, OfflineBrain(), use_voice=False, spawn=lambda f: f(), capture=lambda: Context(af, diff),
                 send=lambda d: f"Sent to {d.recipient} on {d.platform.title()}.", configured=lambda _: True,
                 memory=store, world=world, send_delay_s=4)
    w = Watcher(o, memory=store, world=world)
    win = WorldWindow(o, world, p, w)
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

    def tick_all():
        while p._countdown.isActive():
            p._tick()

    steps = [
        ("idle, camera off", lambda: None),
        ("camera: you + hand", lambda: camera("open hand")),
        ("away", lambda: w.on_vision_event(FACE_AWAY)),
        ("eyes closed", lambda: (w.on_vision_event(FACE_BACK), w.on_vision_event(EYES_CLOSED))),
        ("pinch = talk", lambda: (w.on_vision_event(EYES_OPEN), camera("pinch"), w.on_vision_event(PINCH),
                                  o.bubble.input.setText("message Sarah what I changed"))),
        ("draft + countdown", lambda: (camera("open hand"), o.bubble.primary.click())),
        ("sent", tick_all),
        ("cancelled", lambda: (o.dismiss(), p.trigger(), o.bubble.input.setText("Slack Sarah the fix"),
                               o.bubble.primary.click(), o.bubble.cancel.click())),
        ("paused", lambda: o.set_paused(True)),
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
        pm.save(str(SHOTS / f"{i:02d}_{name.split(' ')[0].replace(',', '')}.png"))
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
              "contents or message text were saved."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps([{"check": n, "status": s, "detail": d} for n, s, d in results],
                                                 indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhooks", action="store_true", help="post a labelled test message to configured webhooks")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-agent", action="store_true", help=argparse.SUPPRESS)   # old flag, no agents any more
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
    bridge_check()
    db_check()
    openai_check()
    eleven_check()
    webhook_check(args.webhooks)
    ui_check(app)
    write_report(started)
    print(f"\nwrote {OUT / 'REPORT.md'}")
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
