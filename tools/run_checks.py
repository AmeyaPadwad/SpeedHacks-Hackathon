"""Run every Pette check and write test_results/REPORT.md (+ junit.xml and UI screenshots).

    py tools/run_checks.py                 # all checks except ones with outside side effects
    py tools/run_checks.py --webhooks      # also post a labelled test message to Slack/Discord
    py tools/run_checks.py --no-camera --no-agent

Live checks report numbers only: no camera frames, screen images or file contents are saved.
"""
import argparse
import hashlib
import json
import os
import platform
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


@check("Claude Code agent (live, read-only)")
def agent_check():
    from pet.agents import AgentTask, ClaudeCodeAgent
    from pet.analyst import local_analysis
    from pet.context import ActiveFile, Context, Diagnostic
    agent = ClaudeCodeAgent()
    if not agent.available():
        record("Claude Code agent (live, read-only)", "SKIP", "claude CLI not on PATH")
        return
    ws = ROOT / "fixtures" / "buggy_app"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ws.glob("*.py")}
    af = ActiveFile(str(ws / "dashboard.py"), "python", (ws / "dashboard.py").read_text(), workspace=str(ws),
                    diagnostics=(Diagnostic(12, "error", "TypeError: 'NoneType' object is not subscriptable", "pytest"),))
    ctx = Context(None, af)
    t = time.time()
    res = agent.run(AgentTask.from_context(local_analysis(ctx, "check"), ctx))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ws.glob("*.py")}
    found = "none" in res.text.lower() and ("diff" in res.text.lower() or "@@" in res.text)
    unchanged = before == after
    record("Claude Code agent (live, read-only)", "PASS" if found and unchanged else "FAIL",
           f"answered in {time.time() - t:.0f}s, {len(res.text)} chars, root cause + diff: {'yes' if found else 'no'}, "
           f"fixture files unchanged: {'yes' if unchanged else 'NO'}")


@check("Hermes agent")
def hermes_check():
    from pet.agents import HermesAgent
    record("Hermes agent", "INFO" if HermesAgent().available() else "SKIP",
           "installed (adapter unit-tested; live run not part of checks)" if HermesAgent().available() else "not installed")


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
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap

    from db import Store, WorldModel
    from pet.agents import AgentResult
    from pet.analyst import LocalAnalyst
    from pet.brain import OfflineBrain
    from pet.context import ActiveFile, Context, Diagnostic
    from pet.pipeline import Pipeline
    from pet.ui import Overlay
    from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
    from pet.watcher import Watcher
    import pet.pipeline as pp
    pp.MIN_INTERVAL_S = 0

    SHOTS.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    store = Store(Path(tmp) / "c.sqlite3")
    world = WorldModel(store)
    for _ in range(3):
        store.log_message("sarah", "slack", "sent")
    store.log_event("away_ended", 18.0)
    af = ActiveFile("C:/repo/dashboard.py", "python", "", diagnostics=(Diagnostic(12, "error", "TypeError: 'NoneType'", "pytest"),))

    class FakeAgent:
        name, label = "claude", "Claude Code"

        def run(self, task):
            return AgentResult("claude", "diff")

    o = Overlay(tray=False)
    p = Pipeline(o, OfflineBrain(), analyst=LocalAnalyst(), use_voice=False, spawn=lambda f: f(), capture_delay_ms=0,
                 capture=lambda: Context(None, af), read_file=lambda: af,
                 send=lambda d: f"Sent to {d.recipient} on {d.platform.title()}.", configured=lambda _: True,
                 pick_agent=lambda n=None: FakeAgent(), memory=store, world=world)
    w = Watcher(o, lambda: p.busy, read_file=lambda: None, memory=store, world=world)
    o.place_default()
    o.show()
    shots = []

    def typed(text):
        o.bubble.input.setText(text)
        o.bubble.primary.click()

    steps = [
        ("idle", lambda: None),
        ("camera on, away", lambda: (o.set_camera_indicator(True), w.on_vision_event(FACE_AWAY))),
        ("eyes closed", lambda: (w.on_vision_event(FACE_BACK), w.on_vision_event(EYES_CLOSED))),
        ("pinch = push-to-talk", lambda: (w.on_vision_event(EYES_OPEN), w.on_vision_event(PINCH),
                                          o.bubble.input.setText("message Sarah saying the fix is in"))),
        ("learned platform draft", lambda: o.bubble.primary.click()),
        ("back from away: nudge", lambda: (w.on_vision_event(FACE_AWAY), w.on_vision_event(FACE_BACK))),
        ("sent", lambda: o.bubble.primary.click()),
        # offers never interrupt a visible bubble, so clear the "Sent" confirmation first
        ("proactive offer", lambda: (o.dismiss(), p.offer_investigation("dashboard.py line 12: TypeError: 'NoneType'"))),
        ("CV analysis -> agent", lambda: o.bubble.primary.click()),
        ("what Pette learned", lambda: (o.bubble.cancel.click(), o.say(world.summary(), title="Pette's memory",
                                                                       seconds=None))),
        ("paused", lambda: o.set_paused(True)),
    ]
    done = {"v": False}

    def run(i=0):
        if i == len(steps):
            done["v"] = True
            return
        name, action = steps[i]
        action()
        QTimer.singleShot(600, lambda: (app.processEvents(), shots.append((name, o.grab())), run(i + 1)))

    QTimer.singleShot(300, run)
    deadline = time.time() + 30
    while not done["v"] and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    o.hide()
    store.close()

    for i, (name, pm) in enumerate(shots, 1):
        pm.save(str(SHOTS / f"{i:02d}_{name.split(' ')[0].replace(',', '')}.png"))
    pad, cap = 14, 26
    cols = 6
    rows = [shots[i:i + cols] for i in range(0, len(shots), cols)]
    cw = max(pm.width() for _, pm in shots) + pad
    rh = max(pm.height() for _, pm in shots) + cap + pad
    sheet = QPixmap(cw * cols + pad, rh * len(rows) + pad)
    sheet.fill(QColor("#3A4A5C"))
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
    record("UI states (screenshots)", "PASS" if len(shots) == len(steps) else "FAIL",
           f"{len(shots)}/{len(steps)} states rendered on the real display -> test_results/ui_states.png")


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
    ap.add_argument("--no-agent", action="store_true")
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
    if args.no_agent:
        record("Claude Code agent (live, read-only)", "SKIP", "--no-agent")
    else:
        agent_check()
    hermes_check()
    webhook_check(args.webhooks)
    ui_check(app)
    write_report(started)
    print(f"\nwrote {OUT / 'REPORT.md'}")
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
