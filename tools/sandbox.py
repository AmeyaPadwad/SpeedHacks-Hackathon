"""Manual test sandbox: the real Pette app plus a control panel for checking every feature by hand.

    py tools/sandbox.py                    # real Groq + voice, sandbox webhooks/IDE/db
    py tools/sandbox.py --demo             # offline: typed commands, keyword drafter
    py tools/sandbox.py --live-webhooks    # really post to the Slack/Discord webhooks in .env
    py tools/sandbox.py --real-ide         # use the VS Code extension's snapshot instead of the fake one

The app is built by main.build(), so hotkeys, tray, focus gate, watcher and pipeline are the real ones.
The sandbox only swaps the outside world:
  - Slack/Discord webhooks -> a local inbox shown in the panel (nothing reaches real channels)
  - email -> not configured (SMTP_HOST blank), unless --live-webhooks
  - VS Code snapshot -> a temp file pointing at a throwaway git repo (uncommitted fix / clean / no file)
  - memory db + contacts -> a temp folder, deleted on exit; your real db/ cache is untouched
Webcam events can be simulated from the panel; the real webcam feeds the orb unless --no-camera.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "fixtures" / "buggy_app" / "dashboard.py"
REPORT = ROOT / "test_results" / "MANUAL_TEST.md"

CONTACTS = {
    "sarah": {"slack": "U0SARAH", "discord": "111111111111111111", "email": "sarah@example.com"},
    "alex": {"slack": "U0ALEX", "discord": "222222222222222222"},
}

# (area, scenario, how, expected). Voice steps: in --demo / --no-voice, type the same words instead.
SCENARIOS = [
    ("Launch", "Startup", "Start the sandbox with VS Code or a terminal in front.",
     "Pette, the camera orb above her (you + a grey ring) and the world-model window beside her."),
    ("Workflow", "Click to talk", "Panel: “Buggy file (uncommitted fix)”. Click Pette, say “Slack Sarah what I changed in this file”.",
     "Orb ring turns violet “Listening”; your words appear in the bubble as “You said …”. Flow-chan is silent."),
    ("Workflow", "Groq draft", "Wait a second after speaking.",
     "Orb “Thinking”, then a “Slack to Sarah” card with a summary of the change and “＋ dashboard.py · uncommitted changes”."),
    ("Workflow", "Auto-dispatch", "Do nothing during the countdown.",
     "Gradient border + “Sending in Ns”; then inbox shows SLACK with the summary and a ```diff``` block; orb “Sent ✓”."),
    ("Workflow", "Send now", "Repeat, click “Send now” during the countdown.", "Sent immediately; one post only."),
    ("Workflow", "Cancel", "Repeat, click “Cancel · Ns”.", "Nothing in the inbox; orb “Not sent”; bubble says nothing was sent."),
    ("Workflow", "Click again stops", "Click Pette, then click her again before speaking.", "Listening stops; back to idle."),
    ("Workflow", "Discord + code limit", "“Discord Alex what changed”.",
     "DISCORD post ≤ 2000 chars, allowed_mentions only Alex, code block truncated with “… (truncated)” if long."),
    ("Workflow", "No changes", "Panel: “Committed file (no changes)”, then message Sarah about it.",
     "Message summarises the file; the file itself is attached as code (no diff)."),
    ("Workflow", "Email not configured", "“Email Sarah the summary”.",
     "“Email isn't configured, so nothing was sent.” Orb “Not sent”."),
    ("Workflow", "Webhook failure", "Tick “Make webhooks fail”, send a Slack message.",
     "Bubble “Webhook returned HTTP 500”, orb red “Not sent”. No retry."),
    ("Workflow", "Question", "Ask “what does this file do?”.", "Plain answer in the bubble, nothing sent."),
    ("Workflow", "Typed fallback", "Run with --no-voice (or deny the mic), click Pette.", "A text box opens; typing works the same."),
    ("Orb", "You + hands", "Real camera on: sit in front of it, raise an open hand, then pinch.",
     "Mirrored you in the circle, a hand skeleton, and a “pinch” chip only while pinching."),
    ("Orb", "Hide camera hotkey", "Press Ctrl+Alt+C twice (or tray → Hide / show camera view).",
     "The orb disappears and comes back; Pette doesn't move; pinch still sends a pending draft while hidden."),
    ("World model", "Hide panel hotkey", "Press Ctrl+Alt+W twice.", "The world-model panel hides, then docks beside Pette again."),
    ("Bubble", "Glass", "Show any bubble over a busy window (code, a photo).",
     "Frosted, translucent card: the window behind shows through; text and buttons stay readable."),
    ("Orb", "Status ring", "Watch the ring through one full send.",
     "violet Listening → blue Thinking → amber Sending in Ns → green Sent ✓ (or grey/red Not sent)."),
    ("Webcam", "Pinch rules", "Pinch when idle; pinch during a countdown.",
     "Idle → nothing happens. Countdown → sent now (same as Send now), once."),
    ("Webcam", "Away / blink / nudge", "Start a send, Face away, Face back; Eyes closed, Eyes open.",
     "Pette dozes while away; back → “Still sending?”; blinks on eyes closed."),
    ("World model", "Live status", "Watch the window during a send.",
     "Status pill mirrors the orb (moving border while busy); camera/hand/presence tiles update."),
    ("World model", "Learns platform", "Send 2 Slack messages to Sarah, then “message Sarah saying hi” (no platform).",
     "Draft is “Slack (usual) to Sarah”; window shows the rule “message Sarah → Slack” with the counts."),
    ("World model", "Adaptive countdown", "Cancel 3 of your next 6 messages.",
     "Window: countdown rule changes to 6s “because you cancelled 3 of 6”; the next send waits 6s."),
    ("World model", "Clear", "Tray → Clear Pette memory.", "Window resets to defaults; no “(usual)”."),
    ("Control", "Pause", "Tray → Pause, click Pette, press the hotkey, Pinch; then unpause.",
     "Paused: nothing reacts, camera off, countdown stopped. Unpaused: works again."),
    ("Control", "Focus gate", "Switch to a browser, then back to VS Code.",
     "Pette hides outside VS Code/terminals and reappears; clicks/hotkey ignored while hidden."),
    ("SOMA", "Suggestion appears", "./run_pette_soma.sh (or sandbox with --soma <endpoint>); edit a file in VS Code.",
     "Within a few seconds: glass card “SOMA suggests · Next step” (e.g. summary of your changes), Do it / Not now."),
    ("SOMA", "Nothing runs on its own", "Leave the card alone for 15 s.",
     "The card goes away; nothing is drafted or sent (SOMA logs it as ignored, not dismissed)."),
    ("SOMA", "Do it", "Accept a “Send the team a summary…” card.",
     "Normal flow: Thinking → draft to the team with the diff → countdown; Cancel still stops it."),
    ("SOMA", "Not now", "Dismiss a card.", "It doesn't come back for 5 minutes; tray/world counts it as dismissed."),
    ("SOMA", "Toggle + pause", "Ctrl+Alt+S, then tray → Pause all.",
     "“Suggestions off” and no cards; Pause also pauses SOMA's capture. Resume brings suggestions back."),
    ("SOMA", "Diagnose", "./run_pette_soma.sh --diagnose with VS Code in front.",
     "Prints SOMA's candidates and why it would speak or stay silent."),
]


# --- local webhook inbox (stdlib only; runs before the app is imported) -----------------------
class Inbox:
    def __init__(self):
        self.fail = False
        self.on_post = lambda platform, payload, status: None
        inbox = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    payload = json.loads(raw or b"{}")
                except ValueError:
                    payload = {"_raw": raw.decode("utf-8", "replace")}
                status = 500 if inbox.fail else (204 if self.path.endswith("discord") else 200)
                inbox.on_post(self.path.strip("/"), payload, status)
                self.send_response(status)
                self.end_headers()
                if status == 200:
                    self.wfile.write(b"ok")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def parse():
    ap = argparse.ArgumentParser(description="Pette manual test sandbox", epilog="Other flags go to main.py.")
    ap.add_argument("--live-webhooks", action="store_true", help="post to the real webhooks/SMTP from .env")
    ap.add_argument("--real-ide", action="store_true", help="read the real VS Code extension snapshot")
    return ap.parse_known_args()


def setup_env(args, tmp: Path, inbox: Inbox | None):
    """Must run before main/pet are imported: they read these settings at import time."""
    os.environ["FLOWSTATE_DB"] = str(tmp / "cache.sqlite3")
    os.environ["FLOWSTATE_CONTACTS"] = str(tmp / "contacts.json")
    (tmp / "contacts.json").write_text(json.dumps(CONTACTS, indent=2), encoding="utf-8")
    if not args.real_ide:
        os.environ["FLOWSTATE_ACTIVE_FILE"] = str(tmp / "active_file.json")
    if inbox:
        os.environ["SLACK_WEBHOOK_URL"] = inbox.url + "/slack"
        os.environ["DISCORD_WEBHOOK_URL"] = inbox.url + "/discord"
        os.environ["SMTP_HOST"] = ""  # blank wins over .env (load_dotenv never overrides)


FIXED = '''"""Demo fixture: a deliberate bug for the FlowState investigate -> coding agent flow."""


def load_user(user_id: int) -> dict | None:
    users = {1: {"name": "Sarah", "stats": {"commits": 42}}}
    return users.get(user_id)


def render_dashboard(user_id: int) -> str:
    user = load_user(user_id)
    if user is None:
        return "Unknown user"
    commits = user.get("stats", {}).get("commits", 0)
    return f"{user['name']} has {commits} commits"


if __name__ == "__main__":
    print(render_dashboard(1))
    print(render_dashboard(2))
'''


def git(repo: Path, *args):
    subprocess.run(["git", "-c", "user.email=sandbox@pette", "-c", "user.name=Pette sandbox", *args], cwd=repo,
                   check=True, capture_output=True)


def make_fixture_repo(tmp: Path) -> Path:
    """A throwaway git repo with the buggy fixture committed; the panel layers the fix on top as a diff."""
    repo = tmp / "buggy_app"
    repo.mkdir()
    shutil.copy2(FIXTURE, repo / FIXTURE.name)
    git(repo, "init", "-q")
    git(repo, "add", FIXTURE.name)
    git(repo, "commit", "-q", "-m", "buggy dashboard")
    return repo


def fixture_hashes() -> dict[str, str]:
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in FIXTURE.parent.glob("*.py")}


def main() -> int:
    args, rest = parse()
    tmp = Path(tempfile.mkdtemp(prefix="pette-sandbox-"))
    inbox = None if args.live_webhooks else Inbox()
    setup_env(args, tmp, inbox)

    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                                   QHeaderView, QLabel, QPlainTextEdit, QPushButton, QTableWidget,
                                   QTableWidgetItem, QVBoxLayout, QWidget)

    class Bridge(QObject):
        post = Signal(str, object, int)   # queued onto the main thread from the HTTP thread

    app = QApplication.instance() or QApplication(sys.argv)
    bridge = Bridge()
    if inbox:
        inbox.on_post = bridge.post.emit

    import main as app_main
    from pet import context, llm, voice
    from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH

    main_args = app_main.parse_args(rest)
    ctx = app_main.build(main_args)
    pipeline, watcher, overlay = ctx.pipeline, ctx.watcher, ctx.overlay
    hashes_before = fixture_hashes()

    # --- panel --------------------------------------------------------------------------
    panel = QWidget(None, Qt.Window)
    panel.setWindowTitle("Pette sandbox · manual tests")
    panel.resize(1180, 820)
    root = QVBoxLayout(panel)

    brain = "DEMO (offline keyword drafter)" if main_args.demo else f"Groq {llm.chat_model()}"
    voice_on = not (main_args.demo or main_args.no_voice)
    voice_mode = (f"ElevenLabs {ctx.settings.mode}" if ctx.settings else f"Whisper ({llm.stt_model()})") if voice_on else "typed"
    status = QLabel()
    status.setWordWrap(True)
    status.setTextInteractionFlags(Qt.TextSelectableByMouse)

    hooks = '<span style="color:#A14B3A">LIVE (.env)</span>' if inbox is None else f"sandbox inbox {inbox.url}"

    def refresh_status():
        af = context.read_active_file()
        file_state = (f"{Path(af.path).name} ({'uncommitted changes' if context.git_changes(af) else 'no changes'})"
                      if af else "no file")
        status.setText(
            f"<b>Brain:</b> {brain} &nbsp; <b>Voice:</b> {voice_mode} &nbsp; "
            f"<b>State:</b> {pipeline.state} &nbsp; <b>Send countdown:</b> {pipeline.delay_s():g}s &nbsp; "
            f"<b>Hotkey:</b> {voice.HOTKEY}<br>"
            f"<b>Webhooks:</b> {hooks}"
            f" &nbsp; <b>IDE:</b> {'real VS Code extension' if args.real_ide else 'fake snapshot'} → {file_state}"
            f" &nbsp; <b>Busy:</b> {pipeline.busy} &nbsp; <b>Paused:</b> {overlay.paused}"
            f" &nbsp; <b>Repo fixture untouched:</b> {fixture_hashes() == hashes_before}<br>"
            f"<b>Temp state:</b> {tmp}")

    root.addWidget(status)
    status_timer = QTimer(panel, interval=1000, timeout=refresh_status)
    status_timer.start()

    grid = QGridLayout()
    root.addLayout(grid)

    def group(title, row, col, buttons):
        box = QGroupBox(title)
        lay = QHBoxLayout(box)
        for label, fn in buttons:
            if isinstance(label, QWidget):
                lay.addWidget(label)
                continue
            b = QPushButton(label)
            b.clicked.connect(fn)
            lay.addWidget(b)
        grid.addWidget(box, row, col)
        return box

    # fake VS Code snapshot, pointing at a throwaway git repo so "what changed" is a real diff
    snap = Path(os.environ.get("FLOWSTATE_ACTIVE_FILE", ""))
    repo = make_fixture_repo(tmp)
    target = repo / FIXTURE.name

    def write_snapshot(mode: str | None):
        """mode: 'fix' = uncommitted fix (diff), 'clean' = committed file, None = no file."""
        if mode is None:
            snap.unlink(missing_ok=True)
        else:
            git(repo, "checkout", "-q", "--", FIXTURE.name)
            if mode == "fix":
                target.write_text(FIXED, encoding="utf-8")
            snap.write_text(json.dumps({
                "path": str(target), "workspace": str(repo), "language": "python",
                "content": target.read_text(encoding="utf-8"), "selection": "", "line": 12,
                "diagnostics": [], "updatedAt": dt.datetime.now().isoformat()}), encoding="utf-8")
        refresh_status()

    ide = group("Fake VS Code (active file)", 0, 0, [
        ("Buggy file (uncommitted fix)", lambda: write_snapshot("fix")),
        ("Committed file (no changes)", lambda: write_snapshot("clean")),
        ("No file", lambda: write_snapshot(None)),
    ])
    if args.real_ide:
        ide.setEnabled(False)
        ide.setTitle("Fake VS Code (off: --real-ide)")
    else:
        write_snapshot("fix")

    group("Simulated webcam events", 1, 0, [
        (n, (lambda ev=ev: watcher.on_vision_event(ev)))
        for n, ev in (("Face away", FACE_AWAY), ("Face back", FACE_BACK), ("Eyes closed", EYES_CLOSED),
                      ("Eyes open", EYES_OPEN), ("Pinch", PINCH))
    ])

    group("Triggers", 2, 0, [
        ("Click Pette (talk)", overlay.pet.clicked.emit),
        ("Pause / resume", lambda: overlay.set_paused(not overlay.paused)),
        ("World model window", ctx.worldwin.toggle),
        ("Camera on/off", lambda: watcher.set_camera(watcher.camera is None)),
    ])

    fail = QCheckBox("Make webhooks fail (HTTP 500)")
    fail.toggled.connect(lambda on: setattr(inbox, "fail", on) if inbox else None)
    fail.setEnabled(inbox is not None)
    inbox_view = QPlainTextEdit(readOnly=True)
    inbox_view.setPlaceholderText("Slack / Discord posts land here instead of real channels.")
    clear_inbox = QPushButton("Clear")
    clear_inbox.clicked.connect(inbox_view.clear)
    box = QGroupBox("Webhook inbox" + (" (disabled: --live-webhooks)" if not inbox else ""))
    lay = QVBoxLayout(box)
    row = QHBoxLayout()
    row.addWidget(fail)
    row.addStretch()
    row.addWidget(clear_inbox)
    lay.addLayout(row)
    lay.addWidget(inbox_view)
    grid.addWidget(box, 0, 1, 3, 1)
    inbox_view.setMinimumHeight(120)
    grid.setColumnStretch(0, 3)
    grid.setColumnStretch(1, 2)

    def on_post(platform, payload, code):
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        inbox_view.appendPlainText(f"[{stamp}] {platform.upper()} → HTTP {code}\n"
                                   f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n")

    bridge.post.connect(on_post)

    # checklist
    table = QTableWidget(len(SCENARIOS), 6)
    table.setHorizontalHeaderLabels(["Area", "Scenario", "How", "Expected", "Result", "Notes"])
    for r, (area, name, how, expected) in enumerate(SCENARIOS):
        for c, text in enumerate((area, name, how, expected)):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setToolTip(text)
            table.setItem(r, c, item)
        combo = QComboBox()
        combo.addItems(["—", "Pass", "Fail", "Skip"])
        table.setCellWidget(r, 4, combo)
        table.setItem(r, 5, QTableWidgetItem(""))
    hdr = table.horizontalHeader()
    for c, mode in enumerate((QHeaderView.ResizeToContents, QHeaderView.ResizeToContents, QHeaderView.Stretch,
                              QHeaderView.Stretch, QHeaderView.ResizeToContents, QHeaderView.Interactive)):
        hdr.setSectionResizeMode(c, mode)
    table.setColumnWidth(5, 160)
    table.setWordWrap(True)
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    root.addWidget(table, 1)

    summary = QLabel()
    save = QPushButton("Save report → test_results/MANUAL_TEST.md")

    def results():
        return [table.cellWidget(r, 4).currentText() for r in range(table.rowCount())]

    def update_summary():
        res = results()
        summary.setText(f"{res.count('Pass')} pass · {res.count('Fail')} fail · {res.count('Skip')} skip · "
                        f"{res.count('—')} not run")

    for r in range(table.rowCount()):
        table.cellWidget(r, 4).currentTextChanged.connect(lambda _: update_summary())
    update_summary()

    def save_report():
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
        lines = ["# Manual test report", "",
                 f"- Date: {dt.datetime.now():%Y-%m-%d %H:%M}", f"- Commit: `{commit}`",
                 f"- Brain: {brain}; voice: {voice_mode}; send countdown: {pipeline.delay_s():g}s",
                 f"- Webhooks: {'LIVE' if args.live_webhooks else 'sandbox inbox'}; "
                 f"IDE: {'real extension' if args.real_ide else 'fake snapshot'}",
                 f"- Repo fixture untouched (sandbox works on a copy): {fixture_hashes() == hashes_before}",
                 f"- Result: {summary.text()}", "",
                 "| Area | Scenario | Result | Notes |", "|---|---|---|---|"]
        for r, (area, name, _, _) in enumerate(SCENARIOS):
            note = (table.item(r, 5).text() or "").replace("|", "/")
            lines.append(f"| {area} | {name} | {results()[r]} | {note} |")
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        save.setText(f"Saved {dt.datetime.now():%H:%M:%S} → {REPORT.relative_to(ROOT)}")

    save.clicked.connect(save_report)
    foot = QHBoxLayout()
    foot.addWidget(summary)
    foot.addStretch()
    foot.addWidget(save)
    root.addLayout(foot)

    def cleanup():
        if inbox:
            inbox.server.shutdown()
        time.sleep(0.1)
        shutil.rmtree(tmp, ignore_errors=True)

    app.aboutToQuit.connect(cleanup)
    panel.closeEvent = lambda e: (e.accept(), app.quit())   # closing the panel ends the session
    refresh_status()
    panel.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
