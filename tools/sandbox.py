"""Manual test sandbox: the real Pette app plus a control panel for checking every feature by hand.

    py tools/sandbox.py                    # real OpenAI + voice, sandbox webhooks/IDE/db
    py tools/sandbox.py --demo             # offline: typed commands, keyword drafter
    py tools/sandbox.py --live-webhooks    # really post to the Slack/Discord webhooks in .env
    py tools/sandbox.py --real-ide         # use the VS Code extension's snapshot instead of the fake one

The app is built by main.build(), so hotkeys, tray, focus gate, watcher and pipeline are the real ones.
The sandbox only swaps the outside world:
  - Slack/Discord webhooks -> a local inbox shown in the panel (nothing reaches real channels)
  - email -> not configured (SMTP_HOST blank), unless --live-webhooks
  - VS Code snapshot -> a temp file the panel writes (buggy fixture / no errors / no file)
  - memory db + contacts -> a temp folder, deleted on exit; your real db/ cache is untouched
Webcam events can be simulated from the panel; --camera still turns on the real webcam.
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
     "Pette appears bottom-right with a greeting that names the mode (DEMO or hotkeys)."),
    ("Message", "Slack draft", "Ctrl+Shift+Space, say “Slack Sarah saying the build is green”.",
     "Bubble titled “Slack to Sarah” with a drafted message and Send / Cancel."),
    ("Message", "Send", "Click Send on the Slack draft.",
     "Inbox shows SLACK “<@U0SARAH> …”; bubble says “Sent to Sarah on Slack.”"),
    ("Message", "Cancel", "Draft another message, click Cancel.", "Nothing new in the inbox."),
    ("Message", "Discord + mentions", "“Discord Alex saying standup moved to 3”, Send.",
     "Inbox shows DISCORD with allowed_mentions.users = [Alex's id] only."),
    ("Message", "Email not configured", "“Email Sarah asking for the report”, Send.",
     "Title ends “(not configured)”; Send shows “SMTP_HOST is not set.” Nothing sent."),
    ("Message", "Webhook failure", "Tick “Make webhooks fail”, draft + Send a Slack message.",
     "Bubble: “Webhook returned HTTP 500”. No retry; inbox shows the rejected post."),
    ("Message", "Question", "Ask “what does this file do?”.",
     "Plain answer in the bubble, no Send button. (Demo mode: says it can only draft messages.)"),
    ("Message", "Typed fallback", "Click Pette (or ask with the mic muted/denied).",
     "A text box opens; typing a command works the same as speaking."),
    ("Investigate", "File handoff", "Panel: “Buggy file + error”. Press Ctrl+Alt+Space.",
     "Pette does NOT hide/flash (no screenshot). Bubble names dashboard.py and offers “Ask <agent>”."),
    ("Investigate", "Agent result", "Click “Ask <agent>”.",
     "Result panel: root cause + diff for dashboard.py line 12. Fixture files unchanged (panel shows hash check)."),
    ("Investigate", "Copy task", "Tray → Coding agent: Copy task, investigate again, approve.",
     "Clipboard holds a task containing the whole line-numbered dashboard.py."),
    ("Investigate", "No file", "Panel: “No file”, then Ctrl+Alt+Space.",
     "“No active file from VS Code…”; low-confidence proposal, nothing crashes."),
    ("Investigate", "Voice route", "With the buggy file, ask “why is this failing, ask Claude”.",
     "Goes to the investigate proposal (not a message draft)."),
    ("Proactive", "Offer", "Tick “Fast proactive”, panel: “Buggy file + error”, wait ~5 s.",
     "“Looks like you're stuck” offer with Investigate. Investigate → analysis proposal."),
    ("Proactive", "No nag", "Dismiss the offer; keep the same error.",
     "No repeat offer for that error (cooldown). “Reset offers” allows it again."),
    ("Proactive", "Defer while away", "Reset offers, Face away, set the buggy file, wait, then Face back.",
     "No offer while away; it appears shortly after Face back."),
    ("Webcam", "Away / blink", "Face away, then Face back; Eyes closed, Eyes open.",
     "Pette dozes while away and blinks on eyes closed; returns to idle."),
    ("Webcam", "Nudge", "Draft a message, Face away, Face back.", "Pending draft re-titled “Still …?”."),
    ("Webcam", "Pinch rules", "Pinch when idle; pinch on a message draft; pinch on an agent proposal.",
     "Idle → ask opens. Draft → discarded and re-dictate. Agent proposal → untouched. Never sends."),
    ("Control", "Pause", "Tray → Pause, press both hotkeys, Pinch; then unpause.",
     "Paused: bubble cleared, nothing reacts. Unpaused: works again."),
    ("Control", "Focus gate", "Switch to a browser, then back to VS Code.",
     "Pette hides outside VS Code/terminals and reappears; hotkeys ignored while hidden."),
    ("Memory", "Learned platform", "Send 2 Slack messages to Sarah, then “message Sarah saying hi” (no platform).",
     "Draft is “Slack (usual) to Sarah”. Tray → What Pette learned mentions it."),
    ("Memory", "Clear", "Tray → Clear Pette memory, repeat the last step.", "No “(usual)”; back to defaults."),
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
    from pet import context, voice
    from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
    from pet.watcher import InterventionPolicy

    main_args = app_main.parse_args(rest)
    ctx = app_main.build(main_args)
    pipeline, watcher, overlay = ctx.pipeline, ctx.watcher, ctx.overlay
    hashes_before = fixture_hashes()

    # --- panel --------------------------------------------------------------------------
    panel = QWidget(None, Qt.Window)
    panel.setWindowTitle("Pette sandbox · manual tests")
    panel.resize(1180, 820)
    root = QVBoxLayout(panel)

    brain = "DEMO (offline keyword drafter)" if main_args.demo else f"OpenAI {os.environ.get('OPENAI_MODEL') or 'gpt-4o'}"
    voice_on = not (main_args.demo or main_args.no_voice)
    status = QLabel()
    status.setWordWrap(True)
    status.setTextInteractionFlags(Qt.TextSelectableByMouse)

    hooks = '<span style="color:#A14B3A">LIVE (.env)</span>' if inbox is None else f"sandbox inbox {inbox.url}"

    def refresh_status():
        af = context.read_active_file()
        file_state = f"{Path(af.path).name} ({len(af.errors)} error(s))" if af else "no file"
        status.setText(
            f"<b>Brain:</b> {brain} &nbsp; <b>Voice:</b> {'on' if voice_on else 'typed'} &nbsp; "
            f"<b>Agent:</b> {pipeline.agent.label} &nbsp; <b>Hotkeys:</b> {voice.HOTKEY} / {voice.INVESTIGATE_HOTKEY}<br>"
            f"<b>Webhooks:</b> {hooks}"
            f" &nbsp; <b>IDE:</b> {'real VS Code extension' if args.real_ide else 'fake snapshot'} → {file_state}"
            f" &nbsp; <b>Busy:</b> {pipeline.busy} &nbsp; <b>Paused:</b> {overlay.paused}"
            f" &nbsp; <b>Fixture unchanged:</b> {fixture_hashes() == hashes_before}<br>"
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

    # fake VS Code snapshot
    snap = Path(os.environ.get("FLOWSTATE_ACTIVE_FILE", ""))

    def write_snapshot(errors: bool | None):
        if errors is None:
            snap.unlink(missing_ok=True)
        else:
            diags = [{"line": 12, "severity": "error", "source": "pytest", "code": "",
                      "message": "TypeError: 'NoneType' object is not subscriptable"}] if errors else []
            snap.write_text(json.dumps({
                "path": str(FIXTURE), "workspace": str(FIXTURE.parent), "language": "python",
                "content": FIXTURE.read_text(encoding="utf-8"), "selection": "", "line": 12,
                "diagnostics": diags, "updatedAt": dt.datetime.now().isoformat()}), encoding="utf-8")
        refresh_status()

    ide = group("Fake VS Code (active file)", 0, 0, [
        ("Buggy file + error", lambda: write_snapshot(True)),
        ("Same file, no errors", lambda: write_snapshot(False)),
        ("No file", lambda: write_snapshot(None)),
    ])
    if args.real_ide:
        ide.setEnabled(False)
        ide.setTitle("Fake VS Code (off: --real-ide)")
    else:
        write_snapshot(None)

    group("Simulated webcam events", 1, 0, [
        (n, (lambda ev=ev: watcher.on_vision_event(ev)))
        for n, ev in (("Face away", FACE_AWAY), ("Face back", FACE_BACK), ("Eyes closed", EYES_CLOSED),
                      ("Eyes open", EYES_OPEN), ("Pinch", PINCH))
    ])

    fast = QCheckBox("Fast proactive (3 s)")

    def reset_offers():
        watcher.policy = InterventionPolicy()
        if fast.isChecked():
            watcher.policy.DIAG_PERSIST_S = 3
        watcher._timer.setInterval(1000 if fast.isChecked() else 5000)

    fast.toggled.connect(lambda _: reset_offers())
    group("Triggers", 2, 0, [
        ("Ask (hotkey)", pipeline.hotkey_pressed.emit),
        ("Investigate (hotkey)", pipeline.investigate_hotkey.emit),
        ("Pause / resume", lambda: overlay.set_paused(not overlay.paused)),
        (fast, None),
        ("Reset offers", reset_offers),
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
                 f"- Brain: {brain}; voice: {'on' if voice_on else 'typed'}; agent: {pipeline.agent.label}",
                 f"- Webhooks: {'LIVE' if args.live_webhooks else 'sandbox inbox'}; "
                 f"IDE: {'real extension' if args.real_ide else 'fake snapshot'}",
                 f"- Fixture files unchanged by agents: {fixture_hashes() == hashes_before}",
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
