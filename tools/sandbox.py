"""Manual test sandbox: the real Pette app plus a control panel, around a scripted fake SOMA.

    py tools/sandbox.py                      # fake SOMA you drive from the panel
    py tools/sandbox.py --real-soma EP.json  # the real SOMA (start SOMA/scripts/serve.py with --endpoint-file EP.json)

The app is built by main.build(), so hotkeys, tray, focus gate, watcher and the SOMA hub are the real ones.
The fake SOMA is a token-protected localhost server with the same endpoints as SOMA/soma/api/local_api.py;
the panel queues suggestions on it and shows every request Pette makes (feedback, look-now, pause, folders).
Pette's memory db and the files a suggestion opens live in a temp folder, deleted on exit.
"""
import argparse
import datetime as dt
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "test_results" / "MANUAL_TEST.md"

# (area, scenario, how, expected)
SCENARIOS = [
    ("Launch", "Startup", "Start the sandbox.",
     "Pette, the camera orb, the world-model panel; bubble “Ready · SOMA”."),
    ("Cards", "Open a file", "Panel: “Open file at line”.",
     "Glass card “SOMA suggests · Open file”, Open / Not now, a draining bar. Open → the temp file opens "
     "(VS Code at line 7 if `code` is on PATH, else selected in Explorer); orb “Done ✓”."),
    ("Cards", "Your files", "Panel: “Your files (memory)”, then Open.", "The indexed note opens; nothing else runs."),
    ("Cards", "Copy", "Panel: “Review changes”, then Copy.", "The text is on the clipboard; bubble says Copied."),
    ("Cards", "Not now", "Queue a card, click Not now.", "Card closes, orb “Not now”, log shows feedback rejected."),
    ("Cards", "Ignore", "Queue a card, wait 15 s.", "Bar drains, card closes, log shows feedback ignored."),
    ("Cards", "Pinch", "Queue a card, panel: Pinch.", "Same as Open/Copy, once."),
    ("Cards", "Muted kind", "Answer Not now to “Write a test” 4 times, then queue it again.",
     "It no longer shows; world panel lists it as muted with the reason."),
    ("Look now", "Click Pette", "Click Pette (or Ctrl+Shift+Space).",
     "“Looking at your screen…”, log shows trigger; the queued (or next) card appears."),
    ("Look now", "Nothing found", "Tick “Look finds nothing”, click Pette, wait 12 s.",
     "“Nothing worth suggesting right now.”"),
    ("Folders", "Add", "Tray → Add a folder for SOMA…, pick any folder.",
     "“Indexing …” then “Indexed N file(s)”; tray → Indexed folders lists it; world panel shows it."),
    ("Folders", "Remove", "Tray → Indexed folders → Remove.", "“SOMA forgot N file(s)”; the list empties."),
    ("Control", "Toggle", "Ctrl+Alt+S twice.", "“Suggestions off” (log: pause), then on (log: resume); no cards while off."),
    ("Control", "Pause", "Tray → Pause all, queue a card, unpause.",
     "No card while paused, camera off; log: pause/resume."),
    ("Control", "Focus gate", "Tray → Show only in VS Code / Terminal; switch to a browser.",
     "Pette hides; cards wait; back in VS Code they show again."),
    ("Control", "Offline", "Click “Stop fake SOMA”, wait ~20 s.", "Tray notification; orb “SOMA offline”."),
    ("Orb", "Hide camera", "Ctrl+Alt+C twice.", "The picture in the orb becomes “camera hidden” and returns; Pette stays."),
    ("Orb", "Click / drag", "Click the orb; then drag it somewhere else.", "Click = look now; drag moves Pette (no look)."),
    ("Bubble", "Bottom of the screen", "Queue any card.",
     "The glass speech bubble appears at the bottom centre of the screen (tail pointing down), not over Pette."),
    ("Bubble", "Toggle", "Ctrl+Alt+B (or tray → Speech bubble), queue a card, Ctrl+Alt+B again.",
     "Off: bubble gone, the card doesn't show (tray note). On: the next card shows."),
    ("Gestures", "Next / previous tab", "Panel: “Point right (3 s)”, switch to a browser with several tabs.",
     "After 3 s the browser goes to the next tab (log: ctrl+tab); “Point left” goes back."),
    ("Gestures", "Scroll", "Panel: “Pinch-drag up (3 s)”, put the mouse over a long page.",
     "The page scrolls down (log: wheel -N); “Pinch-drag down” scrolls up."),
    ("Gestures", "Not outside a browser", "Use the same buttons with VS Code in front.", "Nothing is typed or scrolled."),
    ("Gestures", "Real camera", "With the camera on: point left/right, pinch and move your hand.",
     "The orb chip shows “point left/right” / “pinch”; tabs switch and the page scrolls in a browser."),
    ("World model", "Hide panel", "Ctrl+Alt+W twice.", "Panel hides and docks again."),
    ("World model", "Facts only", "Answer a few cards, open the panel.",
     "Counts per kind and folder names only; no suggestion text or file names."),
    ("Webcam", "Away / blink", "Face away / Face back with a card up; Eyes closed.",
     "“While you were away · …”; Pette blinks on eyes closed."),
]

SAMPLES = {
    "Open file at line": {"action": "OPEN_RELEVANT_FILE", "template": "", "text": "Open calc.py at line 7",
                          "file": "calc.py", "line": 7},
    "Your files (memory)": {"action": "COMPARE_WITH_PRIOR_WORK", "template": "",
                            "text": 'Compare with your earlier document "Retry notes"', "memory_source": "retries.md"},
    "Review changes": {"action": "SUGGEST_NEXT_STEP", "template": "review_changes",
                       "text": 'Review your changes to "calc.py" before you commit'},
    "Write a test": {"action": "SUGGEST_NEXT_STEP", "template": "write_tests", "text": 'Write a test for the code in "calc.py"'},
    "Explain error": {"action": "EXPLAIN_ERROR", "template": "",
                      "text": 'Explain the error "ZeroDivisionError: division by zero"'},
}


class FakeSoma:
    """SOMA's local API, scripted. `queue(sample)` makes the next /suggestion return it (once)."""

    def __init__(self, files: Path, on_request):
        self.files, self.on_request = files, on_request
        self.token = secrets.token_urlsafe(16)
        self.current: dict = {"available": True, "silent": True}
        self.paused = False
        self.empty_look = False
        self.folders: list[dict] = []
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _ok(self):
                if self.headers.get("Authorization") != f"Bearer {fake.token}":
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self):
                if not self._ok():
                    return
                if self.path == "/suggestion":
                    self._send(200, fake.next_suggestion())
                elif self.path == "/folders":
                    self._send(200, {"folders": fake.folders, "indexing": None,
                                     "last_report": fake.last_report})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._ok():
                    return
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.on_request(self.path, body)
                if self.path in ("/pause", "/resume"):
                    fake.paused = self.path == "/pause"
                    self._send(200, {"paused": fake.paused})
                elif self.path == "/trigger":
                    if not fake.paused and not fake.empty_look:
                        fake.queue(SAMPLES["Review changes"])
                    self._send(200, {"triggered": not fake.paused})
                elif self.path == "/feedback":
                    self._send(200, {"ok": True})
                elif self.path == "/folders":
                    p = Path(body.get("path", ""))
                    if not p.is_dir():
                        self._send(400, {"error": f"not a folder: {p}"})
                        return
                    n_files = sum(1 for f in p.rglob("*") if f.is_file())
                    fake.folders = [f for f in fake.folders if f["path"] != str(p)] + \
                        [{"path": str(p), "memories": n_files}]
                    fake.last_report = {"path": str(p), "added": n_files}
                    self._send(200, {"path": str(p), "indexing": True})
                elif self.path == "/folders/remove":
                    gone = [f for f in fake.folders if f["path"] == body.get("path")]
                    fake.folders = [f for f in fake.folders if f["path"] != body.get("path")]
                    self._send(200, {"path": body.get("path"), "forgotten": sum(f["memories"] for f in gone)})
                else:
                    self._send(404, {"error": "not found"})

            def log_message(self, *a):
                pass

        self.last_report: dict = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def queue(self, sample: dict):
        s = {"available": True, "silent": False, "valid": True, "id": uuid.uuid4().hex, "evidence": [],
             "file": None, "line": None, "memory_source": None, **sample}
        for key in ("file", "memory_source"):
            if s.get(key):
                s[key] = str(self.files / s[key])
        with self._lock:
            self._pending.append(s)

    def next_suggestion(self) -> dict:
        with self._lock:
            if self._pending and not self.paused:
                self.current = self._pending.pop(0)
            return self.current

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real-soma", metavar="ENDPOINT_FILE", help="use the real SOMA service instead of the fake")
    return ap.parse_known_args()


def main() -> int:
    args, rest = parse()
    tmp = Path(tempfile.mkdtemp(prefix="pette-sandbox-"))
    files = tmp / "files"
    files.mkdir()
    (files / "calc.py").write_text("def total(items):\n    count = 0\n    for i in items:\n        count += i\n"
                                   "    return count\n\n\nprint(total([]) / len([]))\n", encoding="utf-8")
    (files / "retries.md").write_text("# Retry notes\nBack off exponentially; cap at 30 s.\n", encoding="utf-8")
    os.environ["FLOWSTATE_DB"] = str(tmp / "pette_cache.sqlite3")

    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                                   QHeaderView, QLabel, QPlainTextEdit, QPushButton, QTableWidget,
                                   QTableWidgetItem, QVBoxLayout, QWidget)

    class Bridge(QObject):
        request = Signal(str, object)   # queued onto the main thread from the HTTP thread

    app = QApplication.instance() or QApplication(sys.argv)
    bridge = Bridge()
    fake = None
    if args.real_soma:
        endpoint = Path(args.real_soma)
    else:
        fake = FakeSoma(files, bridge.request.emit)
        endpoint = tmp / "endpoint.json"
        endpoint.write_text(json.dumps({"url": fake.url, "token": fake.token}), encoding="utf-8")
    os.environ["PETTE_SOMA_ENDPOINT"] = str(endpoint)

    import main as app_main
    from pet import hotkeys
    from pet.vision import (DRAG_END, DRAG_START, EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH, POINT_LEFT,
                            POINT_RIGHT)

    ctx = app_main.build(app_main.parse_args(["--soma", str(endpoint), *rest]))
    watcher, overlay, suggester = ctx.watcher, ctx.overlay, ctx.suggester

    # --- panel --------------------------------------------------------------------------
    panel = QWidget(None, Qt.Window)
    panel.setWindowTitle("Pette sandbox · manual tests")
    panel.resize(1180, 820)
    root = QVBoxLayout(panel)
    status = QLabel()
    status.setWordWrap(True)
    status.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def refresh_status():
        status.setText(
            f"<b>SOMA:</b> {'fake ' + fake.url if fake else 'real (' + str(endpoint) + ')'} &nbsp; "
            f"<b>State:</b> {suggester.state} &nbsp; <b>Suggestions:</b> {'on' if suggester.enabled else 'off'} &nbsp; "
            f"<b>Paused:</b> {overlay.paused} &nbsp; <b>Shown on screen:</b> {overlay.context_allowed}<br>"
            f"<b>Hotkeys:</b> look {hotkeys.LOOK_HOTKEY}, suggestions {hotkeys.SOMA_HOTKEY}, camera "
            f"{hotkeys.CAMERA_HOTKEY}, world {hotkeys.WORLD_HOTKEY} &nbsp; <b>Temp state:</b> {tmp}")

    root.addWidget(status)
    QTimer(panel, interval=1000, timeout=refresh_status).start()
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

    cards = group("Queue a SOMA suggestion (fake SOMA)", 0, 0,
                  [(name, (lambda s=sample: fake.queue(s))) for name, sample in SAMPLES.items()])
    empty = QCheckBox("Look finds nothing")
    if fake:
        empty.toggled.connect(lambda on: setattr(fake, "empty_look", on))
    stop_fake = QPushButton("Stop fake SOMA")
    stop_fake.clicked.connect(lambda: (fake.stop(), stop_fake.setEnabled(False)))
    soma_box = group("Fake SOMA", 1, 0, [(empty, None), (stop_fake, None)])
    if not fake:
        cards.setEnabled(False)
        soma_box.setEnabled(False)
        cards.setTitle("Queue a suggestion (off: --real-soma)")

    group("Simulated webcam events", 2, 0, [
        (n, (lambda ev=ev: watcher.on_vision_event(ev)))
        for n, ev in (("Face away", FACE_AWAY), ("Face back", FACE_BACK), ("Eyes closed", EYES_CLOSED),
                      ("Eyes open", EYES_OPEN), ("Pinch", PINCH))
    ])
    browser = ctx.browser
    real_keys, real_wheel = browser._send_keys, browser._wheel

    def logged_keys(combo):
        log.appendPlainText(f"[{dt.datetime.now():%H:%M:%S}] gesture → keys {combo}")
        real_keys(combo)

    def logged_wheel(notches):
        log.appendPlainText(f"[{dt.datetime.now():%H:%M:%S}] gesture → wheel {notches:+d}")
        real_wheel(notches)

    browser._send_keys, browser._wheel = logged_keys, logged_wheel

    def later(fn):
        return lambda: QTimer.singleShot(3000, fn)    # time to bring a browser to the front

    def drag(direction):
        watcher.on_vision_event(DRAG_START)
        for _ in range(5):
            watcher.on_scroll(0.04 * direction)
        watcher.on_vision_event(DRAG_END)

    group("Simulated gestures (fire after 3 s: switch to a browser)", 4, 0, [
        ("Point left (3 s)", later(lambda: watcher.on_vision_event(POINT_LEFT))),
        ("Point right (3 s)", later(lambda: watcher.on_vision_event(POINT_RIGHT))),
        ("Pinch-drag up (3 s)", later(lambda: drag(-1))),
        ("Pinch-drag down (3 s)", later(lambda: drag(+1))),
    ])
    group("Triggers", 3, 0, [
        ("Click Pette (look now)", overlay.orb.clicked.emit),
        ("Speech bubble on/off", overlay.toggle_bubble),
        ("Pause / resume", lambda: overlay.set_paused(not overlay.paused)),
        ("Suggestions on/off", suggester.toggle),
        ("World model window", ctx.worldwin.toggle),
        ("Camera on/off", lambda: watcher.set_camera(watcher.camera is None)),
    ])

    log = QPlainTextEdit(readOnly=True)
    log.setPlaceholderText("Every request Pette makes to SOMA lands here.")
    box = QGroupBox("Requests to SOMA")
    QVBoxLayout(box).addWidget(log)
    grid.addWidget(box, 0, 1, 5, 1)
    grid.setColumnStretch(0, 3)
    grid.setColumnStretch(1, 2)
    bridge.request.connect(lambda path, body: log.appendPlainText(
        f"[{dt.datetime.now():%H:%M:%S}] POST {path} {json.dumps(body) if body else ''}"))

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
                 f"- SOMA: {'fake (scripted)' if fake else 'real'}",
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
        if fake and stop_fake.isEnabled():
            fake.stop()
        time.sleep(0.1)
        shutil.rmtree(tmp, ignore_errors=True)

    app.aboutToQuit.connect(cleanup)
    panel.closeEvent = lambda e: (e.accept(), app.quit())   # closing the panel ends the session
    refresh_status()
    panel.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
