"""End-to-end: Pette assembled like main.py (minus the tray) around a fake SOMA, driven through a session."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from db import Store, WorldModel
from pet import soma
from pet.focus import DEFAULT_APPS, FocusGate
from pet.ui import Overlay
from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
from pet.watcher import Watcher
from pet.worldview import WorldWindow
from tests.test_soma import FakeClient, sug

ROOT = Path(__file__).resolve().parent.parent
SECRET_TEXT = 'Review your changes to "payroll_secret.py" before you commit'


def test_full_session(qapp, tmp_path):
    store = Store(tmp_path / "cache.sqlite3")
    world = WorldModel(store)
    overlay = Overlay(tray=False)
    client, opened, copied = FakeClient(), [], []
    s = soma.Suggester(overlay, client, memory=store, world=world, spawn=lambda f: f(), clipboard=copied.append,
                       open_file=lambda p, line=None: opened.append((p, line)) or "Opened.")
    watcher = Watcher(overlay, memory=store, world=world)
    win = WorldWindow(overlay, world, s, watcher, web=False)
    fg = {"exe": "Code.exe"}
    gate = FocusGate(overlay, info=lambda: (fg["exe"], "Win", 99), apps=set(DEFAULT_APPS))
    gate.enabled = True
    overlay.show()
    win.show_near()

    def deliver(x):
        client.queue.append(x)
        s.poll_once()
        qapp.processEvents()

    # 1. Index a folder from the tray; SOMA reports it done.
    s.add_folder("C:/notes")
    client.folder_data = {"folders": [{"path": "C:/notes", "memories": 7}], "indexing": None,
                          "last_report": {"path": "C:/notes", "added": 7}}
    s.refresh_folders()
    assert "Indexed 7 file(s)" in overlay.bubble.body.text()

    # 2. In VS Code: a suggestion to open a file; pinch does it.
    gate.poll()
    deliver(sug("s1", template="", action="OPEN_RELEVANT_FILE", text="Open calc.py at line 40",
                file="C:/notes/calc.py", line=40))
    assert overlay.bubble.proposal.title.endswith("Open file") and overlay.orb.label() == "Suggestion"
    watcher.on_vision_event(PINCH)
    assert opened == [("C:/notes/calc.py", 40)] and overlay.orb.label() == "Done ✓"

    # 3. Switch to an app outside the gate: Pette hides, cards and clicks wait.
    fg["exe"] = "chrome.exe"
    gate.poll()
    assert overlay.isHidden()
    deliver(sug("s2", text=SECRET_TEXT))
    assert overlay.bubble.proposal is None
    overlay.look_requested.emit()
    assert ("trigger",) not in client.calls

    # 4. Back in the terminal: walk away while a card is up, come back to a nudge, Not now.
    fg["exe"] = "WindowsTerminal.exe"
    gate.poll()
    deliver(sug("s3", text=SECRET_TEXT))
    watcher.on_vision_event(FACE_AWAY)
    watcher.on_vision_event(FACE_BACK)
    assert overlay.bubble.title.text().startswith("While you were away")
    overlay.bubble.cancel.click()
    assert copied == [] and overlay.orb.label() == "Not now"

    # 5. Eyes closed: Pette mirrors (only while idle).
    overlay.dismiss()
    watcher.on_vision_event(EYES_CLOSED)
    assert overlay.pet.state == "blink"
    watcher.on_vision_event(EYES_OPEN)

    # 6. Pause: SOMA pauses, a pinch does nothing, no card shows.
    overlay.set_paused(True)
    assert ("pause",) in client.calls
    watcher.on_vision_event(PINCH)
    deliver(sug("s4"))
    assert overlay.bubble.proposal is None
    overlay.set_paused(False)

    # 7. The world panel shows facts only: kinds and outcomes, folder names, never suggestion text.
    data = win.data()
    assert data["soma"] == {"shown": 2, "accepted": 1, "dismissed": 1, "ignored": 0}
    assert data["folders"] == [{"name": "notes", "path": "C:/notes", "memories": 7}]
    assert data["live"]["suggestions"] is True and world.presence_today()["pinches"] == 1
    dump = " ".join(str(r) for t in ("events", "offers") for r in store.query(f"SELECT * FROM {t}"))
    assert "payroll" not in dump and "calc.py" not in dump
    assert "payroll" not in str(data) and "calc.py" not in str(data)
    store.close()


@pytest.mark.parametrize("flags", [[], ["--focus-only", "--no-world"]])
def test_main_launches_and_stays_up_without_soma(flags, tmp_path):
    """The real entry point starts without SOMA running (it connects lazily) and keeps running."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "FLOWSTATE_DB": str(tmp_path / "c.sqlite3"),
           soma.ENDPOINT_VAR: str(tmp_path / "no-endpoint.json")}
    proc = subprocess.Popen([sys.executable, str(ROOT / "main.py"), *flags], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=8)                       # still running after 8 s = healthy
    finally:
        proc.kill()
        _, err = proc.communicate()
    assert "Traceback" not in err, err
    assert (tmp_path / "c.sqlite3").exists()           # db cache created
