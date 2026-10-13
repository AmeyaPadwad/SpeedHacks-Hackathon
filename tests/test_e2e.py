"""End-to-end: Pette assembled like main.py (minus the tray), driven through a realistic session."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from db import Store, WorldModel
from pet.brain import OfflineBrain
from pet.context import ActiveFile, Context
from pet.focus import DEFAULT_APPS, FocusGate
from pet.pipeline import Pipeline
from pet.ui import Overlay
from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
from pet.watcher import Watcher
from pet.worldview import WorldWindow

ROOT = Path(__file__).resolve().parent.parent
AF = ActiveFile(str(ROOT / "fixtures/buggy_app/dashboard.py"), "python", "x\n" * 20,
                workspace=str(ROOT / "fixtures/buggy_app"))
DIFF = "--- a/dashboard.py\n+++ b/dashboard.py\n+    if user is None:\n+        return 'unknown user'"


def test_full_session(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    store = Store(tmp_path / "cache.sqlite3")
    world = WorldModel(store)
    overlay = Overlay(tray=False)
    sent = []
    pipeline = Pipeline(overlay, OfflineBrain(), capture=lambda: Context(AF, DIFF),
                        send=lambda d: sent.append(d) or f"Sent to {d.recipient}.", configured=lambda _: True,
                        use_voice=False, spawn=lambda f: f(), memory=store, world=world, send_delay_s=2)
    watcher = Watcher(overlay, memory=store, world=world)
    win = WorldWindow(overlay, world, pipeline, watcher, web=False)
    fg = {"exe": "Code.exe"}
    gate = FocusGate(overlay, info=lambda: (fg["exe"], "Win", 99), apps=set(DEFAULT_APPS))
    gate.enabled = True
    overlay.show()
    win.show_near()

    def say_command(text):
        overlay.bubble.input.setText(text)
        overlay.bubble.primary.click()

    def countdown():
        while pipeline._countdown.isActive():
            pipeline._tick()

    # 1. In VS Code: click to talk, Slack Sarah; the post carries the summary + the diff.
    gate.poll()
    assert overlay.isVisible()
    pipeline.trigger()
    say_command("Slack Sarah asking her to review the dashboard fix")
    assert overlay.bubble.proposal.title.endswith("Slack to Sarah") and overlay.orb.label() == "Sending in 2s"
    countdown()
    assert [d.recipient for d in sent] == ["Sarah"] and sent[0].code == DIFF and "2 line(s) added" in sent[0].message
    assert overlay.orb.label() == "Sent ✓"

    # 2. Switch to the browser: Pette hides, the hotkey does nothing.
    fg["exe"] = "chrome.exe"
    gate.poll()
    assert overlay.isHidden()
    pipeline.trigger()
    assert not pipeline.busy

    # 3. Back in the terminal: "message Sarah" (no platform) -> learned Slack.
    fg["exe"] = "WindowsTerminal.exe"
    gate.poll()
    pipeline.trigger()
    say_command("message Sarah saying tests pass now")
    assert "Slack (usual) to Sarah" in overlay.bubble.proposal.title

    # 4. Walk away mid-countdown; come back to a nudge; Cancel. Nothing sent.
    watcher.on_vision_event(FACE_AWAY)
    watcher.on_vision_event(FACE_BACK)
    assert overlay.bubble.title.text().startswith("Still sending?")
    overlay.bubble.cancel.click()
    countdown()
    assert len(sent) == 1 and overlay.orb.label() == "Not sent"

    # 5. Eyes closed: Pette mirrors (only while idle).
    overlay.dismiss()
    watcher.on_vision_event(EYES_CLOSED)
    assert overlay.pet.state == "blink"
    watcher.on_vision_event(EYES_OPEN)

    # 6. Pause: everything stops; camera-driven pinch is ignored.
    overlay.set_paused(True)
    watcher.on_vision_event(PINCH)
    assert not pipeline.busy and not overlay.bubble.isVisible()
    overlay.set_paused(False)

    # 7. The world model learned from the session and the window shows it; no message text was stored.
    assert world.preferred_platform("sarah") == "slack"
    assert world.presence_today()["pinches"] == 0              # no pinches while active; the paused one is not logged
    data = win.data()
    assert data["outcomes"]["sent"] == 1 and data["outcomes"]["cancelled"] == 1
    assert data["contacts"][0]["name"] == "Sarah" and data["live"]["state"] == "idle"
    assert any("Slack" in r["then"] for r in data["rules"])
    dump = " ".join(str(r) for t in ("events", "messages", "offers") for r in store.query(f"SELECT * FROM {t}"))
    assert "tests pass" not in dump and "review" not in dump and "user is None" not in dump
    assert "tests pass" not in str(data) and "user is None" not in str(data)
    store.close()


@pytest.mark.parametrize("flags", [["--demo"], ["--demo", "--always-show"]])
def test_main_launches_and_stays_up(flags, tmp_path):
    """The real entry point starts, builds the tray/watchers/db/world window, and keeps running."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "FLOWSTATE_DB": str(tmp_path / "c.sqlite3")}
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
