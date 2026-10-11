"""End-to-end: Pette assembled like main.py (minus the tray), driven through a realistic session."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from db import Store, WorldModel
from pet.agents import AgentResult
from pet.analyst import LocalAnalyst
from pet.brain import OfflineBrain
from pet.context import ActiveFile, Context, Diagnostic
from pet.focus import DEFAULT_APPS, FocusGate
from pet.pipeline import Pipeline
from pet.ui import Overlay
from pet.vision import EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH
from pet.watcher import Watcher

ROOT = Path(__file__).resolve().parent.parent
AF = ActiveFile(str(ROOT / "fixtures/buggy_app/dashboard.py"), "python", "x\n" * 20,
                workspace=str(ROOT / "fixtures/buggy_app"),
                diagnostics=(Diagnostic(12, "error", "TypeError: 'NoneType' object is not subscriptable", "pytest"),))


class Agent:
    name, label = "claude", "Claude Code"

    def __init__(self):
        self.tasks = []

    def run(self, task):
        self.tasks.append(task)
        return AgentResult("claude", "--- a/dashboard.py\n+++ b/dashboard.py\n+    if user is None: ...")


def test_full_session(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    monkeypatch.setattr("pet.watcher.InterventionPolicy.DIAG_PERSIST_S", 0)
    store = Store(tmp_path / "cache.sqlite3")
    world = WorldModel(store)
    overlay = Overlay(tray=False)
    agent, sent, shown = Agent(), [], []
    pipeline = Pipeline(overlay, OfflineBrain(), analyst=LocalAnalyst(), capture=lambda: Context(b"jpg", AF), read_file=lambda: AF,
                        send=lambda d: sent.append(d) or f"Sent to {d.recipient}.", configured=lambda _: True,
                        pick_agent=lambda name=None: agent, show_result=lambda t, x: shown.append(x),
                        use_voice=False, spawn=lambda f: f(), capture_delay_ms=0, memory=store, world=world)
    watcher = Watcher(overlay, lambda: pipeline.busy, read_file=lambda: AF, memory=store, world=world)
    watcher.offer.connect(pipeline.offer_investigation)
    fg = {"exe": "Code.exe"}
    gate = FocusGate(overlay, info=lambda: (fg["exe"], "Win", 99), apps=set(DEFAULT_APPS))
    gate.enabled = True
    overlay.show()

    def say_command(text):
        overlay.bubble.input.setText(text)
        overlay.bubble.primary.click()

    # 1. In VS Code: pinch to talk, send a Slack message to Sarah.
    gate.poll()
    assert overlay.isVisible()
    watcher.on_vision_event(PINCH)
    say_command("Slack Sarah asking her to review the dashboard fix")
    assert overlay.bubble.proposal.title.endswith("Slack to Sarah")
    overlay.bubble.primary.click()
    assert [d.recipient for d in sent] == ["Sarah"]

    # 2. Switch to the browser: Pette hides, hotkey does nothing.
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

    # 4. Walk away with the draft pending; come back to a nudge; pinch to re-dictate; send.
    watcher.on_vision_event(FACE_AWAY)
    watcher.poll()                                   # proactive offer is deferred while away
    assert "stuck" not in overlay.bubble.title.text()
    watcher.on_vision_event(FACE_BACK)
    assert overlay.bubble.title.text().startswith("Still send?")
    watcher.on_vision_event(PINCH)
    say_command("Slack Sarah saying all tests pass now")
    overlay.bubble.primary.click()
    assert sent[-1].message == "Hi Sarah, all tests pass now in dashboard.py. Thanks!" and len(sent) == 2

    # 5. Eyes closed: Pette mirrors (only while idle).
    overlay.dismiss()
    watcher.on_vision_event(EYES_CLOSED)
    assert overlay.pet.state == "blink"
    watcher.on_vision_event(EYES_OPEN)

    # 6. Persistent IDE error -> proactive offer -> investigate -> agent proposal -> read-only result.
    watcher.poll()
    assert overlay.bubble.proposal.action_label == "Investigate"
    overlay.bubble.primary.click()
    assert overlay.bubble.proposal.action_label == "Ask Claude Code"
    overlay.bubble.primary.click()
    task = agent.tasks[0]
    assert task.file_path == AF.path and task.file_text and "NoneType" in task.diagnostics[0]
    assert shown and shown[0].startswith("--- a/dashboard.py")

    # 7. Pause: everything stops; camera-driven pinch is ignored.
    overlay.set_paused(True)
    watcher.on_vision_event(PINCH)
    assert not pipeline.busy and not overlay.bubble.isVisible()
    overlay.set_paused(False)

    # 8. The world model learned from the session; no message text was stored.
    assert world.preferred_platform("sarah") == "slack"
    assert world.presence_today()["pinches"] == 2
    assert store.query("SELECT outcome FROM offers ORDER BY ts") == [("shown",), ("accepted",), ("accepted",)]
    dump = " ".join(str(r) for t in ("events", "messages", "offers") for r in store.query(f"SELECT * FROM {t}"))
    assert "tests pass" not in dump and "review" not in dump
    store.close()


@pytest.mark.parametrize("flags", [["--demo"], ["--demo", "--always-show"]])
def test_main_launches_and_stays_up(flags, tmp_path):
    """The real entry point starts, builds the tray/watchers/db, and keeps running."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "FLOWSTATE_DB": str(tmp_path / "c.sqlite3")}
    proc = subprocess.Popen([sys.executable, str(ROOT / "main.py"), *flags], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=6)                       # still running after 6 s = healthy
    finally:
        proc.kill()
        _, err = proc.communicate()
    assert "Traceback" not in err, err
    assert (tmp_path / "c.sqlite3").exists()           # db cache created
