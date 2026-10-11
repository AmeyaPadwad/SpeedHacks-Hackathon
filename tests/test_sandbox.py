import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pet import dispatch
from pet.brain import Draft
from pet.dispatch import DispatchError

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sandbox", ROOT / "tools" / "sandbox.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)


@pytest.fixture
def inbox(monkeypatch):
    box, posts = sandbox.Inbox(), []
    box.on_post = lambda platform, payload, status: posts.append((platform, payload, status))
    monkeypatch.setenv("SLACK_WEBHOOK_URL", box.url + "/slack")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", box.url + "/discord")
    yield box, posts
    box.server.shutdown()


def test_inbox_captures_real_dispatch(inbox):
    box, posts = inbox
    dispatch.dispatch(Draft("slack", "Sarah", "", "build is green"), sandbox.CONTACTS)
    dispatch.dispatch(Draft("discord", "alex", "", "standup at 3"), sandbox.CONTACTS)
    assert posts[0] == ("slack", {"text": "<@U0SARAH> build is green"}, 200)
    platform, payload, status = posts[1]
    assert platform == "discord" and status == 204
    assert payload["allowed_mentions"] == {"users": ["222222222222222222"]}


def test_inbox_failure_mode_surfaces_dispatch_error(inbox):
    box, posts = inbox
    box.fail = True
    with pytest.raises(DispatchError, match="HTTP 500"):
        dispatch.dispatch(Draft("slack", "Sarah", "", "hi"), sandbox.CONTACTS)
    assert posts[0][2] == 500


def test_every_scenario_is_complete():
    assert len({(a, n) for a, n, _, _ in sandbox.SCENARIOS}) == len(sandbox.SCENARIOS)
    assert all(all(s.strip() for s in row) for row in sandbox.SCENARIOS)


def test_sandbox_launches_and_stays_up():
    """The sandbox builds the real app + panel offscreen and keeps running; the real db is untouched."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    env.pop("FLOWSTATE_DB", None)
    real_db = ROOT / "db" / "pette_cache.sqlite3"
    before = real_db.stat().st_mtime if real_db.exists() else None
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "sandbox.py"), "--demo"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=6)
    finally:
        proc.kill()
        _, err = proc.communicate()
    assert "Traceback" not in err, err
    assert (real_db.stat().st_mtime if real_db.exists() else None) == before
