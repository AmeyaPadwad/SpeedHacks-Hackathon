"""The sandbox's fake SOMA speaks the same API as SOMA, checked through Pette's real client."""
import importlib.util
import json
from pathlib import Path

import pytest

from pet import soma

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sandbox", ROOT / "tools" / "sandbox.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)


@pytest.fixture
def fake(tmp_path):
    requests = []
    f = sandbox.FakeSoma(tmp_path, lambda path, body: requests.append((path, body)))
    ep = tmp_path / "ep.json"
    ep.write_text(json.dumps({"url": f.url, "token": f.token}))
    yield f, soma.SomaClient.from_endpoint(ep), requests
    f.stop()


def test_queued_suggestion_is_served_with_absolute_paths(fake, tmp_path):
    f, client, _ = fake
    assert client.suggestion()["silent"] is True
    f.queue(sandbox.SAMPLES["Open file at line"])
    s = client.suggestion()
    assert s["file"] == str(tmp_path / "calc.py") and s["line"] == 7 and soma.opener(s) == (s["file"], 7)
    assert client.suggestion()["id"] == s["id"]                    # stays current until the next one


def test_trigger_pause_feedback_and_folders(fake, tmp_path):
    f, client, requests = fake
    assert client.trigger() == {"triggered": True}
    assert client.suggestion()["template"] == "review_changes"
    client.pause()
    assert client.trigger() == {"triggered": False}
    client.feedback("x", "rejected")
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("a")
    client.add_folder(str(notes))
    assert client.folders()["folders"] == [{"path": str(notes), "memories": 1}]
    assert client.remove_folder(str(notes))["forgotten"] == 1
    with pytest.raises(soma.SomaError, match="not a folder"):
        client.add_folder(str(tmp_path / "missing"))
    assert [p for p, _ in requests][:4] == ["/trigger", "/pause", "/trigger", "/feedback"]


def test_wrong_token_is_refused(fake):
    f, _, _ = fake
    with pytest.raises(soma.SomaError):
        soma.SomaClient(f.url, "wrong").suggestion()


def test_scenarios_are_well_formed():
    assert all(len(row) == 4 and all(row) for row in sandbox.SCENARIOS)
    assert {"Cards", "Look now", "Folders", "Control", "Bubble", "Gestures"} <= {area for area, *_ in sandbox.SCENARIOS}
