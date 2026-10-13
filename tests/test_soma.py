"""Pette as SOMA's UI: cards only when free, nothing happens without "Do it", feedback and facts flow back."""
import json

import pytest

from db import Store, WorldModel
from pet import actions, soma
from pet.ui import Overlay


def sug(sid="s1", template="review_changes", action="SUGGEST_NEXT_STEP", **kw):
    base = {"available": True, "silent": False, "valid": True, "id": sid, "action": action,
            "text": 'Review your changes to "app.py" before you commit', "evidence": ['editing "app.py"'],
            "template": template, "file": None, "line": None, "memory_source": None}
    return {**base, **kw}


class FakeClient:
    def __init__(self):
        self.queue, self.calls = [], []
        self.triggered = True
        self.folder_data = {"folders": [], "indexing": None, "last_report": {}}

    def suggestion(self):
        return self.queue.pop(0) if self.queue else {"available": True, "silent": True}

    def feedback(self, sid, outcome):
        self.calls.append(("feedback", sid, outcome))

    def trigger(self):
        self.calls.append(("trigger",))
        return {"triggered": self.triggered}

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def folders(self):
        return self.folder_data

    def add_folder(self, path):
        self.calls.append(("add_folder", path))
        if path == "bad":
            raise soma.SomaError("Refusing to index an entire home directory or drive root.")
        self.folder_data = {"folders": [{"path": path, "memories": 0}], "indexing": path, "last_report": {}}
        return {"path": path, "indexing": True}

    def remove_folder(self, path):
        self.calls.append(("remove_folder", path))
        self.folder_data = {"folders": [], "indexing": None, "last_report": {}}
        return {"path": path, "forgotten": 3}


@pytest.fixture
def rig(qapp, tmp_path):
    o = Overlay(tray=False)
    store = Store(tmp_path / "m.sqlite3")
    world = WorldModel(store)
    client, opened, copied, states = FakeClient(), [], [], []

    def open_file(path, line=None):
        opened.append((path, line))
        return f"Opened {path}"

    s = soma.Suggester(o, client, memory=store, world=world, spawn=lambda f: f(), clipboard=copied.append,
                       open_file=open_file, show_s=2)
    s.status.connect(lambda st, d: states.append(st))

    def deliver(*suggestions):
        client.queue.extend(suggestions)
        for _ in suggestions:
            s.poll_once()
        qapp.processEvents()

    r = type("Rig", (), {})()
    r.o, r.s, r.client, r.store, r.world, r.opened, r.copied, r.states, r.deliver = \
        o, s, client, store, world, opened, copied, states, deliver
    yield r
    store.close()


def test_card_then_copy(rig):
    rig.deliver(sug())
    b = rig.o.bubble
    assert b.proposal.title == "SOMA suggests · Next step"
    assert b.primary.text() == "Copy" and b.cancel.text() == "Not now" and b.progress == 1.0
    assert rig.copied == [] and rig.opened == [] and rig.o.orb.label() == "Suggestion"
    b.primary.click()
    assert rig.copied == ['Review your changes to "app.py" before you commit']
    assert ("feedback", "s1", "accepted") in rig.client.calls and rig.o.orb.label() == "Done ✓"
    assert rig.o.bubble.progress is None                            # "Done" message has no time-left bar
    assert rig.store.query("SELECT kind, outcome FROM offers ORDER BY rowid") == [
        ("soma:review_changes", "shown"), ("soma:review_changes", "accepted")]


def test_open_file_at_line_and_memory_source(rig):
    rig.deliver(sug("s1", template="", action="OPEN_RELEVANT_FILE", text="Open calc.py at line 40",
                    file="C:/proj/calc.py", line=40))
    assert rig.o.bubble.primary.text() == "Open" and rig.o.bubble.proposal.title.endswith("Open file")
    rig.o.bubble.primary.click()
    assert rig.opened == [("C:/proj/calc.py", 40)]
    rig.deliver(sug("s2", template="", action="COMPARE_WITH_PRIOR_WORK", text='Compare with "retries.md"',
                    memory_source="C:/notes/retries.md"))
    rig.o.pinch()                                                   # pinch = Do it
    assert rig.opened[-1] == ("C:/notes/retries.md", None) and rig.copied == []
    assert rig.store.query("SELECT kind FROM offers WHERE outcome='accepted' ORDER BY rowid") == [
        ("soma:open_relevant_file",), ("soma:compare_with_prior_work",)]


def test_missing_file_is_reported_not_raised(rig):
    def gone(path, line=None):
        raise actions.ActionError("calc.py isn't there any more.")
    rig.s._open_file = gone
    rig.deliver(sug(file="C:/proj/calc.py"))
    rig.o.bubble.primary.click()
    assert "isn't there" in rig.o.bubble.body.text() and rig.states[-1] == "failed"


def test_not_now_and_expiry(rig, qapp):
    rig.deliver(sug("s1"))
    rig.o.bubble.cancel.click()
    assert rig.o.bubble.proposal is None and rig.client.calls[-1] == ("feedback", "s1", "rejected")
    assert rig.states[-1] == "dismissed"
    rig.deliver(sug("s2"))
    for _ in range(8):                                              # 2 s at 250 ms ticks
        rig.s._count_down()
    assert rig.o.bubble.proposal is None and rig.client.calls[-1] == ("feedback", "s2", "ignored")
    assert rig.store.query("SELECT outcome FROM offers ORDER BY rowid") == [
        ("shown",), ("dismissed",), ("shown",), ("ignored",)]


def test_same_suggestion_once_and_not_while_busy_hidden_or_paused(rig):
    rig.deliver(sug("s1"))
    rig.o.bubble.cancel.click()
    rig.deliver(sug("s1"))
    assert rig.o.bubble.proposal is None                            # already handled
    rig.deliver(sug("s2", silent=True), sug("s3", valid=False), sug("s4", text=""))
    assert rig.o.bubble.proposal is None
    rig.o.context_allowed = False                                   # focus gate hid Pette
    rig.deliver(sug("s5"))
    assert rig.o.bubble.proposal is None
    rig.o.context_allowed = True
    rig.o.set_paused(True)
    rig.deliver(sug("s6"))
    assert rig.o.bubble.proposal is None and ("pause",) in rig.client.calls
    rig.o.set_paused(False)
    rig.deliver(sug("s7"))
    assert rig.o.bubble.proposal is not None and ("resume",) in rig.client.calls


def test_muted_kind_is_skipped_unless_you_asked(rig):
    for _ in range(4):
        rig.store.log_offer("soma:write_tests", "dismissed")
    rig.deliver(sug("s1", template="write_tests", text="Write a test for app.py"))
    assert rig.o.bubble.proposal is None and rig.client.calls[-1] == ("feedback", "s1", "ignored")
    rig.o.orb.clicked.emit()                                        # you asked: show it anyway
    rig.deliver(sug("s2", template="write_tests", text="Write a test for app.py"))
    assert rig.o.bubble.proposal is not None


def test_click_looks_now_then_card_or_nothing(rig):
    rig.o.orb.clicked.emit()
    assert ("trigger",) in rig.client.calls and rig.states[-1] == "looking"
    assert "Looking at your screen" in rig.o.bubble.body.text()
    rig.deliver(sug("s1"))
    assert rig.o.bubble.proposal is not None and rig.states[-1] == "suggesting"
    rig.o.bubble.cancel.click()
    rig.o.orb.clicked.emit()
    rig.s._nothing_found()                                          # the look timed out
    assert "Nothing worth suggesting" in rig.o.bubble.body.text() and rig.states[-1] == "idle"
    rig.client.triggered = False                                    # SOMA paused on its side
    rig.o.orb.clicked.emit()
    assert "paused" in rig.o.bubble.body.text() and rig.states[-1] == "failed"


def test_toggle_off_pauses_soma_and_clears_the_card(rig):
    rig.deliver(sug())
    rig.s.toggle()
    assert not rig.s.enabled and ("pause",) in rig.client.calls
    assert ("feedback", "s1", "ignored") in rig.client.calls
    rig.deliver(sug("s2"))
    assert "Suggestions off" in rig.o.bubble.body.text()
    rig.o.orb.clicked.emit()
    assert ("trigger",) not in rig.client.calls                     # off means off
    rig.s.toggle()
    assert rig.s.enabled and rig.client.calls[-1] == ("resume",)


def test_folders_add_index_done_and_remove(rig):
    seen = []
    rig.s.folders_changed.connect(seen.append)
    rig.s.add_folder("C:/notes")
    assert ("add_folder", "C:/notes") in rig.client.calls and rig.states[-1] == "indexing"
    assert rig.s._indexing.isActive()
    rig.client.folder_data = {"folders": [{"path": "C:/notes", "memories": 12}], "indexing": None,
                              "last_report": {"path": "C:/notes", "added": 12, "updated": 0}}
    rig.s.refresh_folders()
    assert not rig.s._indexing.isActive() and "Indexed 12 file(s) from notes" in rig.o.bubble.body.text()
    assert seen[-1] == [{"path": "C:/notes", "memories": 12}] and rig.states[-1] == "idle"
    rig.s.remove_folder("C:/notes")
    assert "forgot 3 file(s)" in rig.o.bubble.body.text() and seen[-1] == []
    rig.s.add_folder("bad")
    assert "home directory" in rig.o.bubble.body.text() and rig.states[-1] == "failed"


def test_lazy_connect_and_late_offline_report(qapp, monkeypatch):
    o = Overlay(tray=False)
    notes, attempts = [], []
    monkeypatch.setattr(o, "notify", lambda title, text: notes.append(text))

    def connect():
        attempts.append(1)
        raise soma.SomaError("SOMA isn't running.")

    s = soma.Suggester(o, connect=connect, spawn=lambda f: f())
    for _ in range(soma.REPORT_AFTER - 1):
        s.poll_once()
    assert notes == [] and len(attempts) == soma.REPORT_AFTER - 1   # still starting up: stay quiet
    s.poll_once()
    qapp.processEvents()
    s.poll_once()
    qapp.processEvents()
    assert len(notes) == 1 and o.orb.label() == "SOMA offline"


def test_client_endpoint_errors_and_localhost_only(tmp_path):
    ep = tmp_path / "ep.json"
    ep.write_text(json.dumps({"url": "http://127.0.0.1:5123", "token": "t0k"}))
    seen = []
    c = soma.SomaClient.from_endpoint(ep, request=lambda m, path, body: seen.append((m, path, body)) or (200, {}))
    c.add_folder("C:/notes")
    c.trigger()
    assert seen == [("POST", "/folders", {"path": "C:/notes"}), ("POST", "/trigger", {})]
    with pytest.raises(soma.SomaError):
        soma.SomaClient("http://evil.example:80", "t")
    with pytest.raises(soma.SomaError):
        soma.SomaClient.from_endpoint(tmp_path / "missing.json")
    bad = soma.SomaClient("http://127.0.0.1:1", "t", request=lambda *a: (400, {"error": "not a folder: X"}))
    with pytest.raises(soma.SomaError, match="not a folder"):
        bad.add_folder("X")


def test_kind_and_opener():
    assert soma.kind_of(sug()) == "soma:review_changes"
    assert soma.kind_of(sug(template="", action="EXPLAIN_ERROR")) == "soma:explain_error"
    assert soma.opener(sug()) is None
    assert soma.opener(sug(file="a.py", line=3)) == ("a.py", 3)
    assert soma.opener(sug(file="a.py", line="3")) == ("a.py", None)  # only real ints become a line
    assert soma.opener(sug(memory_source="n.md")) == ("n.md", None)
