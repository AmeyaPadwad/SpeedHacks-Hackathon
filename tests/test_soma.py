"""SOMA suggestions in Pette: shown only when Pette is free, nothing runs without "Do it", feedback goes back."""
import json

import pytest

from db import Store
from pet import soma
from pet.brain import Draft
from pet.context import ActiveFile, Context
from pet.pipeline import Pipeline
from pet.ui import Overlay

CTX = Context(ActiveFile("C:/repo/app.py", "python", "x = 1\n"), "--- a/app.py\n+++ b/app.py\n+x = 1")
DRAFT = Draft("slack", "team", "", "Summary: set x.", code=CTX.changes, code_label="app.py", code_lang="diff")


def sug(sid="s1", template="share_changes", action="SUGGEST_NEXT_STEP", **kw):
    base = {"available": True, "silent": False, "valid": True, "id": sid, "action": action,
            "text": 'Send the team a summary of your changes to "app.py"', "evidence": ['editing "app.py"'],
            "template": template}
    return {**base, **kw}


class FakeClient:
    def __init__(self, *suggestions):
        self.queue = list(suggestions)
        self.calls = []

    def suggestion(self):
        if not self.queue:
            return {"available": True, "silent": True}
        s = self.queue.pop(0)
        if isinstance(s, Exception):
            raise s
        return s

    def feedback(self, sid, outcome):
        self.calls.append(("feedback", sid, outcome))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))


class FakeBrain:
    def __init__(self):
        self.commands = []

    def draft(self, command, ctx):
        self.commands.append(command)
        return DRAFT


@pytest.fixture
def rig(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    o = Overlay(tray=False)
    brain, sent, copied = FakeBrain(), [], []
    p = Pipeline(o, brain, capture=lambda: CTX, send=lambda d: sent.append(d) or "Sent.",
                 configured=lambda _: True, use_voice=False, spawn=lambda f: f(), send_delay_s=2)
    store = Store(tmp_path / "m.sqlite3")
    client = FakeClient()
    s = soma.Suggester(o, p, client, memory=store, spawn=lambda f: f(), clipboard=copied.append)

    def deliver(*suggestions):
        client.queue.extend(suggestions)
        for _ in suggestions:
            s.poll_once()
        qapp.processEvents()

    yield o, p, s, client, brain, sent, copied, store, deliver
    store.close()


def finish(p):
    while p._countdown.isActive():
        p._tick()


def test_suggestion_card_then_do_it_runs_the_pette_workflow(rig):
    o, p, s, client, brain, sent, _, store, deliver = rig
    deliver(sug())
    assert o.bubble.proposal.title == "SOMA suggests · Next step"
    assert o.bubble.primary.text() == "Do it" and o.bubble.cancel.text() == "Not now"
    assert o.orb.label() == ""                                     # ring reset: no stale status pill
    assert sent == [] and brain.commands == []                     # nothing runs on its own
    o.bubble.primary.click()
    assert brain.commands == [soma.COMMANDS["share_changes"]]      # fixed text, no screen content
    assert p.state == "pending"                                    # normal countdown: Cancel still works
    finish(p)
    assert sent == [DRAFT]
    assert ("feedback", "s1", "accepted") in client.calls
    assert store.query("SELECT kind, outcome FROM offers ORDER BY ts") == [("soma:share_changes", "shown"),
                                                                          ("soma:share_changes", "accepted")]


def test_not_now_dismisses_and_reports_rejected(rig):
    o, p, s, client, brain, sent, *_ = rig
    deliver = rig[-1]
    deliver(sug())
    o.bubble.cancel.click()
    assert o.bubble.proposal is None and brain.commands == [] and sent == []
    assert client.calls[-1] == ("feedback", "s1", "rejected")


def test_unanswered_card_expires_as_ignored(rig):
    o, p, s, client, *_ = rig
    rig[-1](sug())
    s._expired()
    assert o.bubble.proposal is None and client.calls[-1] == ("feedback", "s1", "ignored")


def test_same_suggestion_is_shown_once(rig):
    o, p, s, client, *_ = rig
    deliver = rig[-1]
    deliver(sug())
    o.bubble.cancel.click()
    deliver(sug())
    assert o.bubble.proposal is None


def test_not_shown_while_busy_hidden_paused_or_silent(rig):
    o, p, s, client, brain, sent, copied, store, deliver = rig
    deliver(sug(silent=True), sug("s2", valid=False), sug("s3", text=""))
    assert o.bubble.proposal is None
    o.context_allowed = False                                      # focus gate hid Pette
    deliver(sug("s4"))
    assert o.bubble.proposal is None
    o.context_allowed = True
    p.trigger()                                                    # typing a command: Pette is busy
    deliver(sug("s5"))
    assert o.bubble.asking and o.bubble.proposal is None
    o.bubble.cancel.click()
    o.set_paused(True)
    deliver(sug("s6"))
    assert o.bubble.proposal is None and ("pause",) in client.calls
    o.set_paused(False)
    deliver(sug("s7"))
    assert o.bubble.proposal is not None and ("resume",) in client.calls


def test_non_actionable_suggestion_is_copied(rig):
    o, p, s, client, brain, sent, copied, store, deliver = rig
    deliver(sug(template="", action="EXPLAIN_ERROR", text='Explain the error "TypeError: x"'))
    assert o.bubble.primary.text() == "Copy"
    o.bubble.primary.click()
    assert copied == ['Explain the error "TypeError: x"'] and brain.commands == []
    assert "Copied" in o.bubble.body.text()


def test_search_tools_goes_to_darwin_with_a_clean_topic():
    s = sug(template="search_tools", evidence=["Vector DBs <script>`rm`</script>"])
    assert soma.command_for(s) == "search Darwin for Vector DBs script rm script"
    assert soma.command_for(sug(template="search_tools", evidence=["<>"])) is None
    assert soma.command_for(sug(template="unknown")) is None


def test_toggle_off_pauses_soma_and_clears_the_card(rig):
    o, p, s, client, *_ = rig
    rig[-1](sug())
    s.toggle()
    assert not s.enabled and ("pause",) in client.calls and client.calls[0] == ("feedback", "s1", "ignored")
    rig[-1](sug("s2"))
    assert "Suggestions off" in o.bubble.body.text()
    s.toggle()
    assert s.enabled and client.calls[-1] == ("resume",)


def test_lazy_connect_and_late_error_report(rig, monkeypatch):
    o, p, *_ = rig
    notes, attempts = [], []
    monkeypatch.setattr(o, "notify", lambda title, text: notes.append(text))

    def connect():
        attempts.append(1)
        raise soma.SomaError("SOMA isn't running (no endpoint file).")

    s = soma.Suggester(o, p, connect=connect, spawn=lambda f: f())
    for _ in range(soma.REPORT_AFTER - 1):
        s.poll_once()
    assert notes == [] and len(attempts) == soma.REPORT_AFTER - 1   # still starting up: stay quiet
    s.poll_once()
    s.poll_once()
    assert len(notes) == 1 and "isn't running" in notes[0]


def test_client_reads_endpoint_and_refuses_remote(tmp_path):
    ep = tmp_path / "ep.json"
    ep.write_text(json.dumps({"url": "http://127.0.0.1:5123", "token": "t0k"}))
    seen = []
    c = soma.SomaClient.from_endpoint(ep, request=lambda m, path, body: seen.append((m, path, body)) or (200, {}))
    c.feedback("s1", "accepted")
    assert seen == [("POST", "/feedback", {"suggestion_id": "s1", "outcome": "accepted"})]
    with pytest.raises(soma.SomaError):
        soma.SomaClient("http://evil.example:80", "t")
    with pytest.raises(soma.SomaError):
        soma.SomaClient.from_endpoint(tmp_path / "missing.json")
    bad = soma.SomaClient("http://127.0.0.1:1", "t", request=lambda *a: (401, {}))
    with pytest.raises(soma.SomaError, match="401"):
        bad.suggestion()
