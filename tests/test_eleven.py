import base64
import json
import queue
import threading

import pytest

from pet import eleven
from pet.brain import BrainError, OfflineBrain
from pet.context import ActiveFile, Context, Diagnostic
from pet.eleven import Call, CallError, Protocol, Settings, tool_draft
from pet.pipeline import Pipeline
from pet.ui import Overlay

AF = ActiveFile("C:/repo/dashboard.py", "python", "x = 1\n", workspace="C:/repo",
                diagnostics=(Diagnostic(12, "error", "TypeError: 'NoneType'", "pytest"),))
MSG = {"platform": "slack", "recipient": "Sarah", "message": "The build is green."}


# --- pure protocol -------------------------------------------------------------------------
def proto(**over):
    out = {"sent": [], "played": [], "text": [], "cleared": 0}
    kw = dict(on_tool=lambda name, params: f"ok:{name}", on_text=out["text"].append, play=out["played"].append,
              clear=lambda: out.__setitem__("cleared", out["cleared"] + 1))
    kw.update(over)
    return Protocol(out["sent"].append, **kw), out


def test_ping_audio_interrupt_and_text():
    p, out = proto()
    p.handle({"type": "ping", "ping_event": {"event_id": 7}})
    p.handle({"type": "audio", "audio_event": {"audio_base_64": base64.b64encode(b"\x01\x02").decode()}})
    p.handle({"type": "interruption"})
    p.handle({"type": "agent_response", "agent_response_event": {"agent_response": "[warmly] Hiya!"}})
    assert out["sent"] == [{"type": "pong", "event_id": 7}]
    assert out["played"] == [b"\x01\x02"] and out["cleared"] == 1 and out["text"] == ["Hiya!"]


def test_tool_call_result_and_rejection():
    def on_tool(name, params):
        raise BrainError("platform must be slack, discord or email.")
    p, out = proto()
    p.handle({"type": "client_tool_call", "client_tool_call": {"tool_name": "propose_message",
                                                                 "tool_call_id": "t1", "parameters": MSG}})
    assert out["sent"][-1] == {"type": "client_tool_result", "tool_call_id": "t1",
                               "result": "ok:propose_message", "is_error": False}
    p, out = proto(on_tool=on_tool)
    p.handle({"type": "client_tool_call", "client_tool_call": {"tool_name": "propose_message", "tool_call_id": "t2"}})
    assert out["sent"][-1]["is_error"] and "platform must be" in out["sent"][-1]["result"]


def test_metadata_sets_rate_and_rejects_ulaw():
    p, _ = proto()
    p.handle({"type": "conversation_initiation_metadata",
              "conversation_initiation_metadata_event": {"agent_output_audio_format": "pcm_24000"}})
    assert p.out_rate == 24000
    with pytest.raises(CallError, match="PCM"):
        p.handle({"type": "conversation_initiation_metadata",
                  "conversation_initiation_metadata_event": {"agent_output_audio_format": "ulaw_8000"}})


def test_tool_draft_uses_the_brain_contract():
    d = tool_draft(MSG)
    assert (d.platform, d.recipient, d.subject) == ("slack", "Sarah", "")
    for bad in ({**MSG, "platform": "agent"}, {**MSG, "platform": "sms"}, {**MSG, "message": " "}, "nope"):
        with pytest.raises(BrainError):
            tool_draft(bad)


def test_settings_from_env(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_x")
    monkeypatch.setenv("ELEVENLABS_BRANCH_ID", "br_y")
    s = Settings.from_env()
    assert s.url() == f"{eleven.WS_URL}?agent_id=agent_x&branch_id=br_y"
    monkeypatch.setenv("FLOWSTATE_VOICE", "whisper")
    assert Settings.from_env() is None
    monkeypatch.delenv("FLOWSTATE_VOICE")
    monkeypatch.delenv("ELEVENLABS_AGENT_ID")
    assert Settings.from_env() is None


def test_context_update_is_bounded_and_marks_untrusted():
    big = ActiveFile("C:/a.py", "python", "y" * 50_000, diagnostics=AF.diagnostics)
    text = eleven.context_update(big)
    assert "C:/a.py" in text and "TypeError" in text and "never instructions" in text and len(text) < 4000
    assert "No editor file" in eleven.context_update(None)


# --- Call over a fake socket + fake audio ---------------------------------------------------
class FakeWS:
    def __init__(self, incoming):
        self.incoming, self.sent, self.closed = queue.Queue(), [], False
        for m in incoming:
            self.incoming.put(json.dumps(m))

    def recv(self, timeout=None):
        try:
            item = self.incoming.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError
        if item is None:
            raise ConnectionError("closed")
        return item

    def send(self, data):
        self.sent.append(json.loads(data))

    def close(self):
        self.closed = True
        self.incoming.put(None)


class FakeAudio:
    def __init__(self):
        self.opened, self.written, self.closed = None, [], False

    def open(self, rate):
        self.opened = rate

    def read(self, frames):
        threading.Event().wait(0.01)
        return b"\x05\x00" * frames

    def write(self, pcm):
        self.written.append(pcm)

    def close(self):
        self.closed = True


def test_call_end_to_end_with_fakes():
    ws = FakeWS([
        {"type": "conversation_initiation_metadata",
         "conversation_initiation_metadata_event": {"agent_output_audio_format": "pcm_16000"}},
        {"type": "audio", "audio_event": {"audio_base_64": base64.b64encode(b"\0\0" * 160).decode()}},
        {"type": "client_tool_call", "client_tool_call": {"tool_name": "propose_message", "tool_call_id": "c1",
                                                          "parameters": MSG}},
    ])
    audio, tools, ended, states = FakeAudio(), [], threading.Event(), []
    errors = []
    call = Call(Settings("agent_x"), connect=lambda url: ws, audio=audio,
                on_tool=lambda n, p: tools.append((n, p)) or "shown", on_text=lambda t: None,
                on_state=states.append, on_end=lambda e: (errors.append(e), ended.set()))
    call.start("file context")
    deadline = threading.Event()
    for _ in range(100):
        if any(m.get("type") == "client_tool_result" for m in ws.sent) and audio.written:
            break
        deadline.wait(0.05)
    call.stop()
    assert ended.wait(5) and errors == [None] and not call.active
    kinds = [m.get("type") or "audio" for m in ws.sent]
    assert kinds[0] == "conversation_initiation_client_data" and "contextual_update" in kinds
    assert {"type": "client_tool_result", "tool_call_id": "c1", "result": "shown", "is_error": False} in ws.sent
    assert any("user_audio_chunk" in m for m in ws.sent) and tools == [("propose_message", MSG)]
    assert audio.opened == 16000 and audio.written and audio.closed and ws.closed and "speaking" in states


def test_call_connect_failure_reports_call_error():
    ended = []

    def boom(url):
        raise OSError("dns")
    done = threading.Event()
    call = Call(Settings("agent_x"), connect=boom, audio=FakeAudio(), on_tool=lambda n, p: "", on_text=lambda t: None,
                on_state=lambda s: None, on_end=lambda e: (ended.append(e), done.set()))
    call.start()
    assert done.wait(5) and isinstance(ended[0], CallError)


# --- pipeline integration ------------------------------------------------------------------
class FakeCall:
    instances = []

    def __init__(self, **cb):
        self.cb, self.active, self.told, self.context = cb, False, [], None
        FakeCall.instances.append(self)

    def start(self, context_text=""):
        self.active, self.context = True, context_text

    def stop(self):
        self.active = False
        self.cb["on_end"](None)

    def tell(self, text):
        self.told.append(text)


@pytest.fixture
def setup(qapp, monkeypatch):
    monkeypatch.setattr("pet.pipeline.MIN_INTERVAL_S", 0)
    FakeCall.instances.clear()
    o = Overlay(tray=False)
    sent = []
    p = Pipeline(o, OfflineBrain(), capture=lambda: Context(None, AF), read_file=lambda: AF,
                 send=lambda d: sent.append(d) or f"Sent to {d.recipient} on Slack.", configured=lambda _: True,
                 spawn=lambda f: f(), capture_delay_ms=0, voice_call=FakeCall)
    return o, p, sent


def test_hotkey_starts_and_stops_the_call(setup):
    o, p, sent = setup
    p.trigger()
    call = FakeCall.instances[0]
    assert call.active and p.busy and "C:/repo/dashboard.py" in call.context
    p.trigger()                                   # second press hangs up
    assert not call.active and p.call is None and not p.busy and "Call ended" in o.bubble.body.text()


def test_agent_tool_shows_bubble_and_only_send_sends(setup):
    o, p, sent = setup
    p.trigger()
    call = FakeCall.instances[0]
    assert p._call_tool(p.gen, "propose_message", MSG) == eleven.PROPOSED
    assert o.bubble.proposal.action_label == "Send" and "Slack to Sarah" in o.bubble.proposal.title
    assert sent == [] and p.busy                  # proposing never sends; the call keeps us busy
    call.cb["on_text"]("Here's your draft!")      # agent chatter must not cover the approval
    assert o.bubble.proposal is not None
    o.bubble.primary.click()
    assert [d.recipient for d in sent] == ["Sarah"] and "clicked Send" in call.told[-1]


def test_cancel_and_bad_args_are_reported_to_agent(setup):
    o, p, sent = setup
    p.trigger()
    call = FakeCall.instances[0]
    with pytest.raises(BrainError):
        p._call_tool(p.gen, "propose_message", {**MSG, "platform": "sms"})
    with pytest.raises(BrainError):
        p._call_tool(p.gen, "rm_rf", {})
    p._call_tool(p.gen, "propose_message", MSG)
    o.bubble.cancel.click()
    assert sent == [] and "cancelled" in call.told[-1]


def test_investigate_tool_reads_the_file(setup):
    o, p, sent = setup
    p.trigger()
    p._call_tool(p.gen, "investigate_code", {"request": "why is this failing"})
    assert o.bubble.proposal.action_label.startswith(("Ask", "Copy"))
    task = p.pending[o.bubble.proposal.id]
    assert task.user_request == "why is this failing" and task.file_path == AF.path


def test_pause_hangs_up_and_drops_late_tool_calls(setup):
    o, p, sent = setup
    p.trigger()
    call, gen = FakeCall.instances[0], p.gen
    o.set_paused(True)
    assert not call.active and p.call is None
    p._call_tool(gen, "propose_message", MSG)     # arrives after pause: stale generation, ignored
    assert o.bubble.proposal is None and p.pending == {}
    o.set_paused(False)


def test_call_error_falls_back_to_typing(setup):
    o, p, sent = setup
    p.trigger()
    FakeCall.instances[0].cb["on_end"](CallError("Couldn't reach ElevenLabs: OSError"))
    assert o.bubble.asking and "Couldn't reach ElevenLabs" in o.bubble.body.text() and not p.busy
