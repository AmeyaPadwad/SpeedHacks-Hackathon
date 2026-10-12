import json
import queue
import threading

import pytest

from pet import eleven
from pet.eleven import LiveTranscriber, Settings, scribe
from pet.voice import VoiceError


def test_settings_from_env(monkeypatch):
    for v in ("ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID", "ELEVENLABS_BRANCH_ID", "FLOWSTATE_VOICE"):
        monkeypatch.delenv(v, raising=False)
    assert Settings.from_env() is None
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_x")
    monkeypatch.setenv("ELEVENLABS_BRANCH_ID", "br_y")
    s = Settings.from_env()
    assert s.mode == "live" and s.ws_url() == f"{eleven.WS_URL}?agent_id=agent_x&branch_id=br_y"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi_key")
    assert Settings.from_env().mode == "scribe"
    monkeypatch.setenv("FLOWSTATE_VOICE", "whisper")
    assert Settings.from_env() is None


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._body = code, body or {}

    def json(self):
        return self._body


def test_scribe_request_and_errors():
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return Resp(200, {"text": " Slack Sarah the fix is in. "})
    assert scribe(b"RIFF", "xi_key", post=post) == "Slack Sarah the fix is in."
    assert seen["url"] == eleven.STT_URL and seen["headers"] == {"xi-api-key": "xi_key"}
    assert seen["data"] == {"model_id": "scribe_v1"} and seen["files"]["file"][1] == b"RIFF"
    with pytest.raises(VoiceError, match="rejected the API key"):
        scribe(b"RIFF", "bad", post=lambda url, **kw: Resp(401))
    with pytest.raises(VoiceError, match="make out"):
        scribe(b"RIFF", "k", post=lambda url, **kw: Resp(200, {"text": "  "}))

    def offline(url, **kw):
        raise ConnectionError("dns")
    with pytest.raises(VoiceError, match="Couldn't reach ElevenLabs"):
        scribe(b"RIFF", "k", post=offline)


# --- live transcription over a fake socket + mic --------------------------------------------
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


class FakeMic:
    def __init__(self):
        self.opened = self.closed = False

    def open(self):
        self.opened = True

    def read(self, frames):
        threading.Event().wait(0.01)
        return b"\x00\x30" * frames

    def close(self):
        self.closed = True


def test_live_returns_first_user_transcript_and_ignores_agent():
    ws = FakeWS([
        {"type": "conversation_initiation_metadata", "conversation_initiation_metadata_event": {}},
        {"type": "audio", "audio_event": {"audio_base_64": "AAAA"}},              # her voice: never played
        {"type": "agent_response", "agent_response_event": {"agent_response": "[warmly] Hiya!"}},
        {"type": "ping", "ping_event": {"event_id": 3}},
        {"type": "user_transcript", "user_transcription_event": {"user_transcript": "Slack Sarah the fix is in."}},
    ])
    mic, levels = FakeMic(), []
    text = LiveTranscriber(Settings("agent_x"), connect=lambda url: ws, audio=mic, on_level=levels.append).run()
    assert text == "Slack Sarah the fix is in."
    assert ws.sent[0] == {"type": "conversation_initiation_client_data"}
    assert {"type": "pong", "event_id": 3} in ws.sent
    assert ws.closed and mic.opened and mic.closed


def test_live_times_out_and_cancels():
    clock = iter([0, 0, 100, 100, 100])
    ws = FakeWS([])
    with pytest.raises(VoiceError, match="Didn't hear"):
        LiveTranscriber(Settings("a"), connect=lambda url: ws, audio=FakeMic(), clock=lambda: next(clock)).run()
    ws2 = FakeWS([])
    t = LiveTranscriber(Settings("a"), connect=lambda url: ws2, audio=FakeMic())
    threading.Timer(0.2, t.cancel).start()
    with pytest.raises(VoiceError, match="cancelled"):
        t.run()


def test_live_connect_failure():
    def boom(url):
        raise OSError("dns")
    with pytest.raises(VoiceError, match="Couldn't reach ElevenLabs"):
        LiveTranscriber(Settings("a"), connect=boom, audio=FakeMic()).run()


def test_level_is_bounded():
    assert eleven._level(b"") == 0.0
    assert 0 < eleven._level(b"\x00\x30" * 100) <= 1.0
