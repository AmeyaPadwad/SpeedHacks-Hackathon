from types import SimpleNamespace

import pytest

from pet.voice import VoiceError, transcribe


def client_returning(text=None, exc=None):
    def create(model, file):
        assert model == "whisper-1" and file.name == "command.wav"
        if exc:
            raise exc
        return SimpleNamespace(text=text)
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


def test_transcribe_ok():
    assert transcribe(b"RIFF", client_returning("  send it  ")) == "send it"


def test_transcribe_empty_and_errors():
    with pytest.raises(VoiceError):
        transcribe(b"RIFF", client_returning("   "))
    with pytest.raises(VoiceError, match="ConnectionError"):
        transcribe(b"RIFF", client_returning(exc=ConnectionError()))


def test_transcribe_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(VoiceError, match="OPENAI_API_KEY"):
        transcribe(b"RIFF")
