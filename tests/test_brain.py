import json
from types import SimpleNamespace

import pytest

from pet.brain import Brain, BrainError, build_messages, describe_openai_error, parse_draft
from pet.context import ActiveFile, Context


def ok(**over):
    d = {"platform": "slack", "recipient": "Sarah", "subject": "", "message": "Can you review auth.py?"}
    d.update(over)
    return json.dumps(d)


def test_parse_valid():
    d = parse_draft(ok(platform=" Slack "))
    assert d.platform == "slack" and d.recipient == "Sarah"


@pytest.mark.parametrize("raw", [
    "not json",
    "[]",
    json.dumps({"platform": "slack", "recipient": "S", "message": "hi"}),  # missing key
    ok(action="rm -rf"),  # extra key
    ok(platform="sms"),
    ok(message="   "),
    ok(recipient=""),
    ok(message=5),
])
def test_parse_rejects(raw):
    with pytest.raises(BrainError):
        parse_draft(raw)


def test_parse_none_platform_needs_no_recipient():
    assert parse_draft(ok(platform="none", recipient="")).platform == "none"


def test_build_messages_includes_context():
    ctx = Context(b"\xff\xd8jpeg", ActiveFile("auth.py", "python", "def f(): pass", "f"))
    msgs = build_messages("send it", ctx)
    parts = msgs[1]["content"]
    assert "send it" in parts[0]["text"]
    assert "auth.py" in parts[1]["text"] and "Selected text" in parts[1]["text"]
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_brain_uses_client_and_wraps_errors():
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ok()))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    d = Brain(client, model="m").draft("hi", Context(None, None))
    assert d.recipient == "Sarah" and calls[0]["model"] == "m"
    assert calls[0]["response_format"] == {"type": "json_object"}

    def boom(**kw):
        raise TimeoutError()

    client.chat.completions.create = boom
    with pytest.raises(BrainError, match="timed out"):
        Brain(client).draft("hi", Context(None, None))


def test_describe_quota_error():
    class RateLimitError(Exception):
        pass
    e = RateLimitError("Error code: 429 - {'error': {'type': 'insufficient_quota'}}")
    assert "no credits" in describe_openai_error(e)
    assert "rate limit" in describe_openai_error(RateLimitError("slow down"))


def test_brain_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BrainError, match="OPENAI_API_KEY"):
        Brain().draft("hi", Context(None, None))
