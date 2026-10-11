import json
from types import SimpleNamespace

import pytest

from pet.brain import Brain, BrainError, Draft, build_messages, describe_openai_error, parse_draft
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


def test_build_messages_is_text_only_with_file_and_diff():
    ctx = Context(ActiveFile("auth.py", "python", "def f(): pass", "f"), "+    check_token()")
    msgs = build_messages("send it", ctx)
    text = msgs[1]["content"]
    assert isinstance(text, str)                       # no screenshot / image parts
    assert "send it" in text and "auth.py" in text and "Selected text" in text
    assert "Uncommitted git changes" in text and "check_token" in text
    assert "summary" in msgs[0]["content"].lower()
    assert "No uncommitted git changes" in build_messages("x", Context(ctx.active_file))[1]["content"]


def test_code_is_attached_by_pette_not_the_model():
    from pet.brain import attach_code
    af = ActiveFile("C:/r/auth.py", "python", "def f(): pass", "")
    d = Draft("slack", "Sarah", "", "Added a check.")
    assert attach_code(d, Context(af, "+x")).code == "+x" and attach_code(d, Context(af, "+x")).code_lang == "diff"
    assert attach_code(d, Context(af)).code == "def f(): pass" and attach_code(d, Context(af)).code_label == "auth.py"
    sel = ActiveFile("C:/r/auth.py", "python", "a\nb", "b")
    assert attach_code(d, Context(sel)).code_label == "auth.py · selection"
    assert attach_code(Draft("none", "", "", "hi"), Context(af, "+x")).code == ""


def test_brain_uses_client_and_wraps_errors():
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ok()))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    d = Brain(client, model="m").draft("hi", Context(None))
    assert d.recipient == "Sarah" and calls[0]["model"] == "m"
    assert calls[0]["response_format"] == {"type": "json_object"}

    def boom(**kw):
        raise TimeoutError()

    client.chat.completions.create = boom
    with pytest.raises(BrainError, match="timed out"):
        Brain(client).draft("hi", Context(None))


def test_describe_quota_error():
    class RateLimitError(Exception):
        pass
    e = RateLimitError("Error code: 429 - {'error': {'type': 'insufficient_quota'}}")
    assert "no credits" in describe_openai_error(e)
    assert "rate limit" in describe_openai_error(RateLimitError("slow down"))


def test_brain_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(BrainError, match="GROQ_API_KEY"):
        Brain().draft("hi", Context(None))


def test_offline_brain_drafts():
    from pet.brain import OfflineBrain
    d = OfflineBrain().draft("Send a Slack to Sarah asking her to review this code",
                             Context(ActiveFile("C:/x/auth.py", "python", "", "")))
    assert (d.platform, d.recipient) == ("slack", "Sarah")
    assert d.message == "Hi Sarah, could you review this code in auth.py. Thanks!"
    assert OfflineBrain().draft("what time is it", Context(None)).platform == "none"
    with pytest.raises(BrainError):
        parse_draft(ok(platform="agent", recipient=""))   # the coding-agent route is gone: one workflow
