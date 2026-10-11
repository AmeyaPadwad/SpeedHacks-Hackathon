import json

import pytest

from pet import dispatch as dp
from pet.brain import Draft


class Resp:
    def __init__(self, code):
        self.status_code = code


@pytest.fixture
def posts(monkeypatch):
    sent = []
    monkeypatch.setattr(dp.requests, "post", lambda url, json, timeout: sent.append((url, json)) or Resp(200))
    return sent


def draft(platform="slack", recipient="Sarah"):
    return Draft(platform, recipient, "", "please review")


def test_slack_with_contact(monkeypatch, posts):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/s")
    msg = dp.dispatch(draft(), {"sarah": {"slack": "U1"}})
    assert posts == [("https://hooks.example/s", {"text": "<@U1> please review"})]
    assert "Slack" in msg


def test_discord_restricts_mentions(monkeypatch, posts):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hooks.example/d")
    dp.dispatch(draft("discord"), {})
    assert posts[0][1]["allowed_mentions"] == {"users": []}
    dp.dispatch(draft("discord"), {"sarah": {"discord": 42}})
    assert posts[1][1] == {"content": "<@42> please review", "allowed_mentions": {"users": ["42"]}}


def test_missing_webhook(monkeypatch, posts):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    with pytest.raises(dp.DispatchError, match="SLACK_WEBHOOK_URL"):
        dp.dispatch(draft(), {})
    assert posts == []


def test_http_error(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/s")
    monkeypatch.setattr(dp.requests, "post", lambda *a, **k: Resp(403))
    with pytest.raises(dp.DispatchError, match="403"):
        dp.dispatch(draft(), {})


def test_email_needs_address(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example")
    with pytest.raises(dp.DispatchError, match="No email address"):
        dp.dispatch(draft("email"), {})


def test_none_platform_not_sendable():
    with pytest.raises(dp.DispatchError):
        dp.dispatch(draft("none", ""), {})


def test_load_contacts(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"Sarah ": {"slack": "U1"}, "bad": "x"}))
    assert dp.load_contacts(f) == {"sarah": {"slack": "U1"}}
    assert dp.load_contacts(tmp_path / "missing.json") == {}
