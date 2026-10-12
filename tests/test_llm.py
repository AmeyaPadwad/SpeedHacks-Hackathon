from pet import llm


def test_client_targets_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    c = llm.client(5)
    assert str(c.base_url).rstrip("/") == llm.BASE_URL and c.api_key == "gsk_test" and c.max_retries == 0


def test_client_none_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.client(5) is None and not llm.configured()


def test_models_overridable(monkeypatch):
    for var in ("GROQ_MODEL", "GROQ_TEXT_MODEL", "GROQ_STT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert llm.chat_model() == "qwen/qwen3.8-27b"
    assert llm.text_model() == "openai/gpt-oss-120b" and llm.extra_args(llm.text_model()) == {"reasoning_effort": "low"}
    assert llm.extra_args(llm.chat_model()) == {}
    monkeypatch.setenv("GROQ_MODEL", "other/model")
    assert llm.chat_model() == "other/model"


def test_rate_limit_message_includes_wait():
    class RateLimitError(Exception):
        pass
    msg = llm.describe_error(RateLimitError("Rate limit reached ... Please try again in 7.62s. Need more tokens?"))
    assert msg == "Groq rate limit hit. Try again in 8s."
    assert "gsk_" not in llm.describe_error(type("AuthenticationError", (Exception,), {})("bad key gsk_123"))
