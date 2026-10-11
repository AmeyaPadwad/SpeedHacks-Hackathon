"""LLM provider: Groq through its OpenAI-compatible API (the `openai` SDK with a Groq base URL).

One place for the key, base URL, model names and user-facing error text, so the brain, analyst
and transcription stay provider-agnostic. Models were chosen against Groq's live model list:
qwen/qwen3.8-27b handles JSON mode and screenshots (drafting); openai/gpt-oss-120b is text-only
and fast (analysis). Free tier: ~7k input tokens/min per model, and each screenshot costs ~1.8k.
"""
import os
import re

BASE_URL = "https://api.groq.com/openai/v1"
KEY_VAR = "GROQ_API_KEY"
PROVIDER = "Groq"


def chat_model() -> str:
    return os.environ.get("GROQ_MODEL") or "qwen/qwen3.8-27b"


def vision_model() -> str:
    """Screenshot-capable drafting when an old input configuration selects text-only GPT-OSS."""
    return os.environ.get("GROQ_VISION_MODEL") or "qwen/qwen3.8-27b"


def text_model() -> str:
    """Text-only work (file analysis). A different model has its own rate-limit bucket on Groq."""
    return os.environ.get("GROQ_TEXT_MODEL") or "openai/gpt-oss-120b"


def extra_args(model: str) -> dict:
    """gpt-oss models reason before answering; low effort keeps the 500-token budget for the JSON."""
    return {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}


def stt_model() -> str:
    return os.environ.get("GROQ_STT_MODEL") or "whisper-large-v3-turbo"


def configured() -> bool:
    return bool(os.environ.get(KEY_VAR))


def client(timeout: float):
    """A fresh SDK client, or None when the key isn't set (callers raise their own error type)."""
    if not configured():
        return None
    from openai import OpenAI
    return OpenAI(api_key=os.environ[KEY_VAR], base_url=BASE_URL, timeout=timeout, max_retries=0)


def describe_error(e: Exception) -> str:
    """Short user-facing reason for an SDK failure; never includes the key or response body."""
    name = type(e).__name__
    text = str(e)
    if "insufficient_quota" in text or "credit_balance_exhausted" in text:
        return f"{PROVIDER} account has no credits left."
    if name == "AuthenticationError":
        return f"{PROVIDER} rejected the API key."
    if name == "RateLimitError":
        wait = re.search(r"try again in ([0-9.]+)s", text)
        return (f"{PROVIDER} rate limit hit. Try again in {float(wait[1]):.0f}s." if wait
                else f"{PROVIDER} rate limit hit. Wait a moment and try again.")
    if name in ("APITimeoutError", "TimeoutError"):
        return f"{PROVIDER} timed out."
    if name == "APIConnectionError":
        return f"Couldn't reach {PROVIDER}. Check your connection."
    if name == "NotFoundError" or "model_not_found" in text or "decommissioned" in text:
        return f"{PROVIDER} doesn't offer that model; set GROQ_MODEL."
    return f"AI request failed: {name}"
