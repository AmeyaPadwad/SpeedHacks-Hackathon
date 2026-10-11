import json
import threading
from types import SimpleNamespace

import pytest

from pet.inputs.project import inventory
from pet.inputs.selection import FileSelector, SelectionError, filename_candidates


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs)))
        data = next(self.responses)
        raw = data if isinstance(data, str) else json.dumps(data)
        if "tools" in kwargs:
            tool = SimpleNamespace(id=f"call_{len(self.calls)}", function=SimpleNamespace(name="select_context", arguments=raw))
            message = SimpleNamespace(content=None, tool_calls=[tool])
        else:
            message = SimpleNamespace(content=raw)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_selector_reads_task_relevant_file_then_returns_grounded_excerpt(tmp_path):
    (tmp_path / "auth.py").write_text("def login():\n    return False\n" + "# other\n" * 600)
    (tmp_path / "unrelated.py").write_text("print('hello')")
    client = Client([
        {"reads": [{"path": "auth.py", "start_line": 1, "end_line": 200}]},
        {"selected": [{"path": "auth.py", "start_line": 1, "end_line": 200, "reason": "Implements login"}]},
    ])
    result = FileSelector(client).select("Fix login", inventory(tmp_path), "auth.py")
    assert len(result.files) == 1
    assert result.files[0].path == "auth.py"
    assert "return False" in result.files[0].content
    assert len(result.files[0].context_text()) <= 4000
    assert "unrelated.py" not in client.calls[1]["messages"][-2]["content"]
    first_input = json.loads(client.calls[0]["messages"][1]["content"])
    assert first_input["hints"] == "auth.py"


def test_ambiguous_hints_return_question_without_reading(tmp_path):
    result = FileSelector(Client([{"question": "Which auth.py: src/ or tests/?"}])).select("Fix auth", inventory(tmp_path))
    assert result.question and not result.files


def test_unread_or_external_selected_paths_are_not_accepted(tmp_path):
    (tmp_path / "a.py").write_text("valid")
    client = Client([
        {"reads": [{"path": "../secret.py", "start_line": 1, "end_line": 2}]},
        {"selected": [{"path": "a.py", "start_line": 1, "end_line": 200}]},
        {"selected": [{"path": "../secret.py", "start_line": 1, "end_line": 2}]},
    ])
    result = FileSelector(client).select("fix", inventory(tmp_path))
    assert not result.files and result.warnings
    assert len(client.calls) == 3


@pytest.mark.parametrize("raw", ["not json", "[]", '{"reads": 123}', '{"selected": null}'])
def test_malformed_model_response_is_recoverable(tmp_path, raw):
    with pytest.raises(SelectionError):
        FileSelector(Client([raw])).select("fix", inventory(tmp_path))


def test_missing_key_and_cancel(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FLOWSTATE_INPUT_PROVIDER", raising=False)
    with pytest.raises(SelectionError, match="GROQ_API_KEY"):
        FileSelector().select("fix", inventory(tmp_path))
    cancel = threading.Event()
    cancel.set()
    client = Client([])
    assert not FileSelector(client).select("fix", inventory(tmp_path), cancel=cancel).files
    assert not client.calls


def test_spoken_hyphenated_filename_and_actual_read_range(tmp_path):
    (tmp_path / "macos.py").write_text("def capture_foreground():\n    return 'Code'\n")
    client = Client([
        {"reads": [{"path": "macos.py", "start_line": 1, "end_line": 200}]},
        {"selected": [{"path": "macos.py", "start_line": 1, "end_line": 2, "reason": "Explains macOS capture"}]},
    ])
    result = FileSelector(client).select(
        "I want to send my teammate a brief description of how the file mac-os.py works.", inventory(tmp_path))
    first_input = json.loads(client.calls[0]["messages"][1]["content"])
    assert first_input["filename_candidates"] == ["macos.py"]
    assert len(result.files) == 1 and result.files[0].path == "macos.py"
    assert "capture_foreground" in result.files[0].content and result.files[0].end_line == 2


def test_filename_matching_preserves_ambiguity():
    paths = ("src/mac_os.py", "other/macos.py", "tests/test_macos.py")
    assert filename_candidates("Describe mac-os.py", "", paths) == ["src/mac_os.py", "other/macos.py"]


def test_selected_subrange_contains_only_previously_read_lines(tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\nthree\nfour\n")
    client = Client([
        {"reads": [{"path": "a.py", "start_line": 1, "end_line": 3}]},
        {"selected": [{"path": "a.py", "start_line": 2, "end_line": 2}]},
    ])
    result = FileSelector(client).select("Explain the second line", inventory(tmp_path))
    assert result.files[0].content == "two\n" and result.files[0].start_line == 2

    outside = Client([
        {"reads": [{"path": "a.py", "start_line": 1, "end_line": 3}]},
        {"selected": [{"path": "a.py", "start_line": 2, "end_line": 4}]},
        {"selected": [{"path": "a.py", "start_line": 4, "end_line": 4}]},
    ])
    assert not FileSelector(outside).select("Explain", inventory(tmp_path)).files


def test_unavailable_model_error_explains_configuration_fix(tmp_path):
    class ModelNotFound(Exception):
        code = "model_not_found"

    def fail(**kwargs):
        raise ModelNotFound()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(SelectionError, match="OPENAI_MODEL=gpt-4o"):
        FileSelector(client, model="gpt-4", provider="openai").select("Explain macos.py", inventory(tmp_path))


def test_exhausted_credit_error_explains_manual_fallback(tmp_path):
    class NoCredits(Exception):
        code = "credit_balance_exhausted"

    def fail(**kwargs):
        raise NoCredits()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(SelectionError, match="credit balance is exhausted"):
        FileSelector(client).select("Explain macos.py", inventory(tmp_path))


@pytest.mark.parametrize("provider, key_env, model_env, model, url", [
    ("groq", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1"),
    ("openai", "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o", "https://api.openai.com/v1"),
])
def test_provider_routes_only_its_own_key_and_closes_client(tmp_path, monkeypatch, provider, key_env, model_env, model, url):
    import openai

    (tmp_path / "a.py").write_text("print('hello')\n")
    observed = {}
    client = Client([
        {"reads": [{"path": "a.py", "start_line": 1, "end_line": 1}]},
        {"selected": [{"path": "a.py", "start_line": 1, "end_line": 1}]},
    ])
    client.close = lambda: observed.update(closed=True)

    def construct(**kwargs):
        observed.update(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", construct)
    monkeypatch.delenv("FLOWSTATE_INPUT_MODEL", raising=False)
    monkeypatch.delenv(model_env, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.invalid")
    selector = FileSelector(provider=provider)
    result = selector.select("Explain a.py", inventory(tmp_path))
    assert result.files[0].path == "a.py"
    assert observed["api_key"] == ("groq-test-key" if provider == "groq" else "openai-test-key")
    assert observed["base_url"] == url and observed["max_retries"] == 0 and observed["closed"]
    assert client.calls[0]["model"] == model
    if provider == "groq":
        assert client.calls[0]["reasoning_effort"] == "low"
        assert "response_format" not in client.calls[0]
        assert client.calls[0]["tool_choice"] == {"type": "function", "function": {"name": "select_context"}}
        assert client.calls[0]["parallel_tool_calls"] is False
        assert client.calls[0]["tools"][0]["function"]["parameters"]["additionalProperties"] is False
        assert client.calls[0]["max_completion_tokens"] == 2400
        assert "max_tokens" not in client.calls[0]
    else:
        assert client.calls[0]["response_format"] == {"type": "json_object"}
        assert "reasoning_effort" not in client.calls[0]


def test_groq_never_falls_back_to_paid_openai_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-not-for-groq")
    with pytest.raises(SelectionError, match="GROQ_API_KEY"):
        FileSelector(provider="groq").select("Explain a.py", inventory(tmp_path))


def test_provider_and_model_configuration_are_independent(monkeypatch):
    monkeypatch.delenv("FLOWSTATE_INPUT_PROVIDER", raising=False)
    monkeypatch.delenv("FLOWSTATE_INPUT_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert FileSelector().provider == "groq"
    assert FileSelector().model == "openai/gpt-oss-20b"
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    assert FileSelector().model == "openai/gpt-oss-120b"
    monkeypatch.setenv("FLOWSTATE_INPUT_MODEL", "custom-override")
    assert FileSelector().model == "custom-override"
    assert FileSelector(model="explicit-model").model == "explicit-model"


def test_invalid_provider_has_recoverable_configuration_error(tmp_path):
    with pytest.raises(SelectionError, match="FLOWSTATE_INPUT_PROVIDER"):
        FileSelector(provider="not-a-provider").select("Explain", inventory(tmp_path))


@pytest.mark.parametrize("status, message", [(429, "Groq rate limit reached"), (401, "GROQ_API_KEY")])
def test_groq_errors_do_not_invoke_another_provider(tmp_path, status, message):
    calls = []

    class APIError(Exception):
        status_code = status

    def fail(**kwargs):
        calls.append(kwargs)
        raise APIError()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(SelectionError, match=message):
        FileSelector(client, provider="groq").select("Explain", inventory(tmp_path))
    assert len(calls) == 1


def test_invalid_json_is_not_reported_as_key_or_connection_failure(tmp_path):
    with pytest.raises(SelectionError, match="invalid JSON selection"):
        FileSelector(Client(["not json"])).select("Explain macos.py", inventory(tmp_path))


def test_truncated_model_response_reports_token_limit(tmp_path):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="{"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(SelectionError, match="response token limit"):
        FileSelector(client).select("Explain macos.py", inventory(tmp_path))


@pytest.mark.parametrize("error_class, status, message", [
    ("APIConnectionError", None, "Cannot connect to Groq"),
    ("APITimeoutError", None, "request timed out"),
    ("BadRequestError", 400, "HTTP 400"),
    ("TooLargeError", 413, "request/token limit"),
])
def test_model_and_network_errors_have_specific_messages(tmp_path, error_class, status, message):
    def fail(**kwargs):
        error = type(error_class, (Exception,), {"status_code": status})()
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(SelectionError, match=message):
        FileSelector(client).select("Explain", inventory(tmp_path))


def test_groq_uses_declared_function_and_returns_matching_tool_result(tmp_path):
    (tmp_path / "a.py").write_text("print('hello')\n")
    client = Client([
        {"reads": [{"path": "a.py", "start_line": 1, "end_line": 1}]},
        {"selected": [{"path": "a.py", "start_line": 1, "end_line": 1, "reason": "Explains the task"}]},
    ])
    result = FileSelector(client, provider="groq").select("Explain a.py", inventory(tmp_path))
    assert result.files[0].path == "a.py"
    messages = client.calls[1]["messages"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    tool_result = next(m for m in messages if m["role"] == "tool")
    assert assistant["tool_calls"][0]["function"]["name"] == "select_context"
    assert tool_result["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "print('hello')" in tool_result["content"]


@pytest.mark.parametrize("name, count", [("read_file", 1), ("select_context", 2), ("select_context", 0)])
def test_unknown_or_multiple_function_calls_are_never_executed(tmp_path, monkeypatch, name, count):
    tool = SimpleNamespace(id="call_bad", function=SimpleNamespace(name=name, arguments='{"reads": [{"path":"a.py"}]}'))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool] * count))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr("pet.inputs.selection.ProjectReader.read", lambda *args, **kwargs: pytest.fail("Unexpected function must not read files"))
    with pytest.raises(SelectionError, match="unexpected function call"):
        FileSelector(client).select("Explain", inventory(tmp_path))


def test_groq_output_parse_failure_is_reported_by_its_real_code(tmp_path):
    class OutputParseError(Exception):
        status_code = 400
        code = "output_parse_failed"

    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise OutputParseError()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(SelectionError, match="output_parse_failed"):
        FileSelector(client).select("Explain", inventory(tmp_path))
    assert len(calls) == 1
