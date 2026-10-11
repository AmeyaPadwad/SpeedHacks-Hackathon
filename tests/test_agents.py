import json
import subprocess
from types import SimpleNamespace

import pytest

from pet import agents
from pet.agents import AgentError, AgentTask, ClaudeCodeAgent, HermesAgent, build_prompt, pick_agent
from pet.analyst import Analyst, local_analysis, parse_analysis
from pet.context import ActiveFile, Context, Diagnostic

CODE = "\n".join(f"line{i}" for i in range(1, 101))
ERR = Diagnostic(50, "error", "x is undefined", "Pylance")


def ctx(errors=True):
    af = ActiveFile("C:/repo/app.py", "python", CODE, selection="line50", workspace="C:/repo",
                    diagnostics=(ERR,) if errors else ())
    return Context(b"\xff\xd8full", af)


def ai_json(**over):
    d = {"observation": "NameError", "evidence": ["line 50 error"], "likely_cause": "x not defined",
         "next_step": "define x", "confidence": 0.8}
    d.update(over)
    return json.dumps(d)


def test_parse_analysis_valid_and_clamped():
    a = parse_analysis(ai_json(confidence=3))
    assert a.confidence == 1.0 and a.source == "ai"


@pytest.mark.parametrize("raw", ["nope", ai_json(extra=1), ai_json(evidence="x"), ai_json(confidence="hi"),
                                 ai_json(observation=" "), ai_json(confidence=True)])
def test_parse_analysis_rejects(raw):
    with pytest.raises(Exception):
        parse_analysis(raw)


def test_analyst_falls_back_to_labelled_local(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = Analyst().analyze(ctx())
    assert a.source.startswith("diagnostics only") and "OPENAI_API_KEY" in a.source
    assert "line 50" in a.observation and any("x is undefined" in e for e in a.evidence)


def test_analyst_sends_file_text_only_no_images():
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ai_json()))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    a = Analyst(client).analyze(ctx())
    text = seen["messages"][1]["content"]
    assert a.source == "ai" and isinstance(text, str)   # text only: the screenshot is never sent
    assert "x is undefined" in text and "   50  line50" in text and "Selected text" in text


def test_local_analysis_without_anything():
    a = local_analysis(Context(None, None), "demo")
    assert a.confidence <= 0.1 and "demo" in a.source and "VS Code" in a.observation


def test_prompt_is_propose_only_and_carries_the_whole_file():
    t = AgentTask.from_context(local_analysis(ctx(), "x"), ctx())
    p = build_prompt(t)
    assert "DO NOT modify files" in p and "unified diff" in p
    assert "C:/repo/app.py" in p and "x is undefined" in p
    assert "    1  line1" in p and "  100  line100" in p      # full file, not an excerpt
    assert "Developer's selection" in p and t.workspace == "C:/repo"


def test_large_file_truncates_file_not_instructions():
    big = Context(None, ActiveFile("C:/repo/big.py", "python", "x = 1\n" * 20_000, workspace="C:/repo"))
    p = build_prompt(AgentTask.from_context(local_analysis(big, "x"), big))
    assert len(p) <= agents.MAX_PROMPT_CHARS and "unified diff" in p and "DO NOT modify" in p


class FakeRun:
    def __init__(self, code=0, out="ROOT CAUSE ...", err=""):
        self.calls, self.result = [], SimpleNamespace(returncode=code, stdout=out, stderr=err)

    def __call__(self, args, **kw):
        self.calls.append((args, kw))
        return self.result


def test_claude_adapter_is_read_only(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(agents.shutil, "which", lambda exe: f"C:/bin/{exe}.exe")
    monkeypatch.setattr(agents.subprocess, "run", run)
    res = ClaudeCodeAgent().run(AgentTask.from_context(local_analysis(ctx(), "x"), ctx()))
    args, kw = run.calls[0]
    assert res.text == "ROOT CAUSE ..."
    assert args[:2] == ["C:/bin/claude.exe", "-p"]
    assert args[args.index("--permission-mode") + 1] == "plan"
    assert args[args.index("--tools") + 1] == "Read,Grep,Glob"
    assert "bypassPermissions" not in args and "--dangerously-skip-permissions" not in args
    assert "--add-dir" not in args
    assert "x is undefined" in kw["input"] and "  100  line100" in kw["input"]   # the file goes on stdin


def test_hermes_adapter_planning_toolset_only(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(agents.shutil, "which", lambda exe: exe)
    monkeypatch.setattr(agents.subprocess, "run", run)
    HermesAgent().run(AgentTask.from_context(local_analysis(ctx(), "x"), ctx()))
    args = run.calls[0][0]
    assert args[1] == "-z" and args[args.index("-t") + 1] == "todo"
    assert "--yolo" not in args
    assert "  100  line100" in args[2]   # Hermes has no file tools, so the file is inline


def test_cli_failures_raise_agent_error(monkeypatch):
    monkeypatch.setattr(agents.shutil, "which", lambda exe: exe)
    monkeypatch.setattr(agents.subprocess, "run", FakeRun(code=1, out="", err="boom\nauth required"))
    with pytest.raises(AgentError, match="auth required"):
        ClaudeCodeAgent().run(AgentTask.from_context(local_analysis(ctx(), "x"), ctx()))

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("claude", 1)
    monkeypatch.setattr(agents.subprocess, "run", timeout)
    with pytest.raises(AgentError, match="timed out"):
        ClaudeCodeAgent().run(AgentTask.from_context(local_analysis(ctx(), "x"), ctx()))


def test_pick_agent(monkeypatch):
    monkeypatch.setattr(agents.shutil, "which", lambda exe: None)
    assert pick_agent("claude").name == "clipboard"
    monkeypatch.setattr(agents.shutil, "which", lambda exe: exe if exe == "hermes" else None)
    assert pick_agent().name == "hermes"
    monkeypatch.setenv("FLOWSTATE_AGENT", "clipboard")
    assert pick_agent().name == "clipboard"
