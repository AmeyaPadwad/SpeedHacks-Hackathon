"""Suggestions are not actions: nothing executes without explicit approval, and v0.1 has no executors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from soma.policy.decision import (
    ActionRequest,
    ApprovalLedger,
    ExecutionNotApproved,
    NoExecutorError,
    StaleRequestError,
)
from soma.types import ActionType, Candidate, Suggestion

POLICY_DIR = Path(__file__).resolve().parent.parent / "soma" / "policy"


def shown(version: int = 3) -> Suggestion:
    c = Candidate(ActionType.OPEN_RELEVANT_FILE, "Open calc.py at line 40", file_ref="src/calc.py")
    return Suggestion(candidate=c, confidence=0.9, screen_version=version)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000

    def __call__(self) -> int:
        return self.t


def test_default_ledger_has_no_executors_and_refuses_registration():
    ledger = ApprovalLedger()
    assert ledger.executors == {}
    with pytest.raises(PermissionError):
        ledger.register_executor(ActionType.OPEN_RELEVANT_FILE, lambda req: None)
    assert ledger.executors == {}
    assert ledger.audit_log[-1].event == "denied_registration"


def test_unapproved_execution_raises():
    ledger = ApprovalLedger()
    req = ledger.propose(shown())
    assert isinstance(req, ActionRequest)
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(req.id)
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(req.id, token="guess")
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution("req_unknown", token="x")
    assert ledger.unapproved_execution_count == 0
    assert ledger.executed_count == 0
    assert [e.event for e in ledger.audit_log] == ["proposed", "refused", "refused", "refused"]


def test_approval_needs_explicit_user_confirmation():
    ledger = ApprovalLedger()
    req = ledger.propose(shown())
    for bad in (False, 1, "yes", None):
        with pytest.raises(ExecutionNotApproved):
            ledger.approve(req.id, user_confirmed=bad)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        ledger.approve("req_missing", user_confirmed=True)


def test_approved_request_still_has_no_executor_in_v01():
    ledger = ApprovalLedger()
    req = ledger.propose(shown())
    tok = ledger.approve(req.id, user_confirmed=True)
    with pytest.raises(NoExecutorError):
        ledger.request_execution(req.id, token=tok.token)
    assert ledger.executed_count == 0
    assert ledger.unapproved_execution_count == 0
    # token is single use
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(req.id, token=tok.token)


def test_token_is_bound_to_its_request():
    ledger = ApprovalLedger()
    a, b = ledger.propose(shown()), ledger.propose(shown())
    tok_a = ledger.approve(a.id, user_confirmed=True)
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(b.id, token=tok_a.token)


def test_silent_suggestion_cannot_be_proposed():
    with pytest.raises(ValueError):
        ApprovalLedger().propose(Suggestion(candidate=None, confidence=0.9, screen_version=1))


def test_enabled_ledger_requires_approval_expiry_and_fresh_screen():
    """A future, explicitly enabled ledger still gates every call."""
    clock = Clock()
    calls: list[str] = []
    ledger = ApprovalLedger(clock_ms=clock, approval_ttl_ms=100, executors_enabled=True)
    ledger.register_executor(ActionType.OPEN_RELEVANT_FILE, lambda req: calls.append(req.id) or "ok")
    with pytest.raises(ValueError):
        ledger.register_executor(ActionType.NO_SUGGESTION, lambda req: None)

    req = ledger.propose(shown(version=3))
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(req.id)
    assert calls == []

    tok = ledger.approve(req.id, user_confirmed=True)
    clock.t += 500
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(req.id, token=tok.token)  # expired

    tok = ledger.approve(req.id, user_confirmed=True)
    with pytest.raises(StaleRequestError):
        ledger.request_execution(req.id, token=tok.token, current_version=4)
    assert calls == []

    tok = ledger.approve(req.id, user_confirmed=True)
    assert ledger.request_execution(req.id, token=tok.token, current_version=3) == "ok"
    assert calls == [req.id]
    assert ledger.executed_count == 1
    assert ledger.unapproved_execution_count == 0
    assert ledger.audit_log[-1].event == "executed"


def test_audit_log_has_no_suggestion_text():
    ledger = ApprovalLedger()
    req = ledger.propose(shown())
    with pytest.raises(ExecutionNotApproved):
        ledger.request_execution(req.id)
    for entry in ledger.audit_log:
        assert "calc.py" not in f"{entry.detail}{entry.action}{entry.request_id}"


_FORBIDDEN_MODULES = {"subprocess", "pyautogui", "pynput", "keyboard", "mouse", "win32api",
                      "ctypes", "shutil", "socket", "requests", "urllib", "webbrowser"}
_FORBIDDEN_CALLS = {"system", "popen", "startfile", "exec", "eval", "spawn", "remove", "unlink", "rmtree"}


def test_policy_package_has_no_execution_primitives():
    """Static check: the policy package cannot click, type, run or delete anything."""
    for path in POLICY_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in _FORBIDDEN_MODULES, (path.name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in _FORBIDDEN_MODULES, (path.name, node.module)
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                assert name not in _FORBIDDEN_CALLS, (path.name, name)
