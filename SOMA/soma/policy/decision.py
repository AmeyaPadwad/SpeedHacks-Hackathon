"""Suggestions are not actions: the explicit, audited approval contract.

SOMA v0.1 never clicks, types, runs commands, edits files or communicates.
This module makes that boundary explicit in code:

* A shown ``Suggestion`` can become an ``ActionRequest``; that is all.
* Executing a request requires a single-use ``ApprovalToken`` that is only
  issued when the user explicitly confirms that specific request.
* By default the ledger has **no executors**, and registering one is refused
  unless the ledger was deliberately constructed with ``executors_enabled``
  (a later, separately reviewed release). So even an approved request raises
  ``NoExecutorError`` in v0.1.
* Every proposal, approval, refusal and execution attempt is appended to an
  in-memory audit trail (ids and action names only, no screen content).

``unapproved_execution_count`` counts executor invocations that happened
without a valid approval. The structure of ``request_execution`` makes it
impossible to increment; tests assert it stays 0.
"""

from __future__ import annotations

import hmac
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from soma.types import ActionType, Suggestion, new_id, now_ms

Executor = Callable[["ActionRequest"], object]

DEFAULT_APPROVAL_TTL_MS = 60_000


class ExecutionNotApproved(PermissionError):
    """Raised when execution is requested without a valid, unused approval."""


class StaleRequestError(ExecutionNotApproved):
    """Raised when the screen changed since the request was proposed."""


class NoExecutorError(RuntimeError):
    """Raised when no executor is registered for an action (always, in v0.1)."""


@dataclass(frozen=True)
class ActionRequest:
    """An inert description of something the user *might* approve."""

    action: ActionType
    candidate_key: str
    suggestion_id: str
    screen_version: int
    id: str = field(default_factory=lambda: new_id("req_"))
    created_ms: int = field(default_factory=now_ms)


@dataclass
class ApprovalToken:
    """Single-use, short-lived approval for exactly one request."""

    request_id: str
    token: str
    issued_ms: int
    expires_ms: int
    used: bool = False


@dataclass(frozen=True)
class AuditEntry:
    """One audit record. Ids and action names only: never screen text."""

    ts_ms: int
    event: str          # proposed | approved | refused | executed | no_executor | denied_registration
    request_id: str
    action: str
    detail: str = ""


class ApprovalLedger:
    """Approval + audit trail around a (by default empty) executor registry."""

    def __init__(
        self,
        clock_ms: Callable[[], int] = now_ms,
        approval_ttl_ms: int = DEFAULT_APPROVAL_TTL_MS,
        executors_enabled: bool = False,
    ) -> None:
        self._clock = clock_ms
        self._ttl = approval_ttl_ms
        self._executors_enabled = executors_enabled
        self._lock = threading.Lock()
        self._requests: dict[str, ActionRequest] = {}
        self._tokens: dict[str, ApprovalToken] = {}
        self._executors: dict[ActionType, Executor] = {}
        self._audit: list[AuditEntry] = []
        self._executed = 0
        self._unapproved_executed = 0

    # ------------------------------------------------------------ introspection

    @property
    def executors(self) -> dict[ActionType, Executor]:
        with self._lock:
            return dict(self._executors)

    @property
    def audit_log(self) -> list[AuditEntry]:
        with self._lock:
            return list(self._audit)

    @property
    def executed_count(self) -> int:
        with self._lock:
            return self._executed

    @property
    def unapproved_execution_count(self) -> int:
        with self._lock:
            return self._unapproved_executed

    def _log(self, event: str, req_id: str, action: str, detail: str = "") -> None:
        self._audit.append(AuditEntry(self._clock(), event, req_id, action, detail))

    # ------------------------------------------------------------ registry

    def register_executor(self, action: ActionType, fn: Executor) -> None:
        """Refused unless the ledger was explicitly built with executors enabled."""
        with self._lock:
            if not self._executors_enabled:
                self._log("denied_registration", "", action.value, "executors disabled in v0.1")
                raise PermissionError("SOMA v0.1 has no executors: suggestions are not actions.")
            if action is ActionType.NO_SUGGESTION:
                raise ValueError("NO_SUGGESTION cannot have an executor")
            self._executors[action] = fn

    # ------------------------------------------------------------ flow

    def propose(self, suggestion: Suggestion) -> ActionRequest:
        """Turn a shown suggestion into an inert request (no side effects)."""
        if suggestion.candidate is None:
            raise ValueError("a silent suggestion cannot become an action request")
        req = ActionRequest(
            action=suggestion.candidate.action,
            candidate_key=suggestion.candidate.key,
            suggestion_id=suggestion.id,
            screen_version=suggestion.screen_version,
        )
        with self._lock:
            self._requests[req.id] = req
            self._log("proposed", req.id, req.action.value)
        return req

    def approve(self, request_id: str, *, user_confirmed: bool) -> ApprovalToken:
        """Issue a single-use token. ``user_confirmed`` must be an explicit ``True``."""
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise KeyError(f"unknown request {request_id!r}")
            if user_confirmed is not True:
                self._log("refused", request_id, req.action.value, "approval without user confirmation")
                raise ExecutionNotApproved("approval requires an explicit user confirmation")
            now = self._clock()
            tok = ApprovalToken(request_id, secrets.token_urlsafe(24), now, now + self._ttl)
            self._tokens[request_id] = tok
            self._log("approved", request_id, req.action.value)
            return tok

    def request_execution(
        self,
        request_id: str,
        token: str | None = None,
        current_version: int | None = None,
    ) -> object:
        """Run the executor for an approved request.

        Raises ``ExecutionNotApproved`` without a valid, unused, unexpired token
        for this exact request, ``StaleRequestError`` if ``current_version``
        differs from the request's screen version, and ``NoExecutorError`` when
        no executor is registered (always the case in v0.1).
        """
        with self._lock:
            req = self._requests.get(request_id)
            action = req.action.value if req else "?"
            tok = self._tokens.get(request_id)
            now = self._clock()
            problem = None
            if req is None:
                problem = "unknown request"
            elif not token or tok is None:
                problem = "no approval token"
            elif not hmac.compare_digest(tok.token.encode(), token.encode()):
                problem = "token does not match this request"
            elif tok.used:
                problem = "approval token already used"
            elif now > tok.expires_ms:
                problem = "approval token expired"
            if problem:
                self._log("refused", request_id, action, problem)
                raise ExecutionNotApproved(f"execution refused: {problem}")
            assert req is not None and tok is not None
            tok.used = True  # single use, even if what follows fails
            if current_version is not None and current_version != req.screen_version:
                self._log("refused", request_id, action, "screen changed since proposal")
                raise StaleRequestError("execution refused: the screen changed since the suggestion")
            fn = self._executors.get(req.action)
            if fn is None:
                self._log("no_executor", request_id, action, "suggestions are not actions in v0.1")
                raise NoExecutorError(f"no executor registered for {action}; SOMA only suggests")
            self._executed += 1
            self._log("executed", request_id, action)
        return fn(req)
