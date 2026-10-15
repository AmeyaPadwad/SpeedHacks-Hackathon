/**
 * Suggestions are not actions: the explicit, audited approval contract (port of SOMA/soma/policy/decision.py).
 *
 * SOMA never clicks, types, runs commands, edits files or communicates. This module makes that boundary
 * explicit in code:
 * - A shown `Suggestion` can become an `ActionRequest`; that is all.
 * - Executing a request requires a single-use `ApprovalToken`, only issued when the user explicitly
 *   confirms that specific request.
 * - By default the ledger has **no executors**, and registering one is refused unless the ledger was
 *   deliberately built with `executorsEnabled`. So even an approved request throws `NoExecutorError`.
 * - Every proposal, approval, refusal and execution attempt is appended to an in-memory audit trail (ids
 *   and action names only, no screen content).
 *
 * `unapprovedExecutionCount` counts executor invocations without a valid approval; the structure of
 * `requestExecution` makes it impossible to increment, and tests assert it stays 0.
 */
import { ActionType, candidateKey, newId, nowMs, type Suggestion } from "../types";

export type Executor = (request: ActionRequest) => unknown;

export const DEFAULT_APPROVAL_TTL_MS = 60_000;

/** Thrown when execution is requested without a valid, unused approval. */
export class ExecutionNotApproved extends Error {
  override name = "ExecutionNotApproved";
}

/** Thrown when the screen changed since the request was proposed. */
export class StaleRequestError extends ExecutionNotApproved {
  override name = "StaleRequestError";
}

/** Thrown when no executor is registered for an action (always, in this release). */
export class NoExecutorError extends Error {
  override name = "NoExecutorError";
}

/** An inert description of something the user *might* approve. */
export interface ActionRequest {
  readonly action: ActionType;
  readonly candidateKey: string;
  readonly suggestionId: string;
  readonly screenVersion: number;
  readonly id: string;
  readonly createdMs: number;
}

/** Single-use, short-lived approval for exactly one request. */
export interface ApprovalToken {
  requestId: string;
  token: string;
  issuedMs: number;
  expiresMs: number;
  used: boolean;
}

export type AuditEvent = "proposed" | "approved" | "refused" | "executed" | "no_executor" | "denied_registration";

/** One audit record. Ids and action names only: never screen text. */
export interface AuditEntry {
  tsMs: number;
  event: AuditEvent;
  requestId: string;
  action: string;
  detail: string;
}

function randomToken(): string {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Constant-time string comparison. */
function safeEqual(a: string, b: string): boolean {
  let diff = a.length ^ b.length;
  for (let i = 0; i < Math.max(a.length, b.length); i++) diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  return diff === 0;
}

/** Approval + audit trail around a (by default empty) executor registry. */
export class ApprovalLedger {
  private readonly requests = new Map<string, ActionRequest>();
  private readonly tokens = new Map<string, ApprovalToken>();
  private readonly registry = new Map<ActionType, Executor>();
  private readonly audit: AuditEntry[] = [];
  private executed = 0;
  private readonly unapprovedExecuted = 0;

  constructor(
    private readonly clockMs: () => number = nowMs,
    private readonly approvalTtlMs = DEFAULT_APPROVAL_TTL_MS,
    private readonly executorsEnabled = false,
  ) {}

  get executors(): ReadonlyMap<ActionType, Executor> {
    return new Map(this.registry);
  }

  get auditLog(): AuditEntry[] {
    return [...this.audit];
  }

  get executedCount(): number {
    return this.executed;
  }

  get unapprovedExecutionCount(): number {
    return this.unapprovedExecuted;
  }

  private log(event: AuditEvent, requestId: string, action: string, detail = ""): void {
    this.audit.push({ tsMs: this.clockMs(), event, requestId, action, detail });
  }

  /** Refused unless the ledger was explicitly built with executors enabled. */
  registerExecutor(action: ActionType, fn: Executor): void {
    if (!this.executorsEnabled) {
      this.log("denied_registration", "", action, "executors disabled");
      throw new ExecutionNotApproved("SOMA has no executors: suggestions are not actions.");
    }
    if (action === ActionType.NO_SUGGESTION) throw new RangeError("NO_SUGGESTION cannot have an executor");
    this.registry.set(action, fn);
  }

  /** Turn a shown suggestion into an inert request (no side effects). */
  propose(suggestion: Suggestion): ActionRequest {
    if (suggestion.candidate === null) throw new RangeError("a silent suggestion cannot become an action request");
    const req: ActionRequest = Object.freeze({
      action: suggestion.candidate.action,
      candidateKey: candidateKey(suggestion.candidate),
      suggestionId: suggestion.id,
      screenVersion: suggestion.screenVersion,
      id: newId("req_"),
      createdMs: nowMs(),
    });
    this.requests.set(req.id, req);
    this.log("proposed", req.id, req.action);
    return req;
  }

  /** Issue a single-use token. `userConfirmed` must be an explicit `true`. */
  approve(requestId: string, { userConfirmed }: { userConfirmed: boolean }): ApprovalToken {
    const req = this.requests.get(requestId);
    if (!req) throw new RangeError(`unknown request ${requestId}`);
    if (userConfirmed !== true) {
      this.log("refused", requestId, req.action, "approval without user confirmation");
      throw new ExecutionNotApproved("approval requires an explicit user confirmation");
    }
    const now = this.clockMs();
    const tok: ApprovalToken = { requestId, token: randomToken(), issuedMs: now, expiresMs: now + this.approvalTtlMs, used: false };
    this.tokens.set(requestId, tok);
    this.log("approved", requestId, req.action);
    return tok;
  }

  /**
   * Run the executor for an approved request. Throws `ExecutionNotApproved` without a valid, unused,
   * unexpired token for this exact request, `StaleRequestError` if `currentVersion` differs from the
   * request's screen version, and `NoExecutorError` when no executor is registered (always, here).
   */
  requestExecution(requestId: string, token: string | null = null, currentVersion: number | null = null): unknown {
    const req = this.requests.get(requestId);
    const action = req ? req.action : "?";
    const tok = this.tokens.get(requestId);
    const now = this.clockMs();
    let problem: string | null = null;
    if (!req) problem = "unknown request";
    else if (!token || !tok) problem = "no approval token";
    else if (!safeEqual(tok.token, token)) problem = "token does not match this request";
    else if (tok.used) problem = "approval token already used";
    else if (now > tok.expiresMs) problem = "approval token expired";
    if (problem || !req || !tok) {
      this.log("refused", requestId, action, problem ?? "");
      throw new ExecutionNotApproved(`execution refused: ${problem}`);
    }
    tok.used = true; // single use, even if what follows fails
    if (currentVersion !== null && currentVersion !== req.screenVersion) {
      this.log("refused", requestId, action, "screen changed since proposal");
      throw new StaleRequestError("execution refused: the screen changed since the suggestion");
    }
    const fn = this.registry.get(req.action);
    if (!fn) {
      this.log("no_executor", requestId, action, "suggestions are not actions");
      throw new NoExecutorError(`no executor registered for ${action}; SOMA only suggests`);
    }
    this.executed++;
    this.log("executed", requestId, action);
    return fn(req);
  }
}
