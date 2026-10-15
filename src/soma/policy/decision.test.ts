import { describe, expect, it } from "vitest";
import { ActionType, makeCandidate, type Suggestion } from "../types";
import { ApprovalLedger, ExecutionNotApproved, NoExecutorError, StaleRequestError } from "./decision";

const shown = (): Suggestion => ({
  candidate: makeCandidate({ action: ActionType.EXPLAIN_ERROR, text: "Explain" }),
  confidence: 0.9, screenVersion: 11, reason: "", distribution: {}, id: "sug_1", createdMs: 0,
});

describe("ApprovalLedger", () => {
  it("an approved request still never executes", () => {
    const ledger = new ApprovalLedger();
    const req = ledger.propose(shown());
    const tok = ledger.approve(req.id, { userConfirmed: true });
    expect(() => ledger.requestExecution(req.id, tok.token, 11)).toThrow(NoExecutorError);
    expect(ledger.executors.size).toBe(0);
    expect(ledger.executedCount).toBe(0);
    expect(ledger.unapprovedExecutionCount).toBe(0);
    expect(ledger.auditLog.map((a) => a.event)).toEqual(["proposed", "approved", "no_executor"]);
  });

  it("refuses registration, silent proposals, missing confirmation, bad and reused tokens", () => {
    const ledger = new ApprovalLedger();
    expect(() => ledger.registerExecutor(ActionType.EXPLAIN_ERROR, () => 1)).toThrow(ExecutionNotApproved);
    expect(() => ledger.propose({ ...shown(), candidate: null })).toThrow();
    const req = ledger.propose(shown());
    expect(() => ledger.approve(req.id, { userConfirmed: false })).toThrow(ExecutionNotApproved);
    expect(() => ledger.requestExecution(req.id, null)).toThrow(ExecutionNotApproved);
    const tok = ledger.approve(req.id, { userConfirmed: true });
    expect(() => ledger.requestExecution(req.id, "wrong")).toThrow(ExecutionNotApproved);
    expect(() => ledger.requestExecution(req.id, tok.token, 12)).toThrow(StaleRequestError);
    expect(() => ledger.requestExecution(req.id, tok.token, 11)).toThrow(/already used/);
  });

  it("expired tokens are refused; an enabled ledger runs only approved requests", () => {
    let t = 0;
    const ledger = new ApprovalLedger(() => t, 1000, true);
    let ran = 0;
    ledger.registerExecutor(ActionType.EXPLAIN_ERROR, () => ++ran);
    const req = ledger.propose(shown());
    const tok = ledger.approve(req.id, { userConfirmed: true });
    t = 5000;
    expect(() => ledger.requestExecution(req.id, tok.token)).toThrow(/expired/);
    const req2 = ledger.propose(shown());
    const tok2 = ledger.approve(req2.id, { userConfirmed: true });
    expect(ledger.requestExecution(req2.id, tok2.token)).toBe(1);
    expect(ledger.unapprovedExecutionCount).toBe(0);
  });
});
