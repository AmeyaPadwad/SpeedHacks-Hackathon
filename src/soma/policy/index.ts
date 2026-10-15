/**
 * SOMA policy: candidates, encoding, grounding, safety gate, approval ledger (port of soma/policy/__init__.py).
 *
 * Flow per meaningful screen change:
 *
 *     cands   = generateCandidates(obs, memories, cfg.model.maxCandidates, authorizedFile)
 *     cands   = addContextCandidates(obs, cands, A)            // when policy.contextTemplates
 *     blocked = SafetyGate.blockedMask(cands, authorizedFile, [], memoriesById)
 *     enc     = encoder.encode(cands, memoriesById, A, D, blocked)
 *     logits  = model.step(...)                                // [1 + A], index 0 = NO_SUGGESTION
 *     sugg    = gate.decide(logits, cands, obs, { blocked })
 *
 * Suggestions are never actions: see `ApprovalLedger`.
 */
export * from "./candidateEncoder";
export * from "./candidateGenerator";
export * from "./contextCandidates";
export * from "./decision";
export * from "./grounding";
export * from "./safetyGate";
