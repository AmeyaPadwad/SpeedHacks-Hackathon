"""SOMA policy package: candidates, encoding, grounding, safety gate, approval ledger.

Flow per meaningful screen change::

    cands = generate_candidates(obs, memories, cfg.model.max_candidates, authorized_file)
    emb, types, mask = encoder.encode(cands, memories_by_id, A, D, blocked=SafetyGate.blocked_mask(...))
    logits = model(...)                      # [B, A+1], index 0 = NO_SUGGESTION
    suggestion = gate.decide(logits[0], cands, latest_obs, blocked=...)

Suggestions are never actions: see ``ApprovalLedger``.
"""

from soma.policy.candidate_encoder import CandidateEncoder, action_index
from soma.policy.candidate_generator import (
    deny_all_files,
    detect_role,
    generate_candidates,
    is_suspicious,
    sanitize_snippet,
)
from soma.policy.decision import (
    ActionRequest,
    ApprovalLedger,
    ApprovalToken,
    AuditEntry,
    ExecutionNotApproved,
    NoExecutorError,
    StaleRequestError,
)
from soma.policy.grounding import find_target, ground_box, ground_point, iou, is_stale, verify_target
from soma.policy.safety_gate import SafetyGate, distribution_labels, to_probs

__all__ = [
    "ActionRequest",
    "ApprovalLedger",
    "ApprovalToken",
    "AuditEntry",
    "CandidateEncoder",
    "ExecutionNotApproved",
    "NoExecutorError",
    "SafetyGate",
    "StaleRequestError",
    "action_index",
    "deny_all_files",
    "detect_role",
    "distribution_labels",
    "find_target",
    "generate_candidates",
    "ground_box",
    "ground_point",
    "iou",
    "is_stale",
    "is_suspicious",
    "sanitize_snippet",
    "to_probs",
    "verify_target",
]
