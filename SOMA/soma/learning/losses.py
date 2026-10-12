"""Training losses for the action policy (build plan section 8, Stage B).

All policy logits are ``[B, A + 1]`` with index 0 = ``NO_SUGGESTION``, so the
abstention option is trained like any other class and receives gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reduce(per_example: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return per_example.mean()
    w = weight.to(per_example.dtype)
    return (per_example * w).sum() / w.sum().clamp_min(1e-8)


def policy_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None,
                label_smoothing: float = 0.0) -> torch.Tensor:
    """Weighted cross-entropy over ``[B, A+1]`` logits including ``NO_SUGGESTION``.

    Blocked candidates should already be ``-inf`` in ``logits``; non-finite
    logits are clamped to a large negative value so smoothing stays finite.
    """
    logits = torch.nan_to_num(logits.float(), neginf=-1e4, posinf=1e4)
    per = F.cross_entropy(logits, target.long(), reduction="none", label_smoothing=label_smoothing)
    return _reduce(per, weight)


def retrieval_contrastive_loss(query: torch.Tensor, positives: torch.Tensor,
                               negatives: torch.Tensor, temperature: float = 0.07,
                               negative_mask: torch.Tensor | None = None) -> torch.Tensor:
    """InfoNCE: pull ``query [B,D]`` toward ``positives [B,D]``, away from ``negatives [B,N,D]``.

    Uses cosine similarity divided by ``temperature``. ``negative_mask``
    (``[B,N]``, True = PAD) drops padded negatives.
    """
    q = F.normalize(query.float(), dim=-1)
    p = F.normalize(positives.float(), dim=-1)
    n = F.normalize(negatives.float(), dim=-1)
    pos = (q * p).sum(-1, keepdim=True)                     # [B, 1]
    neg = torch.einsum("bd,bnd->bn", q, n)                   # [B, N]
    if negative_mask is not None:
        neg = neg.masked_fill(negative_mask.bool(), float("-inf"))
    logits = torch.cat([pos, neg], dim=1) / temperature
    target = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, target)


def pairwise_ranking_loss(scores: torch.Tensor, preferred_idx: torch.Tensor,
                          rejected_idx: torch.Tensor, margin: float = 1.0,
                          weight: torch.Tensor | None = None) -> torch.Tensor:
    """Hinge loss ``max(0, margin - (s_pref - s_rej))`` per row of ``scores [B, A+1]``."""
    s_pref = scores.gather(1, preferred_idx.long().view(-1, 1)).squeeze(1)
    s_rej = scores.gather(1, rejected_idx.long().view(-1, 1)).squeeze(1)
    per = F.relu(margin - (s_pref - s_rej))
    return _reduce(per, weight)


def total_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, *,
               label_smoothing: float = 0.0, lambda_retrieval: float = 0.0,
               retrieval: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
               temperature: float = 0.07, lambda_ranking: float = 0.0,
               ranking: tuple[torch.Tensor, torch.Tensor] | None = None,
               margin: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """``policy + lambda_retrieval * retrieval [+ lambda_ranking * ranking]``.

    The retrieval term is skipped when ``lambda_retrieval == 0`` or no
    ``(query, positives, negatives)`` supervision is given (plan: start at 0).
    Returns ``(loss, parts)`` with float parts for logging.
    """
    loss = policy_loss(logits, target, weight, label_smoothing)
    parts = {"policy": float(loss.detach())}
    if lambda_retrieval > 0.0 and retrieval is not None:
        r = retrieval_contrastive_loss(*retrieval, temperature=temperature)
        loss = loss + lambda_retrieval * r
        parts["retrieval"] = float(r.detach())
    if lambda_ranking > 0.0 and ranking is not None:
        rk = pairwise_ranking_loss(logits, ranking[0], ranking[1], margin)
        loss = loss + lambda_ranking * rk
        parts["ranking"] = float(rk.detach())
    parts["total"] = float(loss.detach())
    return loss, parts
