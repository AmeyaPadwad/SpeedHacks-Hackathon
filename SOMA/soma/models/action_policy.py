"""Action policy / ranker: context + candidates -> logits ``[B, A+1]``.

Index 0 is the trainable ``NO_SUGGESTION`` logit; indices ``1..A`` score the
candidates. Blocked or padded candidates get :data:`BLOCKED_LOGIT`, so when all
candidates are blocked the softmax puts ~all mass on ``NO_SUGGESTION``.
"""

from __future__ import annotations

import torch
from torch import nn

from soma.config import ModelConfig
from soma.models.attention import NEG_INF

NO_SUGGESTION_INDEX = 0
#: Logit for blocked/padded candidates; finite and fp16-safe (see ``attention.NEG_INF``).
BLOCKED_LOGIT = NEG_INF


class ActionPolicy(nn.Module):
    """Pairwise ``nn.Sequential`` ranking head plus a context-conditioned abstain head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.d_model
        self.type_embed = nn.Embedding(cfg.n_action_types, D)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.cand_proj = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        self.ctx_proj = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        self.scorer = nn.Sequential(
            nn.Linear(3 * D, D), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1),
        )
        self.abstain = nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1))
        self.abstain_bias = nn.Parameter(torch.zeros(1))

    def forward(self, context: torch.Tensor, candidate_embeddings: torch.Tensor,
                candidate_types: torch.Tensor | None = None,
                candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Score ``candidate_embeddings [B, A, D]`` (``candidate_types [B, A]`` long,
        ``candidate_mask [B, A]`` True = blocked/PAD) against ``context [B, D]``.

        Returns logits ``[B, A+1]``; column 0 is ``NO_SUGGESTION``. ``A`` may be 0.
        """
        B, A, _ = candidate_embeddings.shape
        if candidate_mask is None:
            candidate_mask = torch.zeros(B, A, dtype=torch.bool, device=context.device)
        ctx = self.ctx_proj(context)                                              # [B, D]
        no_sugg = self.abstain(ctx) + self.abstain_bias                           # [B, 1]
        if A == 0:
            return no_sugg
        cand = candidate_embeddings.to(ctx.dtype)
        if candidate_types is not None:
            cand = cand + self.type_embed(candidate_types).to(ctx.dtype)
        cand = self.cand_proj(cand)                                               # [B, A, D]
        ctx_e = ctx.unsqueeze(1).expand_as(cand)
        scores = self.scorer(torch.cat([ctx_e, cand, ctx_e * cand], dim=-1))[..., 0]   # [B, A]
        scores = scores.masked_fill(candidate_mask.bool(), BLOCKED_LOGIT)
        return torch.cat([no_sugg.to(scores.dtype), scores], dim=1)
