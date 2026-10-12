"""Memory fusion: gated cross-attention from the temporal state to retrieved memories.

A learned ``NO_MEMORY`` key/value is always prepended and never masked, so an
empty retrieval attends fully to it instead of softmaxing over nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from soma.config import ModelConfig


@dataclass
class MemoryFusionOutput:
    """Explainable output of :class:`MemoryFusion`."""

    context: torch.Tensor     # [B, D] personalized context
    gate: torch.Tensor        # [B] how much memory was let through, in (0, 1)
    attention: torch.Tensor   # [B, K+1] weights; index 0 = NO_MEMORY


class MemoryFusion(nn.Module):
    """``nn.MultiheadAttention`` (state -> memories) with a sigmoid gate and FFN residual."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.d_model
        self.no_memory = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.memory_proj = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        self.query_norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.gate = nn.Linear(2 * D, 1)
        self.norm1 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(4 * D, D))
        self.norm2 = nn.LayerNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, state: torch.Tensor, memory_embeddings: torch.Tensor | None = None,
                memory_padding_mask: torch.Tensor | None = None) -> MemoryFusionOutput:
        """Fuse ``state [B, D]`` with ``memory_embeddings [B, K, D]``.

        ``memory_padding_mask [B, K]`` (True = PAD) hides empty slots; ``K`` may
        be 0. Returns context ``[B, D]``, gate ``[B]`` and attention ``[B, K+1]``.
        """
        B, D = state.shape
        if memory_embeddings is None:
            memory_embeddings = state.new_zeros(B, 0, D)
        K = memory_embeddings.shape[1]
        if memory_padding_mask is None:
            memory_padding_mask = torch.zeros(B, K, dtype=torch.bool, device=state.device)
        mem = self.memory_proj(memory_embeddings.to(state.dtype))
        mem = mem.masked_fill(memory_padding_mask.bool().unsqueeze(-1), 0.0)
        kv = torch.cat([self.no_memory.to(mem.dtype).expand(B, 1, D), mem], dim=1)   # [B, K+1, D]
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=state.device),
                          memory_padding_mask.bool()], dim=1)
        q = self.query_norm(state).unsqueeze(1)                                       # [B, 1, D]
        attended, weights = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=True,
                                      average_attn_weights=True)
        attended, weights = attended[:, 0], weights[:, 0]                             # [B, D], [B, K+1]
        gate = torch.sigmoid(self.gate(torch.cat([state, attended], dim=-1)))         # [B, 1]
        x = self.norm1(state + self.dropout(gate * attended))
        context = self.norm2(x + self.dropout(self.ffn(x)))
        return MemoryFusionOutput(context=context, gate=gate[:, 0], attention=weights)
