"""Shared masked-attention helpers.

Masks follow the PyTorch convention used across SOMA: ``True`` = PAD (ignored).
The central problem these helpers solve is the all-masked row: softmax over a
set of keys that are all ``-inf`` yields ``NaN``, and a ``NaN`` multiplied by a
zero attention weight downstream is still ``NaN``. Every SOMA attention path
either keeps one key visible (a learned null token, or the query's own slot) or
goes through :func:`safe_padding_mask` and zeroes the padded outputs afterwards.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Large negative logit used instead of ``-inf``. Representable in fp16
#: (max ~65504) and bf16, and ``exp(-1e4)`` underflows to exactly 0.
NEG_INF = -1e4


def all_masked_rows(padding_mask: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` bool mask -> ``[B]`` bool, True where every position is PAD."""
    if padding_mask.shape[-1] == 0:
        return torch.ones(padding_mask.shape[:-1], dtype=torch.bool, device=padding_mask.device)
    return padding_mask.all(dim=-1)


def safe_padding_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Unmask position 0 of rows that are entirely PAD.

    Lets ``nn.TransformerEncoder`` / ``nn.MultiheadAttention`` run without
    producing ``NaN`` for empty inputs. Callers must zero (or otherwise ignore)
    the outputs at the original padded positions.
    """
    empty = all_masked_rows(padding_mask)
    if not bool(empty.any()):
        return padding_mask
    mask = padding_mask.clone()
    mask[empty, 0] = False
    return mask


def zero_padded(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Zero the token vectors of ``x [B, S, D]`` where ``padding_mask [B, S]`` is True."""
    return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid tokens of ``x [B, S, D]`` -> ``[B, D]``; all-PAD rows give zeros."""
    keep = (~padding_mask).unsqueeze(-1).to(x.dtype)
    total = (x * keep).sum(dim=1)
    count = keep.sum(dim=1).clamp_min(1.0)
    return total / count


def masked_softmax(logits: torch.Tensor, padding_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that ignores PAD positions; an all-PAD row returns all zeros, not ``NaN``."""
    filled = logits.masked_fill(padding_mask, NEG_INF)
    probs = torch.softmax(filled.float(), dim=dim).to(logits.dtype)
    return probs.masked_fill(padding_mask, 0.0)


def additive_mask(padding_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bool PAD mask -> additive float mask (0 keep, ``NEG_INF`` drop) of ``dtype``."""
    return torch.zeros(padding_mask.shape, dtype=dtype, device=padding_mask.device).masked_fill(
        padding_mask, NEG_INF)


def causal_padding_attn_mask(padding_mask: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Per-sample causal mask that also hides PAD keys, ``[B * n_heads, S, S]`` bool.

    ``True`` = not allowed. Query ``i`` sees key ``j`` iff ``j <= i`` and key ``j``
    is valid, except that every query always sees itself, so no row is ever
    entirely masked (a padded query only attends to its own slot).
    """
    B, S = padding_mask.shape
    device = padding_mask.device
    future = torch.ones(S, S, dtype=torch.bool, device=device).triu(1)       # [S, S]
    blocked = future.unsqueeze(0) | padding_mask.unsqueeze(1)                # [B, S, S]
    eye = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
    blocked = blocked & ~eye
    return blocked.repeat_interleave(n_heads, dim=0)


class AttentionPool(nn.Module):
    """Learned-query attention pooling ``[B, S, D] -> [B, Q, D]`` (``[B, D]`` when Q == 1).

    Uses ``F.scaled_dot_product_attention`` with a learned null key/value that is
    always visible, so pooling over an all-PAD sequence is well defined.
    """

    def __init__(self, d_model: int, n_heads: int, n_queries: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model, self.n_heads, self.n_queries = d_model, n_heads, n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.null_kv = nn.Parameter(torch.zeros(1, d_model))
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = dropout

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool ``tokens [B, S, D]`` (``padding_mask [B, S]``, True = PAD) to ``[B, D]`` or ``[B, Q, D]``."""
        B, S, D = tokens.shape
        if padding_mask is None:
            padding_mask = torch.zeros(B, S, dtype=torch.bool, device=tokens.device)
        kv = torch.cat([self.null_kv.to(tokens.dtype).expand(B, 1, D), tokens], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device), padding_mask], dim=1)
        q = self._heads(self.queries.to(tokens.dtype).expand(B, -1, -1))
        k, v = self._heads(self.k_proj(kv)), self._heads(self.v_proj(kv))
        bias = additive_mask(mask, q.dtype)[:, None, None, :]                # [B, 1, 1, S+1]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, self.n_queries, D)
        out = self.norm(self.out_proj(out))
        return out[:, 0] if self.n_queries == 1 else out


def transformer_encoder(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.TransformerEncoder:
    """Pre-norm ``nn.TransformerEncoder`` with GELU and nested tensors disabled.

    Nested tensors are disabled so train and eval modes compute identical
    values for padded positions (and to avoid the fast-path warning).
    """
    layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                 enable_nested_tensor=False)
