"""Temporal encoder: current screen state + recent history -> temporal state ``[B, D]``.

The sequence is ``history[0..T-1]`` followed by the current screen state as the
last step. Both variants are causal (``nn.GRU`` by construction, the transformer
through a causal mask) and ignore padded history steps. App identity enters as a
learned ``nn.Embedding`` added to the current step.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from soma.config import ModelConfig
from soma.models.attention import causal_padding_attn_mask, transformer_encoder, zero_padded


class TemporalEncoder(nn.Module):
    """Causal temporal model selected by ``cfg.temporal_kind`` (``"gru"`` | ``"transformer"``)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.temporal_kind not in ("gru", "transformer"):
            raise ValueError(f"unknown temporal_kind {cfg.temporal_kind!r}")
        D = cfg.d_model
        self.kind = cfg.temporal_kind
        self.n_heads = cfg.n_heads
        self.max_steps = cfg.history_len + 1
        self.app_embed = nn.Embedding(cfg.n_apps, D)
        nn.init.normal_(self.app_embed.weight, std=0.02)
        self.in_proj = nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))
        self.dropout = nn.Dropout(cfg.dropout)
        if self.kind == "gru":
            self.rnn = nn.GRU(D, D, num_layers=1, batch_first=True)
        else:
            # Step position counted back from "now", so the current step is always 0.
            self.age_embed = nn.Embedding(self.max_steps, D)
            nn.init.normal_(self.age_embed.weight, std=0.02)
            self.encoder = transformer_encoder(D, cfg.n_heads, max(1, cfg.fusion_layers), cfg.dropout)
        self.out_proj = nn.Linear(D, D)
        self.out_norm = nn.LayerNorm(D)

    # -- variants -------------------------------------------------------------------
    def _gru_steps(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Run the GRU over valid steps only; returns per-step outputs ``[B, S, D]``."""
        B, S, D = x.shape
        # Stable sort valid steps to the front so packing (which needs right padding)
        # works whatever the padding layout; order among valid steps is preserved.
        order = torch.argsort(pad.to(torch.int8), dim=1, stable=True)            # [B, S]
        aligned = torch.gather(x, 1, order.unsqueeze(-1).expand(B, S, D))
        lengths = (~pad).sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(aligned, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=S)       # [B, S, D]
        steps = torch.zeros_like(out).scatter(1, order.unsqueeze(-1).expand(B, S, D), out)
        return zero_padded(steps, pad)

    def _transformer_steps(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        ages = torch.arange(S - 1, -1, -1, device=x.device)
        x = x + self.age_embed(ages).to(x.dtype)
        mask = causal_padding_attn_mask(pad, self.n_heads)
        out = self.encoder(x, mask=mask)
        return zero_padded(out, pad)

    # -- public ---------------------------------------------------------------------
    def forward(self, screen_state: torch.Tensor, history: torch.Tensor | None = None,
                history_padding_mask: torch.Tensor | None = None,
                app_ids: torch.Tensor | None = None, return_steps: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode ``screen_state [B, D]`` with ``history [B, T, D]``.

        ``history_padding_mask [B, T]`` (True = PAD) hides missing steps; an
        all-padded history reduces to the current step alone. ``app_ids [B]``
        are hashed app buckets. Returns ``[B, D]``, or with ``return_steps=True``
        also the per-step causal outputs ``[B, T+1, D]`` (padded steps zeroed).
        """
        B, D = screen_state.shape
        if history is None:
            history = screen_state.new_zeros(B, 0, D)
        T = history.shape[1]
        if T + 1 > self.max_steps:
            history = history[:, -(self.max_steps - 1):]
            if history_padding_mask is not None:
                history_padding_mask = history_padding_mask[:, -(self.max_steps - 1):]
            T = history.shape[1]
        if history_padding_mask is None:
            history_padding_mask = torch.zeros(B, T, dtype=torch.bool, device=screen_state.device)
        current = screen_state
        if app_ids is not None:
            current = current + self.app_embed(app_ids).to(current.dtype)
        seq = torch.cat([history.to(current.dtype), current.unsqueeze(1)], dim=1)   # [B, T+1, D]
        pad = torch.cat([history_padding_mask.bool(),
                         torch.zeros(B, 1, dtype=torch.bool, device=seq.device)], dim=1)
        x = zero_padded(self.dropout(self.in_proj(seq)), pad)
        steps = self._gru_steps(x, pad) if self.kind == "gru" else self._transformer_steps(x, pad)
        last = steps[:, -1]
        state = self.out_norm(screen_state + self.dropout(self.out_proj(last)))
        if return_steps:
            return state, steps
        return state
