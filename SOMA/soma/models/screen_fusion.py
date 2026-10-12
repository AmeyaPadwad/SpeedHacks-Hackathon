"""Screen fusion ``F``: visual + text tokens -> screen state ``[B, D]`` (+ fused tokens)."""

from __future__ import annotations

import torch
from torch import nn

from soma.config import ModelConfig
from soma.models.attention import AttentionPool, transformer_encoder, zero_padded

VISUAL_TYPE = 0
TEXT_TYPE = 1


class ScreenFusion(nn.Module):
    """Concatenate visual and OCR tokens with modality-type embeddings, run a
    small transformer, then pool with a learned query.

    Visual tokens are never masked, so the attention always has valid keys
    even when OCR is empty.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.d_model
        self.type_embed = nn.Embedding(2, D)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.encoder = transformer_encoder(D, cfg.n_heads, cfg.fusion_layers, cfg.dropout)
        self.pool = AttentionPool(D, cfg.n_heads, n_queries=1, dropout=cfg.dropout)

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor,
                ocr_padding_mask: torch.Tensor, return_tokens: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse ``visual_tokens [B, N, D]`` and ``text_tokens [B, L, D]``.

        ``ocr_padding_mask [B, L]`` (True = PAD) hides padded OCR. Returns the
        screen state ``[B, D]``, or with ``return_tokens=True`` the tuple
        ``(state [B, D], fused_tokens [B, N+L, D], fused_padding_mask [B, N+L])``.
        """
        B, N, _ = visual_tokens.shape
        vis = visual_tokens + self.type_embed.weight[VISUAL_TYPE].to(visual_tokens.dtype)
        txt = text_tokens + self.type_embed.weight[TEXT_TYPE].to(text_tokens.dtype)
        txt = zero_padded(txt, ocr_padding_mask)
        tokens = torch.cat([vis, txt], dim=1)
        mask = torch.cat([torch.zeros(B, N, dtype=torch.bool, device=tokens.device),
                          ocr_padding_mask.bool()], dim=1)
        fused = zero_padded(self.encoder(tokens, src_key_padding_mask=mask), mask)
        state = self.pool(fused, mask)
        if return_tokens:
            return state, fused, mask
        return state
