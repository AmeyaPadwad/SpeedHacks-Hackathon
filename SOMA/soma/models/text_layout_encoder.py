"""Text/layout encoder ``E_t``: OCR ids ``[B, L]`` + boxes ``[B, L, 4]`` -> tokens ``[B, L, D]``."""

from __future__ import annotations

import torch
from torch import nn

from soma.config import ModelConfig
from soma.models.attention import safe_padding_mask, transformer_encoder, zero_padded
from soma.models.tokenizer import PAD_ID


def box_features(boxes: torch.Tensor) -> torch.Tensor:
    """Normalized ``xyxy`` boxes ``[..., 4]`` -> ``[..., 6]`` = (x0, y0, x1, y1, w, h)."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([x0, y0, x1, y1, x1 - x0, y1 - y0], dim=-1)


class TextLayoutEncoder(nn.Module):
    """Token embedding + 2D box projection + ``nn.TransformerEncoder`` with key padding mask.

    Padded positions are zero in the output. A row with no OCR at all yields an
    all-zero ``[L, D]`` block (never ``NaN``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.d_model
        self.token_embed = nn.Embedding(cfg.vocab_size, D, padding_idx=PAD_ID)
        self.box_proj = nn.Linear(6, D)
        self.max_len = cfg.max_ocr_tokens
        self.pos_embed = nn.Embedding(cfg.max_ocr_tokens, D)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.norm = nn.LayerNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = transformer_encoder(D, cfg.n_heads, cfg.text_layers, cfg.dropout)

    def forward(self, ocr_ids: torch.Tensor, ocr_boxes: torch.Tensor,
                ocr_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``ocr_ids [B, L]``, ``ocr_boxes [B, L, 4]`` and ``ocr_padding_mask [B, L]``
        (True = PAD) into ``[B, L, D]``."""
        B, L = ocr_ids.shape
        if L > self.max_len:
            raise ValueError(f"{L} OCR tokens exceed max_ocr_tokens={self.max_len}")
        if ocr_padding_mask is None:
            ocr_padding_mask = ocr_ids.eq(PAD_ID)
        ocr_padding_mask = ocr_padding_mask.bool()
        if L == 0:
            return ocr_boxes.new_zeros(B, 0, self.token_embed.embedding_dim)
        pos = self.pos_embed(torch.arange(L, device=ocr_ids.device))
        x = self.token_embed(ocr_ids) + self.box_proj(box_features(ocr_boxes.to(pos.dtype))) + pos
        x = zero_padded(self.dropout(self.norm(x)), ocr_padding_mask)
        x = self.encoder(x, src_key_padding_mask=safe_padding_mask(ocr_padding_mask))
        return zero_padded(x, ocr_padding_mask)
