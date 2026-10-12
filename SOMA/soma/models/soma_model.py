"""``SomaModel``: wires the six modules of build plan section 5.

images ──E_v──┐
              ├─ScreenFusion─> screen_state ─Temporal─> temporal_state ─MemoryFusion─> context ─Policy─> logits [B, A+1]
OCR ────E_t──┘
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from soma.config import ModelConfig
from soma.models.action_policy import ActionPolicy
from soma.models.batch import SomaBatch
from soma.models.memory_fusion import MemoryFusion
from soma.models.screen_fusion import ScreenFusion
from soma.models.temporal_encoder import TemporalEncoder
from soma.models.text_layout_encoder import TextLayoutEncoder
from soma.models.visual_encoder import build_visual_encoder

OUTPUT_KEYS = ("screen_state", "temporal_state", "personalized_context", "action_logits",
               "memory_gate", "memory_attention")


class SomaModel(nn.Module):
    """Screen -> temporal -> memory -> policy model with a ``NO_SUGGESTION`` output at index 0."""

    def __init__(self, vision_encoder: nn.Module, text_layout_encoder: TextLayoutEncoder,
                 screen_fusion: ScreenFusion, temporal_encoder: TemporalEncoder,
                 memory_fusion: MemoryFusion, action_policy: ActionPolicy,
                 cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = vision_encoder
        self.text_layout_encoder = text_layout_encoder
        self.screen_fusion = screen_fusion
        self.temporal_encoder = temporal_encoder
        self.memory_fusion = memory_fusion
        self.action_policy = action_policy

    def encode_screen(self, images: torch.Tensor, ocr_ids: torch.Tensor, ocr_boxes: torch.Tensor,
                      ocr_padding_mask: torch.Tensor | None = None,
                      return_tokens: bool = False
                      ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Screen state ``[B, D]`` for one screenshot + OCR (what ``history`` stores per step).

        With ``return_tokens=True`` returns ``(state, fused_tokens, fused_padding_mask)``.
        """
        if ocr_padding_mask is None:
            ocr_padding_mask = ocr_ids.eq(0)
        visual = self.vision_encoder(images)                                        # [B, N, D]
        text = self.text_layout_encoder(ocr_ids, ocr_boxes, ocr_padding_mask)       # [B, L, D]
        return self.screen_fusion(visual, text, ocr_padding_mask, return_tokens=return_tokens)

    def encode_temporal(self, batch: SomaBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """``(screen_state [B, D], temporal_state [B, D])`` without memory or candidates."""
        screen_state = self.encode_screen(batch.images, batch.ocr_ids, batch.ocr_boxes,
                                          batch.ocr_padding_mask)
        temporal_state = self.temporal_encoder(screen_state, batch.history,
                                               batch.history_padding_mask, batch.app_ids)
        return screen_state, temporal_state

    def query_embedding(self, batch: SomaBatch, normalize: bool = True) -> torch.Tensor:
        """Memory-retrieval query ``[B, D]`` from fused screen + history (L2-normalised by default)."""
        _, temporal_state = self.encode_temporal(batch)
        return F.normalize(temporal_state, dim=-1) if normalize else temporal_state

    def forward(self, batch: SomaBatch) -> dict[str, torch.Tensor]:
        """Run the full model.

        Returns a dict with ``screen_state [B, D]``, ``temporal_state [B, D]``,
        ``personalized_context [B, D]``, ``action_logits [B, A+1]`` (index 0 =
        ``NO_SUGGESTION``), ``memory_gate [B]`` and ``memory_attention [B, K+1]``
        (index 0 = ``NO_MEMORY``).
        """
        screen_state, temporal_state = self.encode_temporal(batch)
        mem = self.memory_fusion(temporal_state, batch.memory_embeddings, batch.memory_padding_mask)
        logits = self.action_policy(mem.context, batch.candidate_embeddings,
                                    batch.candidate_types, batch.candidate_mask)
        return {
            "screen_state": screen_state,
            "temporal_state": temporal_state,
            "personalized_context": mem.context,
            "action_logits": logits,
            "memory_gate": mem.gate,
            "memory_attention": mem.attention,
        }


def build_model(cfg: ModelConfig, *, pretrained: bool = True) -> SomaModel:
    """Build a :class:`SomaModel` from ``cfg``.

    ``pretrained`` only matters for ``vision_backbone="torchvision_vit_b_16"``;
    ``False`` skips the weight download (checkpoint loading supplies weights).
    """
    if cfg.d_model % cfg.n_heads:
        raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
    return SomaModel(
        vision_encoder=build_visual_encoder(cfg, pretrained=pretrained),
        text_layout_encoder=TextLayoutEncoder(cfg),
        screen_fusion=ScreenFusion(cfg),
        temporal_encoder=TemporalEncoder(cfg),
        memory_fusion=MemoryFusion(cfg),
        action_policy=ActionPolicy(cfg),
        cfg=cfg,
    )
