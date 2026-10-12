"""Visual encoder ``E_v``: screenshots ``[B, 3, H, W]`` -> patch tokens ``[B, N, D]``.

Two implementations share the same output contract:

* :class:`VisualEncoder` - a small from-scratch ViT (patch conv, learned 2D
  row/column positional embeddings, ``nn.TransformerEncoder``).
* :class:`PretrainedViTEncoder` - torchvision ``vit_b_16`` patch tokens (not the
  pooled class token) from a frozen backbone, plus a trainable projector.
  torchvision is imported lazily so SOMA works without it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from soma.config import ModelConfig
from soma.models.attention import transformer_encoder


class VisualEncoder(nn.Module):
    """From-scratch patch transformer.

    The grid is ``image_size // patch_size`` on each side; smaller inputs use a
    prefix of the positional tables, larger inputs are rejected.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.image_size % cfg.patch_size:
            raise ValueError("image_size must be a multiple of patch_size")
        D = cfg.d_model
        self.patch_size = cfg.patch_size
        self.grid = cfg.image_size // cfg.patch_size
        self.patch_embed = nn.Conv2d(3, D, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.row_embed = nn.Embedding(self.grid, D)
        self.col_embed = nn.Embedding(self.grid, D)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = transformer_encoder(D, cfg.n_heads, cfg.vision_layers, cfg.dropout)

    @property
    def num_tokens(self) -> int:
        """``N`` for a full-size ``image_size`` input."""
        return self.grid * self.grid

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``images [B, 3, H, W]`` into visual tokens ``[B, N, D]`` with ``N = (H/p)(W/p)``."""
        x = self.patch_embed(images)                                  # [B, D, gh, gw]
        B, D, gh, gw = x.shape
        if gh > self.grid or gw > self.grid:
            raise ValueError(f"image grid {gh}x{gw} exceeds configured {self.grid}x{self.grid}")
        rows = self.row_embed(torch.arange(gh, device=images.device))  # [gh, D]
        cols = self.col_embed(torch.arange(gw, device=images.device))  # [gw, D]
        pos = rows[:, None, :] + cols[None, :, :]                      # [gh, gw, D]
        x = x.permute(0, 2, 3, 1) + pos.to(x.dtype)                    # [B, gh, gw, D]
        x = self.dropout(x.reshape(B, gh * gw, D))
        return self.encoder(x)


class PretrainedViTEncoder(nn.Module):
    """Frozen torchvision ``vit_b_16`` patch tokens + trainable ``Linear``/``LayerNorm`` projector.

    ``weights`` is passed to ``torchvision.models.vit_b_16``: ``None`` builds a
    randomly initialised backbone (no download; used by tests and when loading a
    SOMA checkpoint that already contains the backbone weights), ``"DEFAULT"``
    loads the licensed ImageNet weights. Inputs of any size are resized to the
    backbone's 224x224.
    """

    def __init__(self, cfg: ModelConfig, weights: Any = None) -> None:
        super().__init__()
        try:
            from torchvision.models import vit_b_16
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("PretrainedViTEncoder needs torchvision; install it or use "
                              "vision_backbone='scratch'") from exc
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        hidden = self.backbone.hidden_dim
        self.input_size = self.backbone.image_size
        self.projector = nn.Sequential(nn.Linear(hidden, cfg.d_model), nn.LayerNorm(cfg.d_model))

    def train(self, mode: bool = True) -> "PretrainedViTEncoder":
        """Keep the frozen backbone in eval mode regardless of ``mode``."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``images [B, 3, H, W]`` into ``[B, 196, D]`` patch tokens."""
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False)
        vit = self.backbone
        with torch.no_grad():
            x = vit._process_input(images)                             # [B, 196, 768]
            cls = vit.class_token.expand(x.shape[0], -1, -1)
            x = vit.encoder(torch.cat([cls, x], dim=1))[:, 1:]         # drop class token
        return self.projector(x)


def build_visual_encoder(cfg: ModelConfig, *, pretrained: bool = True) -> nn.Module:
    """Pick the visual encoder named by ``cfg.vision_backbone``.

    ``pretrained=False`` builds the torchvision backbone without downloading
    weights (for checkpoint loading, where the state dict supplies them).
    """
    if cfg.vision_backbone == "scratch":
        return VisualEncoder(cfg)
    if cfg.vision_backbone == "torchvision_vit_b_16":
        return PretrainedViTEncoder(cfg, weights="DEFAULT" if pretrained else None)
    raise ValueError(f"unknown vision_backbone {cfg.vision_backbone!r}")
