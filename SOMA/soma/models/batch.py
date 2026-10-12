"""Tensor contract for ``SomaModel`` (build plan section 5).

Masks follow PyTorch convention: ``True`` = PAD (ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class SomaBatch:
    images: torch.Tensor                 # [B, 3, H, W] float, normalized
    ocr_ids: torch.Tensor                # [B, L] long
    ocr_boxes: torch.Tensor              # [B, L, 4] float, normalized xyxy
    ocr_padding_mask: torch.Tensor       # [B, L] bool, True = PAD
    history: torch.Tensor                # [B, T, D] float, previous screen states
    history_padding_mask: torch.Tensor   # [B, T] bool, True = PAD
    app_ids: torch.Tensor                # [B] long, hashed app bucket
    memory_embeddings: torch.Tensor      # [B, K, D] float
    memory_padding_mask: torch.Tensor    # [B, K] bool, True = PAD
    candidate_embeddings: torch.Tensor   # [B, A, D] float
    candidate_types: torch.Tensor        # [B, A] long, ActionType index
    candidate_mask: torch.Tensor         # [B, A] bool, True = blocked / PAD

    def to(self, device: torch.device | str) -> "SomaBatch":
        return SomaBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])


def random_batch(B: int = 2, *, image_size: int = 64, L: int = 8, T: int = 4, K: int = 3,
                 A: int = 4, D: int = 64, vocab_size: int = 512, n_apps: int = 16,
                 n_action_types: int = 6, empty_ocr: bool = False,
                 empty_memory: bool = False, all_blocked: bool = False,
                 generator: torch.Generator | None = None) -> SomaBatch:
    """Random but well-formed batch for shape tests and smoke runs."""
    g = generator
    ocr_mask = torch.zeros(B, L, dtype=torch.bool)
    if empty_ocr:
        ocr_mask[:] = True
    else:
        ocr_mask[:, L // 2:] = True   # half padded
    mem_mask = torch.zeros(B, K, dtype=torch.bool)
    if empty_memory:
        mem_mask[:] = True
    cand_mask = torch.zeros(B, A, dtype=torch.bool)
    if all_blocked:
        cand_mask[:] = True
    x0 = torch.rand(B, L, 2, generator=g) * 0.8
    boxes = torch.cat([x0, x0 + 0.1], dim=-1)
    return SomaBatch(
        images=torch.randn(B, 3, image_size, image_size, generator=g),
        ocr_ids=torch.randint(1, vocab_size, (B, L), generator=g),
        ocr_boxes=boxes,
        ocr_padding_mask=ocr_mask,
        history=torch.randn(B, T, D, generator=g),
        history_padding_mask=torch.zeros(B, T, dtype=torch.bool),
        app_ids=torch.randint(0, n_apps, (B,), generator=g),
        memory_embeddings=torch.randn(B, K, D, generator=g),
        memory_padding_mask=mem_mask,
        candidate_embeddings=torch.randn(B, A, D, generator=g),
        candidate_types=torch.randint(1, n_action_types, (B, A), generator=g),
        candidate_mask=cand_mask,
    )
