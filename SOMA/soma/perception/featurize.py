"""Turn a :class:`ScreenObservation` into OCR tensors for the text/layout encoder.

One token entry per word (hashed with :func:`soma.models.tokenizer.encode`);
each token carries the normalized ``xyxy`` box of the element it came from.
Sequences are truncated / padded to ``cfg.max_ocr_tokens`` (``L``). The
padding mask follows PyTorch's ``key_padding_mask`` convention: ``True = PAD``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from soma.config import ModelConfig
from soma.models.tokenizer import PAD_ID, encode, hash_bucket
from soma.types import OCRElement, ScreenObservation, WindowInfo

#: Roles moved to the front before truncation so they survive long screens.
PRIORITY_ROLES: tuple[str, ...] = ("error", "file_ref", "button")


def app_bucket(window: WindowInfo | None, n_apps: int) -> int:
    """Stable hashed bucket of the foreground app name (case-insensitive)."""
    name = (window.app_name if window is not None else "") or ""
    return hash_bucket(name.lower(), n_apps)


def _ordered(elements: Sequence[OCRElement], prioritize: bool) -> list[OCRElement]:
    if not prioritize:
        return list(elements)
    rank = {r: i for i, r in enumerate(PRIORITY_ROLES)}
    # stable sort keeps reading order within each group
    return sorted(elements, key=lambda e: rank.get(e.ui_role or "", len(rank)))


def observation_to_tensors(obs: ScreenObservation | None, cfg: ModelConfig,
                           prioritize: bool = True,
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(ocr_ids [L] long, ocr_boxes [L,4] float, ocr_padding_mask [L] bool)``.

    An empty or ``None`` observation yields all-PAD tensors (mask all ``True``).
    """
    L = int(cfg.max_ocr_tokens)
    ids = torch.full((L,), PAD_ID, dtype=torch.long)
    boxes = torch.zeros((L, 4), dtype=torch.float32)
    mask = torch.ones((L,), dtype=torch.bool)
    if obs is None:
        return ids, boxes, mask
    n = 0
    for e in _ordered(obs.elements, prioritize):
        if n >= L:
            break
        toks = encode(e.text, cfg.vocab_size, max_len=L - n)
        if not toks:
            continue
        box = torch.tensor([min(1.0, max(0.0, float(v))) for v in e.bbox_xyxy_normalized],
                           dtype=torch.float32)
        k = len(toks)
        ids[n:n + k] = torch.tensor(toks, dtype=torch.long)
        boxes[n:n + k] = box
        mask[n:n + k] = False
        n += k
    return ids, boxes, mask


def batch_observations(observations: Sequence[ScreenObservation | None], cfg: ModelConfig,
                       prioritize: bool = True) -> dict[str, torch.Tensor]:
    """Stack several observations: ``ocr_ids [B,L]``, ``ocr_boxes [B,L,4]``,
    ``ocr_padding_mask [B,L]`` and ``app_ids [B]``."""
    rows = [observation_to_tensors(o, cfg, prioritize) for o in observations]
    if not rows:
        L = int(cfg.max_ocr_tokens)
        return {"ocr_ids": torch.zeros(0, L, dtype=torch.long),
                "ocr_boxes": torch.zeros(0, L, 4), "ocr_padding_mask": torch.zeros(0, L, dtype=torch.bool),
                "app_ids": torch.zeros(0, dtype=torch.long)}
    return {
        "ocr_ids": torch.stack([r[0] for r in rows]),
        "ocr_boxes": torch.stack([r[1] for r in rows]),
        "ocr_padding_mask": torch.stack([r[2] for r in rows]),
        "app_ids": torch.tensor([app_bucket(o.window if o else None, cfg.n_apps) for o in observations],
                                dtype=torch.long),
    }
