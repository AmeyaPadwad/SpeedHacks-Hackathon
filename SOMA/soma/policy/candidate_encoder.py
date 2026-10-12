"""Encode templated candidates into fixed-size ``[A, D]`` tensors for the scorer.

Each candidate embedding is the LayerNorm'd sum of:

* an action-type ``nn.Embedding`` (index into ``ACTION_TYPES``),
* hashed text tokens (``soma.models.tokenizer``) pooled by ``nn.EmbeddingBag``,
* a projection of the target box ``[x1, y1, x2, y2, has_target]``,
* an optional projection of the referenced memory embedding.

Padding slots and blocked candidates are marked ``True`` in the mask so the
policy model can exclude them. The NO_SUGGESTION logit is the model's job; it
is never a candidate here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from soma.config import ModelConfig
from soma.models.tokenizer import PAD_ID, UNK_ID, encode as encode_text
from soma.types import ACTION_TYPES, Candidate, RetrievedMemory

MemoryLike = RetrievedMemory | np.ndarray | torch.Tensor

_TYPE_INDEX = {a: i for i, a in enumerate(ACTION_TYPES)}


def action_index(candidate: Candidate) -> int:
    """Index of ``candidate.action`` in ``ACTION_TYPES``."""
    return _TYPE_INDEX[candidate.action]


class CandidateEncoder(nn.Module):
    """Candidates -> ``(embeddings [A,D], types [A] long, mask [A] bool)``."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_action_types: int = len(ACTION_TYPES),
        memory_dim: int | None = None,
        max_text_tokens: int = 48,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.memory_dim = memory_dim or d_model
        self.max_text_tokens = max_text_tokens
        self.type_embedding = nn.Embedding(n_action_types, d_model)
        self.text_bag = nn.EmbeddingBag(vocab_size, d_model, mode="mean", padding_idx=PAD_ID)
        self.box_proj = nn.Linear(5, d_model)
        self.memory_proj = nn.Linear(self.memory_dim + 1, d_model)
        self.norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg: ModelConfig, memory_dim: int | None = None) -> "CandidateEncoder":
        return cls(d_model=cfg.d_model, vocab_size=cfg.vocab_size,
                   n_action_types=cfg.n_action_types, memory_dim=memory_dim)

    @property
    def device(self) -> torch.device:
        return self.type_embedding.weight.device

    # ------------------------------------------------------------ features

    def _text_of(self, c: Candidate) -> str:
        parts = [c.text]
        if c.target is not None:
            parts.append(c.target.text)
        return " ".join(parts)

    def _memory_vector(self, mem: MemoryLike | None) -> np.ndarray:
        vec = np.zeros(self.memory_dim + 1, dtype=np.float32)
        if mem is None:
            return vec
        raw = mem.embedding if isinstance(mem, RetrievedMemory) else mem
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()
        arr = np.nan_to_num(np.asarray(raw, dtype=np.float32).reshape(-1))
        n = min(arr.shape[0], self.memory_dim)
        vec[:n] = arr[:n]
        vec[-1] = 1.0
        return vec

    def forward(
        self,
        types: torch.Tensor,          # [N] long
        token_ids: torch.Tensor,      # [T] long (flattened bags)
        offsets: torch.Tensor,        # [N] long
        boxes: torch.Tensor,          # [N, 5]
        mem_vecs: torch.Tensor,       # [N, memory_dim + 1]
    ) -> torch.Tensor:
        """Embed N real candidates -> ``[N, D]``."""
        h = (self.type_embedding(types)
             + self.text_bag(token_ids, offsets)
             + self.box_proj(boxes)
             + self.memory_proj(mem_vecs))
        return self.norm(h)

    # ------------------------------------------------------------ public API

    def encode(
        self,
        candidates: Sequence[Candidate],
        memories_by_id: Mapping[str, MemoryLike] | None = None,
        A: int | None = None,
        D: int | None = None,
        blocked: Sequence[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode up to ``A`` candidates.

        Returns ``embeddings [A, D]`` (zeros in pad slots), ``types [A]`` long
        (pads use index 0) and ``mask [A]`` bool where True means pad/blocked.
        Candidates beyond ``A`` are dropped.
        """
        A = len(candidates) if A is None else A
        D = self.d_model if D is None else D
        if D != self.d_model:
            raise ValueError(f"D={D} does not match encoder d_model={self.d_model}")
        memories_by_id = memories_by_id or {}
        dev = self.device
        emb = torch.zeros(A, D, device=dev)
        types = torch.zeros(A, dtype=torch.long, device=dev)
        mask = torch.ones(A, dtype=torch.bool, device=dev)
        real = list(candidates)[:A]
        if not real:
            return emb, types, mask

        type_ids, tokens, offsets, boxes, mems = [], [], [], [], []
        for c in real:
            type_ids.append(action_index(c))
            ids = encode_text(self._text_of(c), self.vocab_size, self.max_text_tokens) or [UNK_ID]
            offsets.append(len(tokens))
            tokens.extend(ids)
            if c.target is not None:
                x1, y1, x2, y2 = (float(min(max(v, 0.0), 1.0)) for v in c.target.bbox_xyxy_normalized)
                boxes.append([x1, y1, x2, y2, 1.0])
            else:
                boxes.append([0.0, 0.0, 0.0, 0.0, 0.0])
            mems.append(self._memory_vector(memories_by_id.get(c.memory_ref) if c.memory_ref else None))

        n = len(real)
        out = self(
            torch.tensor(type_ids, dtype=torch.long, device=dev),
            torch.tensor(tokens, dtype=torch.long, device=dev),
            torch.tensor(offsets, dtype=torch.long, device=dev),
            torch.tensor(boxes, dtype=torch.float32, device=dev),
            torch.from_numpy(np.stack(mems)).to(dev),
        )
        emb[:n] = out
        types[:n] = torch.tensor(type_ids, dtype=torch.long, device=dev)
        mask[:n] = False
        if blocked is not None:
            for i, b in enumerate(list(blocked)[:n]):
                if b:
                    mask[i] = True
        return emb, types, mask

    def encode_batch(
        self,
        batch: Sequence[Sequence[Candidate]],
        memories_by_id: Sequence[Mapping[str, MemoryLike] | None] | Mapping[str, MemoryLike] | None = None,
        A: int | None = None,
        blocked: Sequence[Sequence[bool] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack per-sample encodings into ``[B, A, D]``, ``[B, A]``, ``[B, A]``."""
        A = max((len(c) for c in batch), default=0) if A is None else A
        A = max(A, 1)
        embs, types, masks = [], [], []
        for i, cands in enumerate(batch):
            if isinstance(memories_by_id, Mapping) or memories_by_id is None:
                mems = memories_by_id
            else:
                mems = memories_by_id[i]
            blk = blocked[i] if blocked is not None else None
            e, t, m = self.encode(cands, mems, A=A, blocked=blk)
            embs.append(e)
            types.append(t)
            masks.append(m)
        return torch.stack(embs), torch.stack(types), torch.stack(masks)
