"""Per-example tensor items, padding collate and seeded splits/loaders (Stage B).

A :class:`SomaItem` is one un-batched ``SomaBatch`` row plus its label. Labels
index the policy logits ``[A + 1]``: ``0`` is ``NO_SUGGESTION`` and ``1 + a``
is candidate slot ``a``. ``collate_soma`` pads variable ``L``/``T``/``K``/``A``
to the batch maximum; padded positions are always marked ``True`` (= PAD) in
their masks, so padding candidates never change label indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from soma.models.batch import SomaBatch

NO_SUGGESTION_INDEX = 0

# Tensor fields of a SomaItem, in SomaBatch order (without the batch dimension).
TENSOR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(SomaBatch))


@dataclass
class SomaItem:
    """One training/eval example: ``SomaBatch`` tensors without the batch dim.

    ``target`` is the logits index of the best action (``0`` = stay silent).
    ``app`` / ``task`` are coarse, non-content labels used for stratification
    and per-app evaluation (e.g. ``"code.exe"``, ``"error"``).
    """

    images: torch.Tensor                 # [3, H, W]
    ocr_ids: torch.Tensor                # [L] long
    ocr_boxes: torch.Tensor              # [L, 4]
    ocr_padding_mask: torch.Tensor       # [L] bool
    history: torch.Tensor                # [T, D]
    history_padding_mask: torch.Tensor   # [T] bool
    app_ids: torch.Tensor                # [] long
    memory_embeddings: torch.Tensor      # [K, D]
    memory_padding_mask: torch.Tensor    # [K] bool
    candidate_embeddings: torch.Tensor   # [A, D]
    candidate_types: torch.Tensor        # [A] long
    candidate_mask: torch.Tensor         # [A] bool
    target: int = NO_SUGGESTION_INDEX
    weight: float = 1.0
    app: str = ""
    task: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return int(self.candidate_embeddings.shape[0])

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in TENSOR_FIELDS}

    def clone(self, **overrides: Any) -> "SomaItem":
        """Detached CPU deep copy (tensor snapshot), optionally overriding fields."""
        data: dict[str, Any] = {n: t.detach().to("cpu").clone() for n, t in self.tensors().items()}
        data.update(target=self.target, weight=self.weight, app=self.app, task=self.task,
                    meta=dict(self.meta))
        data.update(overrides)
        return SomaItem(**data)

    def to_state(self) -> dict[str, Any]:
        """Plain dict for ``state_dict`` serialization (tensors + scalars only)."""
        state: dict[str, Any] = {n: t.detach().cpu().clone() for n, t in self.tensors().items()}
        state.update(target=int(self.target), weight=float(self.weight), app=self.app,
                     task=self.task, meta=dict(self.meta))
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SomaItem":
        return cls(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in state.items()})


def item_from_batch(batch: SomaBatch, index: int, *, target: int = NO_SUGGESTION_INDEX,
                    weight: float = 1.0, app: str = "", task: str = "") -> SomaItem:
    """Snapshot row ``index`` of ``batch`` as a detached CPU :class:`SomaItem`."""
    data = {n: getattr(batch, n)[index].detach().to("cpu").clone() for n in TENSOR_FIELDS}
    return SomaItem(**data, target=int(target), weight=float(weight), app=app, task=task)


def _pad_stack(tensors: Sequence[torch.Tensor], length: int, value: Any) -> torch.Tensor:
    """Stack tensors padded along dim 0 to ``length`` with ``value``."""
    out = []
    for t in tensors:
        n = t.shape[0]
        if n < length:
            pad = torch.full((length - n, *t.shape[1:]), value, dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        out.append(t)
    return torch.stack(out, dim=0)


def collate_soma(items: Sequence[SomaItem]) -> tuple[SomaBatch, torch.Tensor, torch.Tensor]:
    """Batch items into ``(SomaBatch, targets [B] long, weights [B] float)``.

    Pads OCR tokens (``L``), history (``T``), memories (``K``) and candidates
    (``A``) to the per-batch maximum; padded entries are masked ``True``.
    Images must share one size.
    """
    if not items:
        raise ValueError("collate_soma needs at least one item")
    L = max(int(i.ocr_ids.shape[0]) for i in items)
    T = max(int(i.history.shape[0]) for i in items)
    K = max(int(i.memory_embeddings.shape[0]) for i in items)
    A = max(int(i.candidate_embeddings.shape[0]) for i in items)
    batch = SomaBatch(
        images=torch.stack([i.images for i in items]),
        ocr_ids=_pad_stack([i.ocr_ids.long() for i in items], L, 0),
        ocr_boxes=_pad_stack([i.ocr_boxes.float() for i in items], L, 0.0),
        ocr_padding_mask=_pad_stack([i.ocr_padding_mask.bool() for i in items], L, True),
        history=_pad_stack([i.history.float() for i in items], T, 0.0),
        history_padding_mask=_pad_stack([i.history_padding_mask.bool() for i in items], T, True),
        app_ids=torch.stack([i.app_ids.long().reshape(()) for i in items]),
        memory_embeddings=_pad_stack([i.memory_embeddings.float() for i in items], K, 0.0),
        memory_padding_mask=_pad_stack([i.memory_padding_mask.bool() for i in items], K, True),
        candidate_embeddings=_pad_stack([i.candidate_embeddings.float() for i in items], A, 0.0),
        candidate_types=_pad_stack([i.candidate_types.long() for i in items], A, 0),
        candidate_mask=_pad_stack([i.candidate_mask.bool() for i in items], A, True),
    )
    targets = torch.tensor([int(i.target) for i in items], dtype=torch.long)
    weights = torch.tensor([float(i.weight) for i in items], dtype=torch.float32)
    if bool((targets > A).any()) or bool((targets < 0).any()):
        raise ValueError("target index outside [0, A]")
    return batch, targets, weights


class SomaDataset(Dataset):
    """Map-style dataset over in-memory :class:`SomaItem` objects."""

    def __init__(self, items: Iterable[SomaItem]) -> None:
        self.items: list[SomaItem] = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> SomaItem:
        return self.items[index]


def seeded_split(dataset: Dataset, val_fraction: float, seed: int = 0) -> tuple[Subset, Subset]:
    """Deterministic train/val split. Same seed -> same partition."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    n = len(dataset)  # type: ignore[arg-type]
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(round(n * val_fraction))
    if val_fraction > 0 and n > 1:
        n_val = min(max(n_val, 1), n - 1)
    return Subset(dataset, perm[n_val:]), Subset(dataset, perm[:n_val])


def make_loader(dataset: Dataset | Sequence[SomaItem], batch_size: int = 16, *,
                shuffle: bool = False, seed: int = 0, drop_last: bool = False) -> DataLoader:
    """``DataLoader`` with ``collate_soma`` and a seeded shuffling generator."""
    if not isinstance(dataset, Dataset):
        dataset = SomaDataset(dataset)
    g = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
                      drop_last=drop_last, collate_fn=collate_soma, num_workers=0)
