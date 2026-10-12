"""Checkpoints as ``state_dict`` + config + version (never pickled Python objects).

Loading uses ``torch.load(..., weights_only=True)``, which only accepts tensors
and plain containers/scalars, so a tampered file cannot execute code.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from soma.config import ModelConfig
from soma.models.soma_model import SomaModel, build_model

CHECKPOINT_VERSION = "soma-model-1"


def config_from_dict(data: dict[str, Any]) -> ModelConfig:
    """Rebuild a ``ModelConfig`` from ``asdict`` output, ignoring unknown keys."""
    known = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in data.items() if k in known})


def save_checkpoint(model: SomaModel, path: str | Path, cfg: ModelConfig,
                    extra: dict | None = None) -> Path:
    """Write ``{"state_dict", "config", "version", "extra"}`` to ``path``.

    ``extra`` must contain only plain types (str/int/float/bool/None, lists,
    dicts, tensors) so it survives a ``weights_only`` load. Parent directories
    are created. Returns the path written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {"state_dict": state, "config": asdict(cfg), "version": CHECKPOINT_VERSION,
               "extra": dict(extra or {})}
    tmp = p.with_name(p.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    return p


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"
                    ) -> tuple[SomaModel, ModelConfig, dict]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns ``(model, cfg, extra)``; the model is on ``map_location`` and in
    eval mode.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {version!r}")
    cfg = config_from_dict(payload["config"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    return model, cfg, payload.get("extra") or {}


def copy_model(model: SomaModel) -> SomaModel:
    """Independent deep copy (same weights, same train/eval mode, no shared tensors)."""
    return copy.deepcopy(model)
