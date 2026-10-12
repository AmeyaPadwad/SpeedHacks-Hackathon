"""Stage B/D trainer: AdamW, grad clipping, autocast, early stopping, local logs.

* ``trainable`` freezes everything except the policy head (``"policy_head"``),
  the fusion layers + policy (``"fusion+policy"``) or nothing (``"all"``).
  Parameters whose name contains ``"adapter"`` / ``"lora"`` always stay
  trainable so small adapters can be tuned on top of frozen weights.
  Fully frozen top-level modules are kept in ``eval()`` mode.
* Mixed precision: CUDA uses fp16 autocast + ``GradScaler`` when
  ``cfg.use_amp``; CPU bf16 autocast is opt-in (``cpu_amp=True``) and only
  used when supported.
* Loss curves are appended as JSONL to ``<run_dir>/loss_curve.jsonl``
  (local only). The best validation state is checkpointed with
  ``save_checkpoint`` to ``<run_dir>/best.pt`` and restored at the end.
* ``max_steps`` / ``time_budget_s`` bound continual updates.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from soma.config import ModelConfig, TrainConfig
from soma.device import seed_everything, select_device
from soma.learning.losses import total_loss
from soma.models.batch import SomaBatch

TRAINABLE_CHOICES = ("policy_head", "fusion+policy", "all")
_TRAINABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "policy_head": ("action_policy.",),
    "fusion+policy": ("screen_fusion.", "memory_fusion.", "action_policy."),
}
_ALWAYS_TRAINABLE = ("adapter", "lora")


def set_trainable(model: nn.Module, trainable: str = "all") -> list[str]:
    """Freeze parameters per ``trainable``; returns the names left trainable."""
    if trainable not in TRAINABLE_CHOICES:
        raise ValueError(f"trainable must be one of {TRAINABLE_CHOICES}")
    names = []
    for name, p in model.named_parameters():
        if trainable == "all":
            on = True
        else:
            on = name.startswith(_TRAINABLE_PREFIXES[trainable]) or \
                any(tag in name.lower() for tag in _ALWAYS_TRAINABLE)
        p.requires_grad_(on)
        if on:
            names.append(name)
    if not names:
        raise ValueError(f"no parameters matched trainable={trainable!r}")
    return names


def _cpu_bf16_supported() -> bool:
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.ones(2, 2) @ torch.ones(2, 2)
        return True
    except (RuntimeError, TypeError):
        return False


@dataclass
class TrainResult:
    epochs_run: int = 0
    steps: int = 0
    best_val_loss: float = math.inf
    best_epoch: int = -1
    stopped_reason: str = "completed"     # completed | early_stopping | max_steps | time_budget
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_path: str | None = None
    first_train_loss: float = math.nan
    last_train_loss: float = math.nan
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """Train a ``SomaModel`` (or any module returning ``{"action_logits"}``) on SomaBatches."""

    def __init__(self, model: nn.Module, cfg: TrainConfig, device: torch.device | str | None = None, *,
                 trainable: str = "all", run_dir: str | Path | None = None,
                 model_cfg: ModelConfig | None = None, patience: int = 3,
                 max_steps: int | None = None, time_budget_s: float | None = None,
                 label_smoothing: float = 0.0, cpu_amp: bool = False,
                 restore_best: bool = True, log_every: int = 1) -> None:
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else select_device()
        self.model = model.to(self.device)
        self.trainable = trainable
        self.trainable_names = set_trainable(model, trainable)
        self.model_cfg = model_cfg if model_cfg is not None else getattr(model, "cfg", None)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.patience = int(patience)
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        self.label_smoothing = float(label_smoothing)
        self.restore_best = restore_best
        self.log_every = max(1, int(log_every))
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        cuda = self.device.type == "cuda"
        self.amp_dtype: torch.dtype | None = None
        if cfg.use_amp and cuda:
            self.amp_dtype = torch.float16
        elif cfg.use_amp and cpu_amp and _cpu_bf16_supported():
            self.amp_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_dtype is torch.float16))
        self.global_step = 0

    # ------------------------------------------------------------ helpers
    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def _train_mode(self) -> None:
        self.model.train()
        for child in self.model.children():
            if not any(p.requires_grad for p in child.parameters()):
                child.eval()           # frozen backbones: no dropout / BN updates

    def _log(self, record: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "loss_curve.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _forward_loss(self, batch: SomaBatch, targets: torch.Tensor, weights: torch.Tensor | None
                      ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        with self._autocast():
            logits = self.model(batch)["action_logits"]
        loss, parts = total_loss(logits.float(), targets, weights,
                                 label_smoothing=self.label_smoothing,
                                 lambda_retrieval=self.cfg.lambda_retrieval)
        return loss, parts, logits

    # ------------------------------------------------------------ steps
    def train_step(self, batch: SomaBatch, targets: torch.Tensor,
                   weights: torch.Tensor | None = None) -> dict[str, float]:
        """One optimizer step; returns loss parts (floats)."""
        self._train_mode()
        batch = batch.to(self.device)
        targets = targets.to(self.device)
        weights = weights.to(self.device) if weights is not None else None
        self.optimizer.zero_grad(set_to_none=True)
        loss, parts, _ = self._forward_loss(batch, targets, weights)
        if not torch.isfinite(loss):
            parts["skipped_nonfinite"] = 1.0
            return parts
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1
        parts["grad_norm"] = float(grad_norm)
        return parts

    @torch.no_grad()
    def validate(self, loader: Iterable) -> dict[str, float]:
        """Mean weighted policy loss and top-1 accuracy on ``loader``."""
        self.model.eval()
        tot_loss = tot_w = correct = n = 0.0
        for batch, targets, weights in loader:
            batch, targets, weights = batch.to(self.device), targets.to(self.device), weights.to(self.device)
            loss, _, logits = self._forward_loss(batch, targets, weights)
            w = float(weights.sum())
            tot_loss += float(loss) * w
            tot_w += w
            correct += float((logits.argmax(-1) == targets).sum())
            n += targets.numel()
        return {"val_loss": tot_loss / max(tot_w, 1e-8), "val_acc": correct / max(n, 1)}

    # ------------------------------------------------------------ loop
    def fit(self, train_loader: Iterable, val_loader: Iterable | None = None,
            epochs: int | None = None) -> TrainResult:
        """Epoch loop with early stopping on validation loss and best-checkpointing."""
        seed_everything(self.cfg.seed)
        epochs = self.cfg.epochs if epochs is None else epochs
        res = TrainResult()
        start = time.monotonic()
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        stop: str | None = None
        for epoch in range(epochs):
            ep_loss = ep_n = 0.0
            for batch, targets, weights in train_loader:
                parts = self.train_step(batch, targets, weights)
                if "skipped_nonfinite" not in parts:
                    ep_loss += parts["total"]
                    ep_n += 1
                    if math.isnan(res.first_train_loss):
                        res.first_train_loss = parts["total"]
                    res.last_train_loss = parts["total"]
                if self.global_step % self.log_every == 0:
                    self._log({"kind": "step", "epoch": epoch, "step": self.global_step, **parts})
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    stop = "max_steps"
                elif self.time_budget_s is not None and time.monotonic() - start >= self.time_budget_s:
                    stop = "time_budget"
                if stop:
                    break
            record: dict[str, Any] = {"kind": "epoch", "epoch": epoch, "step": self.global_step,
                                      "train_loss": ep_loss / max(ep_n, 1)}
            if val_loader is not None:
                record.update(self.validate(val_loader))
                val = record["val_loss"]
                if val < res.best_val_loss - 1e-6:
                    res.best_val_loss, res.best_epoch, bad_epochs = val, epoch, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                    res.checkpoint_path = self._save_best(epoch, val)
                else:
                    bad_epochs += 1
            res.history.append(record)
            self._log(record)
            res.epochs_run = epoch + 1
            if stop:
                break
            if val_loader is not None and bad_epochs >= self.patience:
                stop = "early_stopping"
                break
        res.stopped_reason = stop or "completed"
        if self.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
        res.steps = self.global_step
        res.elapsed_s = time.monotonic() - start
        self.model.eval()
        return res

    def _save_best(self, epoch: int, val_loss: float) -> str | None:
        if self.run_dir is None or self.model_cfg is None:
            return None
        from soma.models.checkpoint import save_checkpoint

        path = save_checkpoint(self.model, self.run_dir / "best.pt", self.model_cfg,
                               extra={"epoch": int(epoch), "val_loss": float(val_loss),
                                      "trainable": self.trainable, "seed": int(self.cfg.seed)})
        return str(path)
