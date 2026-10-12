"""Offline evaluation of the action policy (build plan section 12).

Metrics over ``[B, A+1]`` logits (index 0 = ``NO_SUGGESTION``):

* ``top1`` / ``topk``: recommendation accuracy, silence counted as a class.
* ``abstain_precision`` / ``abstain_recall``: quality of choosing silence.
* ``unwanted_suggestion_rate``: spoke although the label was silence
  (lower is better; the silence-policy metric).
* ``missed_suggestion_rate``: stayed silent although an action was labeled.
* ``ece``: expected calibration error of the top-1 confidence.
* ``nll``: mean negative log-likelihood of the label.

``evaluate_no_memory`` masks every memory slot: the plan's "no-memory,
screen-only" baseline for ablations.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from soma.models.batch import SomaBatch

LOWER_IS_BETTER = frozenset({"unwanted_suggestion_rate", "missed_suggestion_rate", "ece", "nll"})
METRIC_KEYS = ("top1", "topk", "abstain_precision", "abstain_recall", "unwanted_suggestion_rate",
               "missed_suggestion_rate", "ece", "nll")

LogitsHook = Callable[[SomaBatch, torch.Tensor], torch.Tensor]


@torch.no_grad()
def collect_logits(model: nn.Module, loader: Iterable, device: torch.device | str | None = None,
                   logits_hook: LogitsHook | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` in eval/inference mode; returns ``(logits [N, A+1], targets [N])``.

    Batches can have different ``A``; logits are right-padded with ``-inf``.
    ``logits_hook(batch, logits)`` can add e.g. online-ranker biases.
    """
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for batch, targets, _weights in loader:
                batch = batch.to(device)
                logits = model(batch)["action_logits"].float()
                if logits_hook is not None:
                    logits = logits_hook(batch, logits)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
    finally:
        model.train(was_training)
    if not all_logits:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    width = max(t.shape[1] for t in all_logits)
    padded = [F.pad(t, (0, width - t.shape[1]), value=float("-inf")) for t in all_logits]
    return torch.cat(padded), torch.cat(all_targets)


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """ECE of the argmax confidence with ``n_bins`` equal-width bins."""
    if probs.numel() == 0:
        return 0.0
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += float(in_bin.float().mean()) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))
    return ece


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, k: int = 2,
                        n_bins: int = 10) -> dict[str, float]:
    """All policy metrics from raw logits and labels."""
    n = int(targets.numel())
    if n == 0:
        return {key: 0.0 for key in METRIC_KEYS} | {"n": 0}
    logits = torch.nan_to_num(logits.float(), neginf=-1e9)
    targets = targets.long()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pred = logits.argmax(-1)
    k = max(1, min(k, logits.shape[1]))
    topk = logits.topk(k, dim=-1).indices
    pred0, t0 = pred == 0, targets == 0

    def ratio(num: torch.Tensor, den: torch.Tensor) -> float:
        d = int(den.sum())
        return float(num.sum()) / d if d else 0.0

    return {
        "n": n,
        "top1": float((pred == targets).float().mean()),
        "topk": float((topk == targets[:, None]).any(-1).float().mean()),
        "abstain_precision": ratio(pred0 & t0, pred0),
        "abstain_recall": ratio(pred0 & t0, t0),
        "unwanted_suggestion_rate": ratio(~pred0 & t0, t0),
        "missed_suggestion_rate": ratio(pred0 & ~t0, ~t0),
        "ece": expected_calibration_error(probs, targets, n_bins),
        "nll": float(-log_probs.gather(1, targets[:, None]).mean()),
        "silence_fraction": float(t0.float().mean()),
    }


def evaluate(model: nn.Module, loader: Iterable, *, device: torch.device | str | None = None,
             k: int = 2, n_bins: int = 10, logits_hook: LogitsHook | None = None) -> dict[str, float]:
    """Evaluate ``model`` on ``loader`` of ``(SomaBatch, targets, weights)``."""
    logits, targets = collect_logits(model, loader, device, logits_hook)
    return metrics_from_logits(logits, targets, k=k, n_bins=n_bins)


class _NoMemoryLoader:
    """Wrap a loader so every batch has all memories masked out (and zeroed)."""

    def __init__(self, loader: Iterable) -> None:
        self.loader = loader

    def __iter__(self) -> Iterator[tuple[SomaBatch, torch.Tensor, torch.Tensor]]:
        for batch, targets, weights in self.loader:
            fields = dict(batch.__dict__)
            fields["memory_embeddings"] = torch.zeros_like(batch.memory_embeddings)
            fields["memory_padding_mask"] = torch.ones_like(batch.memory_padding_mask, dtype=torch.bool)
            yield SomaBatch(**fields), targets, weights

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


def mask_memories(loader: Iterable) -> Iterable:
    """Loader view with every memory slot masked (no-memory ablation)."""
    return _NoMemoryLoader(loader)


def evaluate_no_memory(model: nn.Module, loader: Iterable, **kwargs: Any) -> dict[str, float]:
    """No-memory, screen-only baseline: :func:`evaluate` with all memories masked."""
    return evaluate(model, mask_memories(loader), **kwargs)


def compare(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Per-metric ``{baseline, candidate, delta, improved}``; knows lower-is-better metrics."""
    out: dict[str, dict[str, Any]] = {}
    for key in METRIC_KEYS:
        if key not in baseline or key not in candidate:
            continue
        b, c = float(baseline[key]), float(candidate[key])
        delta = c - b
        improved = delta < 0 if key in LOWER_IS_BETTER else delta > 0
        out[key] = {"baseline": b, "candidate": c, "delta": delta, "improved": bool(improved)}
    return out
