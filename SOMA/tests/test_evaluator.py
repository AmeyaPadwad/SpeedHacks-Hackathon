import math

import pytest
import torch
from torch import nn

from soma.learning.datasets import make_loader
from soma.learning.evaluator import (collect_logits, compare, evaluate, evaluate_no_memory,
                                     expected_calibration_error, mask_memories, metrics_from_logits)
from soma.learning.synthetic_data import generate_synthetic


def test_metrics_on_hand_made_logits():
    # rows: correct silence, wrong speak (label silence), correct speak, missed (silent, label 2)
    logits = torch.tensor([[5.0, 0.0, 0.0],
                           [0.0, 5.0, 0.0],
                           [0.0, 0.0, 5.0],
                           [5.0, 0.0, 4.0]])
    targets = torch.tensor([0, 0, 2, 2])
    m = metrics_from_logits(logits, targets, k=2)
    assert m["n"] == 4 and m["top1"] == 0.5
    assert m["topk"] == 1.0                         # row 2: label 0 in top-2 {1, 0/2}; row 4: 2 in top-2
    assert m["abstain_precision"] == 0.5            # predicted silence twice, right once
    assert m["abstain_recall"] == 0.5               # two silence labels, one caught
    assert m["unwanted_suggestion_rate"] == 0.5     # spoke on one of two silence labels
    assert m["missed_suggestion_rate"] == 0.5
    expected_nll = -torch.log_softmax(logits, -1)[torch.arange(4), targets].mean().item()
    assert m["nll"] == pytest.approx(expected_nll)
    assert m["silence_fraction"] == 0.5


def test_ece_perfect_and_overconfident():
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, torch.tensor([0, 1])) == pytest.approx(0.0)
    assert expected_calibration_error(probs, torch.tensor([1, 0])) == pytest.approx(1.0)
    half = torch.tensor([[0.6, 0.4]] * 10)
    t = torch.tensor([0] * 6 + [1] * 4)
    assert expected_calibration_error(half, t) == pytest.approx(0.0, abs=1e-6)


def test_compare_knows_direction():
    c = compare({"top1": 0.5, "nll": 1.0, "unwanted_suggestion_rate": 0.1},
                {"top1": 0.6, "nll": 1.2, "unwanted_suggestion_rate": 0.05})
    assert c["top1"]["improved"] and c["top1"]["delta"] == pytest.approx(0.1)
    assert not c["nll"]["improved"]
    assert c["unwanted_suggestion_rate"]["improved"]


class _MemoryProbe(nn.Module):
    """Speaks (index 1) iff any memory is present; otherwise silent."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        has = (~batch.memory_padding_mask).any(-1).float()
        A = batch.candidate_embeddings.shape[1]
        logits = torch.zeros(batch.batch_size, A + 1)
        logits[:, 0] = 1 - has
        logits[:, 1] = has
        return {"action_logits": logits + self.p}


def test_no_memory_ablation_masks_all_memories(small_cfg):
    items = generate_synthetic(40, small_cfg, seed=0)
    loader = make_loader(items, 8)
    for batch, _, _ in mask_memories(loader):
        assert batch.memory_padding_mask.all() and batch.memory_embeddings.abs().sum() == 0
    model = _MemoryProbe()
    logits, _ = collect_logits(model, mask_memories(loader))
    assert (logits.argmax(-1) == 0).all()
    full = evaluate(model, loader)
    nomem = evaluate_no_memory(model, loader)
    assert nomem["unwanted_suggestion_rate"] == 0.0 and nomem["top1"] == pytest.approx(nomem["silence_fraction"])
    assert full["n"] == nomem["n"] == 40


def test_logits_hook_and_eval_mode_restored(small_cfg):
    items = generate_synthetic(16, small_cfg, seed=1)
    model = _MemoryProbe().train()
    hook = lambda b, l: l + torch.tensor([100.0] + [0.0] * (l.shape[1] - 1))
    m = evaluate(model, make_loader(items, 8), logits_hook=hook)
    assert m["abstain_recall"] == 1.0 and m["unwanted_suggestion_rate"] == 0.0
    assert model.training
    empty = metrics_from_logits(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
    assert empty["n"] == 0 and not math.isnan(empty["top1"])
