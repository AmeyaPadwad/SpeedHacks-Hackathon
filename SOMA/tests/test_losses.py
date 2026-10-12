import math

import pytest
import torch

from soma.learning import metrics as M
from soma.learning.losses import (pairwise_ranking_loss, policy_loss, retrieval_contrastive_loss,
                                  total_loss)


def test_policy_loss_matches_manual_cross_entropy():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([0, 1])
    expected = -(torch.log_softmax(logits, -1)[0, 0] + torch.log_softmax(logits, -1)[1, 1]) / 2
    assert policy_loss(logits, target).item() == pytest.approx(expected.item(), abs=1e-6)


def test_policy_loss_weights_and_blocked_candidates():
    logits = torch.tensor([[0.0, 5.0, float("-inf")], [0.0, 0.0, 0.0]])
    target = torch.tensor([1, 0])
    w = torch.tensor([0.0, 1.0])
    assert policy_loss(logits, target, w).item() == pytest.approx(math.log(3), abs=1e-5)
    assert torch.isfinite(policy_loss(logits, target, label_smoothing=0.1))


def test_abstain_logit_receives_gradient():
    logits = torch.zeros(2, 4, requires_grad=True)
    policy_loss(logits, torch.tensor([0, 0])).backward()
    assert (logits.grad[:, 0] < 0).all()          # pushes NO_SUGGESTION up
    assert (logits.grad[:, 1:] > 0).all()


def test_retrieval_contrastive_loss_prefers_aligned_positive():
    q = torch.tensor([[1.0, 0.0]])
    good = retrieval_contrastive_loss(q, torch.tensor([[1.0, 0.0]]), torch.tensor([[[0.0, 1.0]]]), 0.1)
    bad = retrieval_contrastive_loss(q, torch.tensor([[0.0, 1.0]]), torch.tensor([[[1.0, 0.0]]]), 0.1)
    assert good.item() < 1e-3 and bad.item() > 5
    # Equal sims with N=1 -> log(2)
    eq = retrieval_contrastive_loss(q, torch.tensor([[0.0, 1.0]]), torch.tensor([[[0.0, 1.0]]]), 1.0)
    assert eq.item() == pytest.approx(math.log(2), abs=1e-6)


def test_pairwise_ranking_loss_hinge():
    scores = torch.tensor([[0.0, 3.0, 1.0], [0.0, 0.0, 0.5]])
    loss = pairwise_ranking_loss(scores, torch.tensor([1, 1]), torch.tensor([2, 2]), margin=1.0)
    assert loss.item() == pytest.approx((0.0 + 1.5) / 2)


def test_total_loss_skips_retrieval_when_lambda_zero():
    logits, target = torch.randn(3, 4), torch.tensor([0, 1, 2])
    retr = (torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 2, 8))
    base, parts = total_loss(logits, target, retrieval=retr, lambda_retrieval=0.0)
    assert "retrieval" not in parts and base.item() == pytest.approx(policy_loss(logits, target).item())
    _, parts = total_loss(logits, target, lambda_retrieval=0.5)        # no supervision
    assert "retrieval" not in parts
    combined, parts = total_loss(logits, target, retrieval=retr, lambda_retrieval=0.5)
    assert combined.item() == pytest.approx(parts["policy"] + 0.5 * parts["retrieval"], abs=1e-5)


# ---------------------------------------------------------------- metrics

def test_retrieval_metrics():
    assert M.recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert M.recall_at_k(["a"], set(), 1) == 1.0
    assert M.ndcg_at_k(["a", "b"], {"a"}, 2) == pytest.approx(1.0)
    assert M.ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))
    assert M.ndcg_at_k(["x"], {"a": 2.0}, 1) == 0.0


def test_ocr_metrics():
    assert M.edit_distance("kitten", "sitting") == 3
    assert M.cer("abcd", "abcd") == 0.0 and M.cer("abcd", "abed") == 0.25
    assert M.wer("the cat sat", "the cat sat down") == pytest.approx(1 / 3)
    assert M.cer("", "") == 0.0 and M.wer("", "x") == 1.0


def test_grounding_and_feedback_metrics():
    acc = M.point_in_target_accuracy([(0.5, 0.5), (0.9, 0.9)], [(0, 0, 1, 1), (0, 0, 0.5, 0.5)])
    assert acc == 0.5
    r = M.acceptance_edit_rates(["accepted", "edited", "rejected", "ignored"])
    assert r["acceptance_rate"] == 0.25 and r["edit_rate"] == 0.25 and r["useful_rate"] == 0.5
    assert M.acceptance_edit_rates([])["n"] == 0
    assert M.pairwise_win_rate(3, 1, 2) == pytest.approx(4 / 6)
    assert M.unwanted_suggestions_per_active_hour(3, 1800) == 6.0
    assert M.unwanted_suggestions_per_active_hour(3, 0) == 0.0
