"""Model-level abstention: the NO_SUGGESTION logit (index 0) and blocked candidates."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from soma.models import (BLOCKED_LOGIT, NO_SUGGESTION_INDEX, ActionPolicy, build_model,
                         random_batch)


def test_no_suggestion_is_index_zero():
    assert NO_SUGGESTION_INDEX == 0


@pytest.mark.parametrize("kind", ["gru", "transformer"])
@pytest.mark.parametrize("train", [False, True])
def test_all_blocked_puts_all_mass_on_no_suggestion(small_cfg, kind, train):
    small_cfg.temporal_kind = kind
    model = build_model(small_cfg).train(train)
    out = model(random_batch(3, all_blocked=True, empty_ocr=True, empty_memory=True))
    probs = out["action_logits"].softmax(-1)
    assert torch.isfinite(out["action_logits"]).all()
    assert torch.allclose(probs[:, 0], torch.ones(3), atol=1e-6)
    assert (out["action_logits"].argmax(-1) == NO_SUGGESTION_INDEX).all()


def test_blocked_candidates_never_win(small_cfg):
    pol = ActionPolicy(small_cfg).eval()
    D = small_cfg.d_model
    mask = torch.tensor([[True, False, True, True], [False, True, True, True]])
    logits = pol(torch.randn(2, D), torch.randn(2, 4, D), torch.randint(0, 6, (2, 4)), mask)
    assert (logits[:, 1:][mask] == BLOCKED_LOGIT).all()
    assert logits.softmax(-1)[:, 1:][mask].max() < 1e-30


def test_blocked_logit_is_fp16_safe(small_cfg):
    pol = ActionPolicy(small_cfg).eval().half()
    D = small_cfg.d_model
    logits = pol(torch.randn(2, D).half(), torch.randn(2, 3, D).half(),
                 torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool))
    assert logits.dtype == torch.float16 and torch.isfinite(logits).all()
    assert torch.allclose(logits.float().softmax(-1)[:, 0], torch.ones(2))


def test_no_candidates_at_all_returns_only_abstain(small_cfg):
    pol = ActionPolicy(small_cfg)
    logits = pol(torch.randn(2, small_cfg.d_model), torch.zeros(2, 0, small_cfg.d_model))
    assert logits.shape == (2, 1) and logits.softmax(-1).eq(1).all()


def test_no_suggestion_logit_is_trainable(small_cfg):
    """Training on 'silence' targets raises P(NO_SUGGESTION)."""
    model = build_model(small_cfg).train()
    b = random_batch(4, generator=torch.Generator().manual_seed(2))
    target = torch.zeros(4, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        p0 = model.eval()(b)["action_logits"].softmax(-1)[:, 0].mean()
    model.train()
    for _ in range(20):
        opt.zero_grad()
        F.cross_entropy(model(b)["action_logits"], target).backward()
        opt.step()
    with torch.no_grad():
        p1 = model.eval()(b)["action_logits"].softmax(-1)[:, 0].mean()
    assert p1 > p0


def test_blocked_candidates_get_no_gradient(small_cfg):
    pol = ActionPolicy(small_cfg)
    D = small_cfg.d_model
    cand = torch.randn(1, 3, D, requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    logits = pol(torch.randn(1, D), cand, None, mask)
    F.cross_entropy(logits, torch.tensor([1])).backward()
    assert cand.grad[0, 1].abs().sum() == 0
    assert cand.grad[0, 0].abs().sum() > 0
