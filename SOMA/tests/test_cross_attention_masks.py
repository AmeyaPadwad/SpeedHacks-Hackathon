"""Masks actually mask: padded content never changes outputs, and empty inputs stay finite."""

from __future__ import annotations

import pytest
import torch

from soma.models import (MemoryFusion, ScreenFusion, TemporalEncoder, TextLayoutEncoder,
                         build_model, random_batch)
from soma.models.attention import (AttentionPool, causal_padding_attn_mask, masked_mean,
                                   masked_softmax, safe_padding_mask)

B = 2


def _finite(*ts: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in ts)


# -- helpers ------------------------------------------------------------------------
def test_safe_padding_mask_only_touches_empty_rows():
    m = torch.tensor([[False, True], [True, True]])
    out = safe_padding_mask(m)
    assert out.tolist() == [[False, True], [False, True]]
    assert m.tolist() == [[False, True], [True, True]]          # input not mutated


def test_masked_softmax_and_mean_all_padded():
    logits = torch.randn(2, 3)
    mask = torch.tensor([[False, True, False], [True, True, True]])
    p = masked_softmax(logits, mask)
    assert _finite(p) and torch.allclose(p[0].sum(), torch.tensor(1.0)) and p[1].abs().sum() == 0
    mean = masked_mean(torch.randn(2, 3, 4), mask)
    assert _finite(mean) and mean[1].abs().sum() == 0


def test_causal_padding_mask_never_blocks_a_full_row():
    pad = torch.tensor([[True, True, False], [False, True, False]])
    m = causal_padding_attn_mask(pad, n_heads=2)
    assert m.shape == (4, 3, 3)
    assert not m.all(dim=-1).any()
    assert m[0].tolist() == [[False, True, True], [True, False, True], [True, True, False]]


def test_attention_pool_ignores_padding_and_handles_empty():
    pool = AttentionPool(16, 4).eval()
    x = torch.randn(B, 5, 16)
    mask = torch.tensor([[False, False, True, True, True]] * B)
    y = x.clone()
    y[:, 2:] = 99.0
    assert torch.allclose(pool(x, mask), pool(y, mask), atol=1e-6)
    assert _finite(pool(x, torch.ones(B, 5, dtype=torch.bool)))


# -- modules ------------------------------------------------------------------------
def test_text_encoder_padding_invariance(small_cfg):
    enc = TextLayoutEncoder(small_cfg).eval()
    b = random_batch(B, L=8)
    ids2, boxes2 = b.ocr_ids.clone(), b.ocr_boxes.clone()
    ids2[b.ocr_padding_mask] = 7
    boxes2[b.ocr_padding_mask] = 0.5
    a = enc(b.ocr_ids, b.ocr_boxes, b.ocr_padding_mask)
    c = enc(ids2, boxes2, b.ocr_padding_mask)
    assert torch.allclose(a, c, atol=1e-6)
    assert a[b.ocr_padding_mask].abs().sum() == 0


@pytest.mark.parametrize("train", [False, True])
def test_text_encoder_empty_ocr_is_finite(small_cfg, train):
    enc = TextLayoutEncoder(small_cfg).train(train)
    b = random_batch(B, L=8, empty_ocr=True)
    out = enc(b.ocr_ids, b.ocr_boxes, b.ocr_padding_mask)
    assert _finite(out) and out.abs().sum() == 0


def test_screen_fusion_padding_invariance_and_empty_ocr(small_cfg):
    D = small_cfg.d_model
    fusion = ScreenFusion(small_cfg).eval()
    vis, txt = torch.randn(B, 16, D), torch.randn(B, 8, D)
    mask = torch.zeros(B, 8, dtype=torch.bool)
    mask[:, 3:] = True
    txt2 = txt.clone()
    txt2[:, 3:] = 42.0
    assert torch.allclose(fusion(vis, txt, mask), fusion(vis, txt2, mask), atol=1e-5)
    assert _finite(fusion(vis, txt, torch.ones(B, 8, dtype=torch.bool)))


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_temporal_padding_invariance(small_cfg, kind):
    small_cfg.temporal_kind = kind
    D, T = small_cfg.d_model, small_cfg.history_len
    enc = TemporalEncoder(small_cfg).eval()
    state, hist = torch.randn(B, D), torch.randn(B, T, D)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[0, :2] = True            # left padding
    pad[1, 1] = True             # a gap in the middle
    hist2 = hist.clone()
    hist2[pad] = -7.0
    assert torch.allclose(enc(state, hist, pad), enc(state, hist2, pad), atol=1e-5)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_temporal_all_padded_equals_no_history(small_cfg, kind):
    small_cfg.temporal_kind = kind
    D, T = small_cfg.d_model, small_cfg.history_len
    enc = TemporalEncoder(small_cfg).eval()
    state, app = torch.randn(B, D), torch.tensor([1, 2])
    full_pad = enc(state, torch.randn(B, T, D), torch.ones(B, T, dtype=torch.bool), app)
    none = enc(state, None, None, app)
    assert _finite(full_pad)
    assert torch.allclose(full_pad, none, atol=1e-5)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_temporal_is_causal(small_cfg, kind):
    """Changing step t never changes per-step outputs before t."""
    small_cfg.temporal_kind = kind
    D, T = small_cfg.d_model, small_cfg.history_len
    enc = TemporalEncoder(small_cfg).eval()
    state, hist = torch.randn(B, D), torch.randn(B, T, D)
    _, steps = enc(state, hist, return_steps=True)
    hist2 = hist.clone()
    hist2[:, 2:] = torch.randn(B, T - 2, D)
    _, steps2 = enc(state, hist2, return_steps=True)
    assert torch.allclose(steps[:, :2], steps2[:, :2], atol=1e-6)
    assert not torch.allclose(steps[:, 2:], steps2[:, 2:])


def test_memory_padding_invariance_and_null_key(small_cfg):
    D = small_cfg.d_model
    fusion = MemoryFusion(small_cfg).eval()
    state, mem = torch.randn(B, D), torch.randn(B, 4, D)
    mask = torch.tensor([[False, False, True, True]] * B)
    mem2 = mem.clone()
    mem2[mask] = 13.0
    a, c = fusion(state, mem, mask), fusion(state, mem2, mask)
    assert torch.allclose(a.context, c.context, atol=1e-5)
    padded_slots = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask], dim=1)
    assert a.attention[padded_slots].abs().sum() == 0


def test_memory_all_masked_attends_to_no_memory(small_cfg):
    D = small_cfg.d_model
    for train in (False, True):
        out = MemoryFusion(small_cfg).train(train)(
            torch.randn(B, D), torch.randn(B, 3, D), torch.ones(B, 3, dtype=torch.bool))
        assert _finite(out.context, out.gate, out.attention)
        assert torch.allclose(out.attention[:, 0], torch.ones(B), atol=1e-5)


def test_full_model_ocr_padding_invariance(small_cfg):
    model = build_model(small_cfg).eval()
    b = random_batch(B)
    b.memory_padding_mask[:, -1] = True
    b.candidate_mask[:, -1] = True
    with torch.no_grad():
        a = model(b)["action_logits"]
        b.ocr_ids[b.ocr_padding_mask] = 5
        b.ocr_boxes[b.ocr_padding_mask] = 0.3
        b.memory_embeddings[:, -1] = 100.0
        b.candidate_embeddings[:, -1] = -50.0
        c = model(b)["action_logits"]
    assert torch.allclose(a, c, atol=1e-5)
