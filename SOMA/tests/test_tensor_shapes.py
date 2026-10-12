"""Shape contract for every model module (build plan sections 3 and 5)."""

from __future__ import annotations

import pytest
import torch

from soma.models import (ActionPolicy, MemoryFusion, ScreenFusion, TemporalEncoder,
                         TextLayoutEncoder, VisualEncoder, build_model, random_batch)

B = 2


def _batch(cfg, **kw):
    return random_batch(B, image_size=cfg.image_size, L=8, T=cfg.history_len, K=3, A=4,
                        D=cfg.d_model, vocab_size=cfg.vocab_size, n_apps=cfg.n_apps,
                        n_action_types=cfg.n_action_types, **kw)


def test_visual_encoder_shape(small_cfg):
    enc = VisualEncoder(small_cfg)
    out = enc(torch.randn(B, 3, small_cfg.image_size, small_cfg.image_size))
    n = (small_cfg.image_size // small_cfg.patch_size) ** 2
    assert out.shape == (B, n, small_cfg.d_model) == (B, enc.num_tokens, small_cfg.d_model)


def test_visual_encoder_smaller_input_and_rejects_larger(small_cfg):
    enc = VisualEncoder(small_cfg)
    p = small_cfg.patch_size
    assert enc(torch.randn(1, 3, 2 * p, 3 * p)).shape == (1, 6, small_cfg.d_model)
    with pytest.raises(ValueError):
        enc(torch.randn(1, 3, small_cfg.image_size + p, small_cfg.image_size))


def test_text_layout_encoder_shape(small_cfg):
    b = _batch(small_cfg)
    out = TextLayoutEncoder(small_cfg)(b.ocr_ids, b.ocr_boxes, b.ocr_padding_mask)
    assert out.shape == (B, 8, small_cfg.d_model)


def test_text_layout_encoder_zero_length(small_cfg):
    enc = TextLayoutEncoder(small_cfg)
    out = enc(torch.zeros(B, 0, dtype=torch.long), torch.zeros(B, 0, 4),
              torch.zeros(B, 0, dtype=torch.bool))
    assert out.shape == (B, 0, small_cfg.d_model)


def test_screen_fusion_shapes(small_cfg):
    D = small_cfg.d_model
    vis, txt = torch.randn(B, 16, D), torch.randn(B, 8, D)
    mask = torch.zeros(B, 8, dtype=torch.bool)
    mask[:, 5:] = True
    fusion = ScreenFusion(small_cfg)
    assert fusion(vis, txt, mask).shape == (B, D)
    state, tokens, fmask = fusion(vis, txt, mask, return_tokens=True)
    assert state.shape == (B, D) and tokens.shape == (B, 24, D) and fmask.shape == (B, 24)
    assert not fmask[:, :16].any() and fmask[:, 21:].all()


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_temporal_encoder_shapes(small_cfg, kind):
    small_cfg.temporal_kind = kind
    D, T = small_cfg.d_model, small_cfg.history_len
    enc = TemporalEncoder(small_cfg)
    state = torch.randn(B, D)
    out, steps = enc(state, torch.randn(B, T, D), torch.zeros(B, T, dtype=torch.bool),
                     torch.tensor([0, 3]), return_steps=True)
    assert out.shape == (B, D) and steps.shape == (B, T + 1, D)
    assert enc(state).shape == (B, D)                            # no history at all


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_temporal_encoder_truncates_long_history(small_cfg, kind):
    small_cfg.temporal_kind = kind
    D, T = small_cfg.d_model, small_cfg.history_len
    enc = TemporalEncoder(small_cfg)
    out, steps = enc(torch.randn(B, D), torch.randn(B, T + 5, D), return_steps=True)
    assert out.shape == (B, D) and steps.shape == (B, T + 1, D)


def test_memory_fusion_shapes(small_cfg):
    D = small_cfg.d_model
    out = MemoryFusion(small_cfg)(torch.randn(B, D), torch.randn(B, 3, D),
                                  torch.zeros(B, 3, dtype=torch.bool))
    assert out.context.shape == (B, D)
    assert out.gate.shape == (B,)
    assert out.attention.shape == (B, 4)
    assert torch.allclose(out.attention.sum(-1), torch.ones(B), atol=1e-5)


def test_memory_fusion_zero_memories(small_cfg):
    D = small_cfg.d_model
    out = MemoryFusion(small_cfg)(torch.randn(B, D), torch.zeros(B, 0, D))
    assert out.context.shape == (B, D) and out.attention.shape == (B, 1)


def test_action_policy_shapes(small_cfg):
    D = small_cfg.d_model
    pol = ActionPolicy(small_cfg)
    logits = pol(torch.randn(B, D), torch.randn(B, 4, D), torch.randint(0, 6, (B, 4)),
                 torch.zeros(B, 4, dtype=torch.bool))
    assert logits.shape == (B, 5)
    assert pol(torch.randn(B, D), torch.zeros(B, 0, D)).shape == (B, 1)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_full_model_output_dict(small_cfg, kind):
    small_cfg.temporal_kind = kind
    model = build_model(small_cfg)
    out = model(_batch(small_cfg))
    D = small_cfg.d_model
    assert out["screen_state"].shape == (B, D)
    assert out["temporal_state"].shape == (B, D)
    assert out["personalized_context"].shape == (B, D)
    assert out["action_logits"].shape == (B, 4 + 1)
    assert out["memory_gate"].shape == (B,)
    assert out["memory_attention"].shape == (B, 3 + 1)


def test_encode_screen_and_query_embedding(small_cfg):
    model = build_model(small_cfg).eval()
    b = _batch(small_cfg)
    D = small_cfg.d_model
    assert model.encode_screen(b.images, b.ocr_ids, b.ocr_boxes, b.ocr_padding_mask).shape == (B, D)
    q = model.query_embedding(b)
    assert q.shape == (B, D)
    assert torch.allclose(q.norm(dim=-1), torch.ones(B), atol=1e-5)
    # encode_screen output is a valid history step
    with torch.no_grad():
        s = model.encode_screen(b.images, b.ocr_ids, b.ocr_boxes, b.ocr_padding_mask)
        b.history = s.unsqueeze(1).repeat(1, small_cfg.history_len, 1)
        assert model(b)["action_logits"].shape == (B, 5)


def test_pretrained_vit_adapter_shapes(small_cfg):
    pytest.importorskip("torchvision")
    from soma.models import PretrainedViTEncoder
    enc = PretrainedViTEncoder(small_cfg, weights=None)          # no download
    out = enc(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 196, small_cfg.d_model)
    assert all(not p.requires_grad for p in enc.backbone.parameters())
    assert all(p.requires_grad for p in enc.projector.parameters())
    enc.train()
    assert not enc.backbone.training
