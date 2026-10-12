"""Checkpoint round-trips, determinism, gradients and mixed precision."""

from __future__ import annotations

import pytest
import torch

from soma.config import small_model_config
from soma.models import (CHECKPOINT_VERSION, build_model, copy_model, load_checkpoint,
                         random_batch, save_checkpoint)


def _batch(seed: int = 1, **kw):
    return random_batch(2, generator=torch.Generator().manual_seed(seed), **kw)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_save_load_preserves_eval_outputs(tmp_path, kind):
    cfg = small_model_config()
    cfg.temporal_kind = kind
    model = build_model(cfg).eval()
    b = _batch()
    with torch.no_grad():
        before = model(b)
    path = save_checkpoint(model, tmp_path / "ckpt" / "m.pt", cfg, extra={"step": 3, "note": "x"})
    loaded, cfg2, extra = load_checkpoint(path)
    assert cfg2 == cfg and extra == {"step": 3, "note": "x"}
    assert not loaded.training
    with torch.no_grad():
        after = loaded(b)
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_checkpoint_payload_is_plain(tmp_path):
    cfg = small_model_config()
    path = save_checkpoint(build_model(cfg), tmp_path / "m.pt", cfg)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"state_dict", "config", "version", "extra"}
    assert payload["version"] == CHECKPOINT_VERSION
    assert payload["config"]["d_model"] == cfg.d_model and payload["extra"] == {}
    assert not (tmp_path / "m.pt.tmp").exists()


def test_load_rejects_unknown_version(tmp_path):
    cfg = small_model_config()
    path = save_checkpoint(build_model(cfg), tmp_path / "m.pt", cfg)
    payload = torch.load(path, weights_only=True)
    payload["version"] = "other"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_copy_model_is_independent():
    cfg = small_model_config()
    model = build_model(cfg).eval()
    clone = copy_model(model)
    b = _batch()
    with torch.no_grad():
        assert torch.equal(model(b)["action_logits"], clone(b)["action_logits"])
        next(clone.action_policy.parameters()).add_(1.0)
        assert not torch.equal(model(b)["action_logits"], clone(b)["action_logits"])


def test_seeded_builds_are_deterministic():
    cfg = small_model_config()
    torch.manual_seed(123)
    a = build_model(cfg).eval()
    torch.manual_seed(123)
    c = build_model(cfg).eval()
    b = _batch()
    with torch.no_grad():
        assert torch.equal(a(b)["action_logits"], c(b)["action_logits"])
        assert torch.equal(a(b)["action_logits"], a(b)["action_logits"])


def test_gradients_flow_to_every_trainable_module():
    cfg = small_model_config()
    model = build_model(cfg).train()
    out = model(_batch())
    loss = torch.nn.functional.cross_entropy(out["action_logits"], torch.tensor([0, 2]))
    loss.backward()
    for name in ("vision_encoder", "text_layout_encoder", "screen_fusion", "temporal_encoder",
                 "memory_fusion", "action_policy"):
        grads = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name
    for p in model.action_policy.scorer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.action_policy.abstain_bias.grad is not None


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_cpu_bfloat16_autocast(kind):
    cfg = small_model_config()
    cfg.temporal_kind = kind
    model = build_model(cfg).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(_batch(empty_ocr=True, empty_memory=True))
        loss = torch.nn.functional.cross_entropy(out["action_logits"].float(), torch.tensor([0, 1]))
    loss.backward()
    assert all(torch.isfinite(v.float()).all() for v in out.values())
    assert torch.isfinite(loss)


def test_short_training_run_reduces_loss():
    cfg = small_model_config()
    model = build_model(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    b, target = _batch(), torch.tensor([1, 3])
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(b)["action_logits"], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
