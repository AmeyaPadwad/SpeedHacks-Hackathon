"""The production-sized ``ModelConfig`` (D=512, 448x448 images) builds and runs end to end."""

from __future__ import annotations

import torch

from soma.config import ModelConfig, load_config
from soma.models import build_model, random_batch


def test_default_config_forward_and_param_budget():
    cfg = ModelConfig()
    model = build_model(cfg).eval()
    b = random_batch(1, image_size=cfg.image_size, L=16, T=cfg.history_len, K=cfg.max_memories,
                     A=cfg.max_candidates, D=cfg.d_model, vocab_size=cfg.vocab_size,
                     n_apps=cfg.n_apps, n_action_types=cfg.n_action_types)
    with torch.inference_mode():
        out = model(b)
    assert out["action_logits"].shape == (1, cfg.max_candidates + 1)
    assert model.vision_encoder.num_tokens == 196
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 60_000_000, n_params        # stays desktop-sized


def test_yaml_profiles_build():
    for name in ("configs/dev.yaml", "configs/train.yaml"):
        try:
            cfg = load_config(name).model
        except FileNotFoundError:
            continue
        cfg.vision_backbone = "scratch"           # never download weights in tests
        assert build_model(cfg) is not None
