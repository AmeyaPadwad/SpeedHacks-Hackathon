"""Trainer on synthetic data. Synthetic accuracy is a smoke signal, not real-world quality."""

import json

import pytest
import torch

from soma.config import TrainConfig
from soma.learning.datasets import make_loader
from soma.learning.evaluator import evaluate, evaluate_no_memory
from soma.learning.synthetic_data import generate_synthetic
from soma.learning.trainer import Trainer, set_trainable
from soma.models.checkpoint import load_checkpoint
from soma.models.soma_model import build_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from soma.config import small_model_config

    torch.manual_seed(0)
    cfg = small_model_config()
    train = generate_synthetic(600, cfg, seed=0)
    test = generate_synthetic(300, cfg, seed=123)
    model = build_model(cfg)
    before = evaluate(model, make_loader(test, 64))
    run_dir = tmp_path_factory.mktemp("run")
    tc = TrainConfig(seed=0, batch_size=32, epochs=8, lr=1e-3, use_amp=True)
    trainer = Trainer(model, tc, "cpu", run_dir=run_dir, patience=4)
    result = trainer.fit(make_loader(train, 32, shuffle=True, seed=0), make_loader(test, 64))
    after = evaluate(model, make_loader(test, 64))
    no_mem = evaluate_no_memory(model, make_loader(test, 64))
    return dict(cfg=cfg, model=model, result=result, before=before, after=after, no_mem=no_mem,
                run_dir=run_dir, test=test)


def test_training_reduces_loss_and_beats_chance(trained):
    res, after = trained["result"], trained["after"]
    assert res.last_train_loss < 0.5 * res.first_train_loss
    majority = after["silence_fraction"]           # always-silent baseline
    assert after["top1"] > majority + 0.25, (after, majority)
    assert after["top1"] > trained["before"]["top1"] + 0.2
    assert after["nll"] < trained["before"]["nll"]


def test_memory_ablation_shows_a_difference(trained):
    assert trained["after"]["top1"] - trained["no_mem"]["top1"] > 0.03
    assert trained["no_mem"]["missed_suggestion_rate"] > trained["after"]["missed_suggestion_rate"]


def test_best_checkpoint_and_loss_curve_written(trained):
    res, run_dir = trained["result"], trained["run_dir"]
    assert res.checkpoint_path and (run_dir / "best.pt").exists()
    model, cfg, extra = load_checkpoint(run_dir / "best.pt")
    assert cfg == trained["cfg"] and extra["epoch"] == res.best_epoch
    # restore_best: the in-memory model holds the best weights.
    sd = trained["model"].state_dict()
    assert all(torch.equal(v, sd[k]) for k, v in model.state_dict().items())
    lines = [json.loads(l) for l in (run_dir / "loss_curve.jsonl").read_text().splitlines()]
    assert any(l["kind"] == "epoch" and "val_loss" in l for l in lines)
    assert any(l["kind"] == "step" for l in lines)


def test_policy_head_only_leaves_backbone_untouched(small_cfg):
    model = build_model(small_cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    items = generate_synthetic(64, small_cfg, seed=1)
    tr = Trainer(model, TrainConfig(epochs=1, lr=1e-2, use_amp=False), "cpu", trainable="policy_head")
    tr.fit(make_loader(items, 16))
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith("action_policy.") for k in changed)
    names = set_trainable(model, "fusion+policy")
    assert any(n.startswith("memory_fusion.") for n in names)
    assert not any(n.startswith("vision_encoder.") for n in names)
    with pytest.raises(ValueError):
        set_trainable(model, "bogus")


def test_step_and_time_budgets(small_cfg):
    items = generate_synthetic(64, small_cfg, seed=2)
    model = build_model(small_cfg)
    res = Trainer(model, TrainConfig(epochs=5, use_amp=False), "cpu", max_steps=3).fit(make_loader(items, 8))
    assert res.steps == 3 and res.stopped_reason == "max_steps"
    res = Trainer(model, TrainConfig(epochs=5, use_amp=False), "cpu", time_budget_s=0.0).fit(make_loader(items, 8))
    assert res.stopped_reason == "time_budget" and res.steps == 1    # stops right after the first step


def test_training_is_seeded(small_cfg):
    items = generate_synthetic(48, small_cfg, seed=3)

    def run():
        torch.manual_seed(0)
        m = build_model(small_cfg)
        Trainer(m, TrainConfig(seed=5, epochs=1, use_amp=False), "cpu").fit(make_loader(items, 16, shuffle=True))
        return m.state_dict()

    a, b = run(), run()
    assert all(torch.equal(a[k], b[k]) for k in a)
