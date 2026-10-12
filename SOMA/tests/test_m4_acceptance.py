"""Milestone M4 acceptance on synthetic data (build plan section 11).

Accept when: held-out personalized ranking improves over a fixed baseline;
forgetting and intervention (unwanted-suggestion) rate do not worsen beyond
defined tolerances; a failed update rolls back deterministically.

Synthetic data only: this proves the machinery, not real-world quality.
"""

import json
from dataclasses import replace

import pytest
import torch

from scripts import evaluate as evaluate_script
from soma.config import TrainConfig, small_model_config
from soma.consent import ConsentManager, Scope
from soma.learning import (ControlledRollout, ReplayBuffer, Trainer, as_examples, evaluate,
                           generate_personalized, generate_synthetic, make_loader)
from soma.learning.ablations import TypePriorityBaseline
from soma.models import build_model, copy_model, load_checkpoint

# Tolerances the M4 gates are held to.
MIN_IMPROVEMENT = 0.05
MAX_FORGETTING = 0.05
UNWANTED_TOLERANCE = 0.05
GOOD = TrainConfig(seed=0, batch_size=32, epochs=6, lr=1e-3, use_amp=False, val_fraction=0.15,
                   max_forgetting=MAX_FORGETTING, min_improvement=MIN_IMPROVEMENT)
SABOTAGED = replace(GOOD, lr=5.0, grad_clip=100.0)


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    cfg = small_model_config()
    base = build_model(cfg)
    Trainer(base, TrainConfig(seed=0, batch_size=32, epochs=8, lr=1e-3, use_amp=False), "cpu",
            patience=100).fit(make_loader(generate_synthetic(600, cfg, seed=0), 32, shuffle=True))
    replay = ReplayBuffer(capacity=600, anchor_capacity=600, anchor_fraction=1.0, seed=0)
    for ex in as_examples(generate_synthetic(600, cfg, seed=2), source="anchor"):
        replay.add_anchor(ex)
    return dict(cfg=cfg, base_state={k: v.clone() for k, v in base.state_dict().items()},
                feedback=as_examples(generate_personalized(160, cfg, seed=1), source="feedback"),
                replay=replay, personal=generate_personalized(150, cfg, seed=11),
                prior=generate_synthetic(300, cfg, seed=12))


def _fresh(world):
    m = build_model(world["cfg"])
    m.load_state_dict(world["base_state"])
    return m


def _same(model, state):
    sd = model.state_dict()
    return sd.keys() == state.keys() and all(torch.equal(sd[k], state[k]) for k in sd)


def _consent():
    c = ConsentManager()
    c.grant(Scope.MODEL_TRAINING)
    return c


def _rollout(world, tmp, cfg):
    model = _fresh(world)
    ro = ControlledRollout(model, world["cfg"], cfg, _consent(), tmp,
                           unwanted_tolerance=UNWANTED_TOLERANCE)
    return model, ro.run(world["feedback"], world["replay"], world["personal"], world["prior"])


def _eval(model, items):
    return evaluate(model, make_loader(items, 64), device="cpu")


def test_personalized_ranking_improves_over_fixed_baseline_within_tolerances(world, tmp_path):
    fixed = copy_model(_fresh(world)).eval()               # the frozen baseline, never updated
    model, d = _rollout(world, tmp_path, GOOD)
    assert d.deployed, d.reason

    pers_fixed, pers_new = _eval(fixed, world["personal"]), _eval(model, world["personal"])
    prior_fixed, prior_new = _eval(fixed, world["prior"]), _eval(model, world["prior"])
    rules = _eval(TypePriorityBaseline(), world["personal"])
    print(f"personal top1 fixed={pers_fixed['top1']:.3f} rules={rules['top1']:.3f} "
          f"deployed={pers_new['top1']:.3f}; prior top1 {prior_fixed['top1']:.3f} -> {prior_new['top1']:.3f}")

    # Held-out personalized ranking beats the fixed baseline and the scripted rule.
    assert pers_new["top1"] - pers_fixed["top1"] >= MIN_IMPROVEMENT
    assert pers_new["top1"] > rules["top1"]
    # Forgetting and intervention rate stay within tolerance.
    assert prior_fixed["top1"] - prior_new["top1"] <= MAX_FORGETTING
    assert pers_new["unwanted_suggestion_rate"] - pers_fixed["unwanted_suggestion_rate"] <= UNWANTED_TOLERANCE
    assert prior_new["unwanted_suggestion_rate"] - prior_fixed["unwanted_suggestion_rate"] <= UNWANTED_TOLERANCE
    # The decision record agrees with an independent re-evaluation.
    assert d.candidate_personal["top1"] == pytest.approx(pers_new["top1"])
    assert d.forgetting == pytest.approx(prior_fixed["top1"] - prior_new["top1"])
    record = json.loads(open(d.decision_path, encoding="utf-8").read())
    assert record["deployed"] and record["gates"]["max_forgetting"] == MAX_FORGETTING

    # The deployed checkpoint is evaluable by the offline evaluation script.
    report = evaluate_script.run(d.deployed_checkpoint, 120, 21, device="cpu")
    assert report["results"]["full"]["top1"] > report["results"]["rules_type_priority"]["top1"]


def test_failed_update_rolls_back_deterministically(world, tmp_path):
    runs = [_rollout(world, tmp_path / f"run{i}", SABOTAGED) for i in range(2)]
    for model, d in runs:
        assert not d.deployed and d.reason.startswith("rolled back")
        assert _same(model, world["base_state"])              # bitwise restore
        base_ckpt, _, extra = load_checkpoint(d.base_checkpoint)
        assert extra["role"] == "base" and _same(base_ckpt, world["base_state"])
        assert d.deployed_checkpoint is None
    # Same inputs -> same decision, same metrics, same restored weights.
    (_, a), (_, b) = runs
    keys = ("reason", "improvement", "forgetting", "unwanted_delta_personal", "unwanted_delta_prior",
            "candidate_personal", "candidate_prior", "n_feedback", "n_replay")
    assert {k: getattr(a, k) for k in keys} == {k: getattr(b, k) for k in keys}
    # The live model still behaves exactly like the fixed baseline.
    fixed = _fresh(world).eval()
    assert _eval(runs[0][0], world["personal"]) == _eval(fixed, world["personal"])
