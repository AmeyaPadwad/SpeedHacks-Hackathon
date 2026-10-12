"""Stage D controlled rollout on synthetic data (synthetic != real-world quality)."""

import json
from dataclasses import replace

import pytest
import torch

from soma.config import TrainConfig, small_model_config
from soma.consent import ConsentError, ConsentManager, Scope
from soma.learning.datasets import make_loader
from soma.learning.feedback import as_examples
from soma.learning.replay_buffer import ReplayBuffer
from soma.learning.rollout import ControlledRollout
from soma.learning.synthetic_data import generate_personalized, generate_synthetic
from soma.learning.trainer import Trainer
from soma.models.checkpoint import load_checkpoint
from soma.models.soma_model import build_model


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = small_model_config()
    base = build_model(cfg)
    Trainer(base, TrainConfig(seed=0, batch_size=32, epochs=8, lr=1e-3, use_amp=False), "cpu",
            patience=100).fit(make_loader(generate_synthetic(600, cfg, seed=0), 32, shuffle=True))
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    feedback = as_examples(generate_personalized(160, cfg, seed=1), source="feedback")
    replay = ReplayBuffer(capacity=600, anchor_capacity=600, anchor_fraction=1.0, seed=0)
    for ex in as_examples(generate_synthetic(600, cfg, seed=2), source="anchor"):
        replay.add_anchor(ex)
    return dict(cfg=cfg, base_state=base_state, feedback=feedback, replay=replay,
                personal=generate_personalized(150, cfg, seed=11),
                prior=generate_synthetic(300, cfg, seed=12))


def _consent():
    c = ConsentManager()
    c.grant(Scope.MODEL_TRAINING)
    return c


def _model(setup):
    m = build_model(setup["cfg"])
    m.load_state_dict(setup["base_state"])
    return m


GOOD = TrainConfig(seed=0, batch_size=32, epochs=6, lr=1e-3, use_amp=False, val_fraction=0.15,
                   max_forgetting=0.05, min_improvement=0.05)


def _same(model, state):
    return all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


def test_refuses_without_training_consent(setup, tmp_path):
    c = ConsentManager()
    c.grant(Scope.FEEDBACK_STORAGE)
    ro = ControlledRollout(_model(setup), setup["cfg"], GOOD, c, tmp_path)
    with pytest.raises(ConsentError):
        ro.run(setup["feedback"], setup["replay"], setup["personal"], setup["prior"])
    assert not any(tmp_path.iterdir())


def test_deploys_a_genuinely_better_candidate(setup, tmp_path):
    model = _model(setup)
    ro = ControlledRollout(model, setup["cfg"], GOOD, _consent(), tmp_path, unwanted_tolerance=0.05)
    d = ro.run(setup["feedback"], setup["replay"], setup["personal"], setup["prior"])
    print("personal", d.base_personal["top1"], "->", d.candidate_personal["top1"],
          "prior", d.base_prior["top1"], "->", d.candidate_prior["top1"])
    assert d.deployed, d.reason
    assert d.improvement >= 0.05 and d.forgetting <= 0.05
    assert not _same(model, setup["base_state"])
    deployed, _, extra = load_checkpoint(d.deployed_checkpoint)
    assert _same(deployed, model.state_dict()) and extra["role"] == "deployed"
    record = json.loads(open(d.decision_path, encoding="utf-8").read())
    assert record["deployed"] is True and record["n_replay"] == 3 * len(setup["feedback"])


@pytest.mark.parametrize("sabotage", ["huge_lr", "shuffled_labels"])
def test_rolls_back_a_sabotaged_candidate_bitwise(setup, tmp_path, sabotage):
    model = _model(setup)
    feedback, cfg = setup["feedback"], GOOD
    if sabotage == "huge_lr":
        cfg = replace(GOOD, lr=5.0, grad_clip=100.0)
    else:
        g = torch.Generator().manual_seed(0)
        feedback = as_examples([ex.item.clone(target=int(torch.randint(0, ex.item.n_candidates + 1, (1,), generator=g)))
                                for ex in feedback], source="feedback")
    ro = ControlledRollout(model, setup["cfg"], cfg, _consent(), tmp_path, unwanted_tolerance=0.05)
    d = ro.run(feedback, setup["replay"], setup["personal"], setup["prior"])
    assert not d.deployed and d.reason.startswith("rolled back")
    assert _same(model, setup["base_state"])             # bitwise restore
    base_ckpt, _, _ = load_checkpoint(d.base_checkpoint)
    assert _same(base_ckpt, setup["base_state"])
    assert json.loads(open(d.decision_path, encoding="utf-8").read())["deployed"] is False


def test_ignored_or_unconsented_feedback_does_not_train(setup, tmp_path):
    model = _model(setup)
    fb = as_examples(generate_personalized(10, setup["cfg"], seed=3))
    for ex in fb:
        ex.consented = False
    ro = ControlledRollout(model, setup["cfg"], GOOD, _consent(), tmp_path)
    d = ro.run(fb, setup["replay"], setup["personal"], setup["prior"])
    assert not d.deployed and d.n_feedback == 0 and "no consented" in d.reason
    assert _same(model, setup["base_state"])
    # Explicit rollback is always available and exact.
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    ro.rollback()
    assert _same(model, setup["base_state"])
