"""Stage D: controlled rollout of a fine-tuned candidate with deterministic rollback.

``ControlledRollout.run`` (milestone M4):

1. Refuses to run without ``Scope.MODEL_TRAINING`` consent.
2. Snapshots the deployed (base) weights in memory and to ``base.pt``.
3. Trains a *copy* on consented, unambiguous feedback plus replay samples
   (anchors from prior competencies included) with a seeded train/val split.
4. Evaluates base and candidate on a held-out **personalized** set and on a
   **prior-competency** set (forgetting).
5. Deploys only if the personalized metric improves by at least
   ``min_improvement`` (and strictly > 0), forgetting is at most
   ``max_forgetting``, and the unwanted-suggestion rate does not worsen by
   more than ``unwanted_tolerance`` on either set. Non-finite metrics fail.
6. Otherwise restores the exact base weights (bitwise) into the live model.

Every run writes a decision record (metrics only, no content) as JSON.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from soma.config import ModelConfig, TrainConfig
from soma.consent import ConsentManager, Scope
from soma.learning.datasets import SomaDataset, SomaItem, make_loader, seeded_split
from soma.learning.evaluator import evaluate
from soma.learning.feedback import LabeledExample
from soma.learning.replay_buffer import ReplayBuffer
from soma.learning.trainer import Trainer
from soma.types import now_ms


@dataclass
class RolloutDecision:
    """Outcome of one rollout attempt (JSON-serializable, metrics only)."""

    deployed: bool
    reason: str
    metric: str = "top1"
    improvement: float = math.nan
    forgetting: float = math.nan
    unwanted_delta_personal: float = math.nan
    unwanted_delta_prior: float = math.nan
    base_personal: dict[str, float] = field(default_factory=dict)
    candidate_personal: dict[str, float] = field(default_factory=dict)
    base_prior: dict[str, float] = field(default_factory=dict)
    candidate_prior: dict[str, float] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    n_feedback: int = 0
    n_replay: int = 0
    gates: dict[str, float] = field(default_factory=dict)
    base_checkpoint: str | None = None
    deployed_checkpoint: str | None = None
    decision_path: str | None = None
    created_ms: int = field(default_factory=now_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


class ControlledRollout:
    """Guarded fine-tune -> evaluate -> deploy-or-rollback around a live model."""

    def __init__(self, model: nn.Module, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 consent: ConsentManager, run_dir: str | Path, *,
                 device: torch.device | str = "cpu", trainable: str = "all",
                 metric: str = "top1", unwanted_tolerance: float = 0.02,
                 replay_ratio: float = 3.0, max_steps: int | None = None,
                 time_budget_s: float | None = None, eval_batch_size: int = 64) -> None:
        self.model = model
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.consent = consent
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.trainable = trainable
        self.metric = metric
        self.unwanted_tolerance = float(unwanted_tolerance)
        self.replay_ratio = float(replay_ratio)
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        self.eval_batch_size = int(eval_batch_size)
        self.base_state: dict[str, torch.Tensor] | None = None
        self.runs = 0

    # ------------------------------------------------------------ weights
    def snapshot(self) -> dict[str, torch.Tensor]:
        """Exact detached copy of the live model's weights."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def rollback(self) -> None:
        """Restore the base weights captured at the start of the last run (bitwise)."""
        if self.base_state is None:
            raise RuntimeError("no base snapshot to roll back to")
        self.model.load_state_dict(self.base_state)

    def _eval(self, model: nn.Module, items: Sequence[SomaItem]) -> dict[str, float]:
        return evaluate(model, make_loader(list(items), self.eval_batch_size), device=self.device)

    # ------------------------------------------------------------ run
    def run(self, feedback: Sequence[LabeledExample], replay: ReplayBuffer | None,
            personal_holdout: Sequence[SomaItem], prior_holdout: Sequence[SomaItem]) -> RolloutDecision:
        """Train a candidate and deploy it only if every gate passes."""
        self.consent.require(Scope.MODEL_TRAINING)
        self.runs += 1
        run_dir = self.run_dir / f"rollout_{self.runs:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.train_cfg
        gates = {"min_improvement": float(cfg.min_improvement),
                 "max_forgetting": float(cfg.max_forgetting),
                 "unwanted_tolerance": self.unwanted_tolerance}

        usable = [ex for ex in feedback if ex.usable_for_training]
        self.base_state = self.snapshot()
        from soma.models.checkpoint import copy_model, save_checkpoint

        base_ckpt = save_checkpoint(self.model, run_dir / "base.pt", self.model_cfg,
                                    extra={"role": "base"})
        decision = RolloutDecision(deployed=False, reason="", metric=self.metric, gates=gates,
                                   n_feedback=len(usable), base_checkpoint=str(base_ckpt))
        if not usable:
            decision.reason = "no consented, unambiguous feedback"
            return self._finish(decision, run_dir)

        g = torch.Generator().manual_seed(int(cfg.seed))
        n_replay = int(round(len(usable) * self.replay_ratio)) if replay is not None else 0
        replayed = replay.sample(n_replay, g) if replay is not None and n_replay > 0 else []
        decision.n_replay = len(replayed)
        items = [ex.item for ex in usable] + [ex.item for ex in replayed]
        train_set, val_set = seeded_split(SomaDataset(items), cfg.val_fraction, seed=cfg.seed)

        candidate = copy_model(self.model)
        trainer = Trainer(candidate, cfg, self.device, trainable=self.trainable,
                          run_dir=run_dir / "candidate", model_cfg=self.model_cfg,
                          max_steps=self.max_steps, time_budget_s=self.time_budget_s)
        train_loader = make_loader(train_set, cfg.batch_size, shuffle=True, seed=cfg.seed)
        val_loader = make_loader(val_set, self.eval_batch_size) if len(val_set) else None
        result = trainer.fit(train_loader, val_loader)
        decision.train = {k: v for k, v in result.to_dict().items() if k != "history"}

        decision.base_personal = self._eval(self.model, personal_holdout)
        decision.candidate_personal = self._eval(candidate, personal_holdout)
        decision.base_prior = self._eval(self.model, prior_holdout)
        decision.candidate_prior = self._eval(candidate, prior_holdout)
        m = self.metric
        decision.improvement = decision.candidate_personal[m] - decision.base_personal[m]
        decision.forgetting = decision.base_prior[m] - decision.candidate_prior[m]
        uw = "unwanted_suggestion_rate"
        decision.unwanted_delta_personal = decision.candidate_personal[uw] - decision.base_personal[uw]
        decision.unwanted_delta_prior = decision.candidate_prior[uw] - decision.base_prior[uw]

        failures = []
        finite = _finite(decision.improvement, decision.forgetting, decision.unwanted_delta_personal,
                         decision.unwanted_delta_prior, decision.candidate_personal["nll"],
                         decision.candidate_prior["nll"])
        if not finite:
            failures.append("non-finite metrics")
        else:
            if not (decision.improvement > 0 and decision.improvement >= gates["min_improvement"]):
                failures.append(f"improvement {decision.improvement:+.4f} below gate")
            if decision.forgetting > gates["max_forgetting"]:
                failures.append(f"forgetting {decision.forgetting:.4f} > {gates['max_forgetting']}")
            if max(decision.unwanted_delta_personal, decision.unwanted_delta_prior) > self.unwanted_tolerance:
                failures.append("unwanted-suggestion rate worsened beyond tolerance")

        if failures:
            self.rollback()
            decision.reason = "rolled back: " + "; ".join(failures)
        else:
            self.model.load_state_dict(copy.deepcopy(candidate.state_dict()))
            decision.deployed = True
            decision.reason = "deployed: all gates passed"
            decision.deployed_checkpoint = str(save_checkpoint(
                self.model, run_dir / "deployed.pt", self.model_cfg,
                extra={"role": "deployed", "improvement": float(decision.improvement)}))
        return self._finish(decision, run_dir)

    def _finish(self, decision: RolloutDecision, run_dir: Path) -> RolloutDecision:
        path = run_dir / "decision.json"
        decision.decision_path = str(path)
        path.write_text(json.dumps(decision.to_dict(), indent=2, allow_nan=True), encoding="utf-8")
        return decision
