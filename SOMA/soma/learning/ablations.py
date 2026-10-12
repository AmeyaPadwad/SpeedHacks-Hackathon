"""Scripted baselines and M5 one-variable-at-a-time ablations (build plan sections 11-12).

.. warning::
   Everything here runs on :mod:`soma.learning.synthetic_data`. Synthetic
   numbers show whether a component carries signal *in the synthetic
   generator*; they are **not** evidence of real-world quality.

Baselines (logits ``[B, A+1]``, index 0 = ``NO_SUGGESTION``):

* :class:`TypePriorityBaseline` - screen-blind scripted rule: always propose
  the open candidate whose action type ranks highest in a fixed priority list,
  stay silent only when no candidate is open.
* :class:`KeywordRulesBaseline` - OCR-keyword rules that mirror the synthetic
  generator (an "oracle-ish" upper bound on rule-generated data, not a fair
  real-world baseline).

Ablations (each trains the same budget from the same seed, changes one thing):

* ``memory``     none | retrieval_only | gated
* ``temporal``   current_frame | temporal_context
* ``ocr``        ocr_visual | visual_only
* ``temporal_kind`` gru | transformer
* ``adaptation`` base (no adaptation) | frozen_base_online_ranker | finetuned_policy_head

Ablations are implemented as batch transforms applied identically at train
and eval time (masking memory/history/OCR) or as model-config overrides, so
the model code is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from torch import nn

from soma.config import ModelConfig, TrainConfig, small_model_config
from soma.device import seed_everything
from soma.learning.datasets import SomaItem, make_loader
from soma.learning.evaluator import evaluate
from soma.learning.online_ranker import OnlinePreferenceRanker
from soma.learning.synthetic_data import (SCENARIOS, TYPE_INDEX, generate_personalized,
                                          generate_synthetic)
from soma.learning.trainer import Trainer
from soma.models.batch import SomaBatch
from soma.models.checkpoint import copy_model
from soma.models.soma_model import build_model
from soma.models.tokenizer import encode
from soma.types import ACTION_TYPES, ActionType, FeedbackOutcome

SYNTHETIC_NOTICE = ("SYNTHETIC DATA ONLY: these results come from soma.learning.synthetic_data "
                    "and are not evidence of real-world quality.")

#: Metrics every ablation row reports.
REPORT_KEYS = ("top1", "unwanted_suggestion_rate", "ece", "nll")

BatchTransform = Callable[[SomaBatch, torch.Tensor], SomaBatch]


# ---------------------------------------------------------------- batch transforms

def _replace(batch: SomaBatch, **fields_: torch.Tensor) -> SomaBatch:
    data = dict(batch.__dict__)
    data.update(fields_)
    return SomaBatch(**data)


def mask_memory(batch: SomaBatch, targets: torch.Tensor | None = None) -> SomaBatch:
    """No memory at all: every memory slot zeroed and masked."""
    return _replace(batch, memory_embeddings=torch.zeros_like(batch.memory_embeddings),
                    memory_padding_mask=torch.ones_like(batch.memory_padding_mask, dtype=torch.bool))


def retrieval_only(batch: SomaBatch, targets: torch.Tensor | None = None) -> SomaBatch:
    """Memory is used only to *build candidates*, never fused into the context.

    * ``COMPARE_WITH_PRIOR_WORK`` candidates are dropped (masked) when retrieval
      returned nothing - the way a retrieval-backed candidate generator works.
    * Open compare candidates get the mean retrieved-memory embedding added
      (memory as a candidate feature).
    * All memory slots are then masked, so memory fusion sees only ``NO_MEMORY``.

    The labeled target slot is never masked.
    """
    compare = TYPE_INDEX[ActionType.COMPARE_WITH_PRIOR_WORK]
    valid = ~batch.memory_padding_mask.bool()                                  # [B, K]
    has_mem = valid.any(dim=1)                                                 # [B]
    n = valid.sum(dim=1, keepdim=True).clamp_min(1).to(batch.memory_embeddings.dtype)
    mem_mean = (batch.memory_embeddings * valid.unsqueeze(-1)).sum(dim=1) / n   # [B, D]
    is_compare = (batch.candidate_types == compare) & ~batch.candidate_mask.bool()
    drop = is_compare & ~has_mem.unsqueeze(1)
    if targets is not None:
        A = batch.candidate_mask.shape[1]
        slots = torch.arange(1, A + 1, device=targets.device).unsqueeze(0)
        drop = drop & (slots != targets.unsqueeze(1))
    emb = batch.candidate_embeddings + is_compare.unsqueeze(-1) * mem_mean.unsqueeze(1)
    out = _replace(batch, candidate_embeddings=emb, candidate_mask=batch.candidate_mask.bool() | drop)
    return mask_memory(out)


def mask_history(batch: SomaBatch, targets: torch.Tensor | None = None) -> SomaBatch:
    """Current frame only: every history step masked."""
    return _replace(batch, history=torch.zeros_like(batch.history),
                    history_padding_mask=torch.ones_like(batch.history_padding_mask, dtype=torch.bool))


def mask_ocr(batch: SomaBatch, targets: torch.Tensor | None = None) -> SomaBatch:
    """Visual-only screen understanding: every OCR token masked."""
    return _replace(batch, ocr_ids=torch.zeros_like(batch.ocr_ids),
                    ocr_boxes=torch.zeros_like(batch.ocr_boxes),
                    ocr_padding_mask=torch.ones_like(batch.ocr_padding_mask, dtype=torch.bool))


TRANSFORMS: dict[str, BatchTransform] = {
    "mask_memory": mask_memory, "retrieval_only": retrieval_only,
    "mask_history": mask_history, "mask_ocr": mask_ocr,
}


class TransformedLoader:
    """Re-iterable loader view that applies ``transform(batch, targets)`` to each batch."""

    def __init__(self, loader: Iterable, transform: BatchTransform | None) -> None:
        self.loader = loader
        self.transform = transform

    def __iter__(self) -> Iterator[tuple[SomaBatch, torch.Tensor, torch.Tensor]]:
        for batch, targets, weights in self.loader:
            if self.transform is not None:
                batch = self.transform(batch, targets)
            yield batch, targets, weights

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


# ---------------------------------------------------------------- scripted baselines

#: Fixed "most generically useful first" order used by :class:`TypePriorityBaseline`.
TYPE_PRIORITY: tuple[ActionType, ...] = (
    ActionType.EXPLAIN_ERROR, ActionType.OPEN_RELEVANT_FILE, ActionType.SUGGEST_NEXT_STEP,
    ActionType.COMPARE_WITH_PRIOR_WORK, ActionType.ASK_CLARIFYING_QUESTION,
)


class _ScriptedBaseline(nn.Module):
    """No learnable weights; ``forward(batch) -> {"action_logits": [B, A+1]}``."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    def logits(self, batch: SomaBatch) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, batch: SomaBatch) -> dict[str, torch.Tensor]:
        return {"action_logits": self.logits(batch)}


class TypePriorityBaseline(_ScriptedBaseline):
    """Screen-blind rule: speak with the highest-priority open candidate type."""

    def __init__(self, priority: Sequence[ActionType] = TYPE_PRIORITY,
                 abstain_logit: float = 0.5) -> None:
        super().__init__()
        n = len(ACTION_TYPES)
        table = torch.zeros(n)
        for rank, a in enumerate(priority):
            table[TYPE_INDEX[a]] = float(len(priority) - rank)
        self.register_buffer("type_score", table, persistent=False)
        self.abstain_logit = float(abstain_logit)

    def logits(self, batch: SomaBatch) -> torch.Tensor:
        types = batch.candidate_types.clamp(0, self.type_score.numel() - 1)
        scores = self.type_score[types].masked_fill(batch.candidate_mask.bool(), -1e4)
        abstain = torch.full((scores.shape[0], 1), self.abstain_logit, device=scores.device)
        return torch.cat([abstain, scores], dim=1)


class KeywordRulesBaseline(_ScriptedBaseline):
    """OCR-keyword rules mirroring the synthetic generator.

    A scenario fires when at least one of its keywords is among the (unpadded)
    OCR tokens; ``prior_work`` also needs at least one retrieved memory. The
    rule proposes the open candidate of the scenario's action type, otherwise
    stays silent. Because the synthetic labels are themselves rule-generated,
    this is close to an upper bound there and says nothing about real screens.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.rules: list[tuple[torch.Tensor, int, bool]] = []
        for sc in SCENARIOS.values():
            if not sc.keywords or sc.action is ActionType.NO_SUGGESTION:
                continue
            ids = torch.tensor([encode(w, cfg.vocab_size)[0] for w in sc.keywords], dtype=torch.long)
            self.rules.append((ids, TYPE_INDEX[sc.action], sc.name == "prior_work"))

    def logits(self, batch: SomaBatch) -> torch.Tensor:
        B, A = batch.candidate_mask.shape
        ocr = batch.ocr_ids.masked_fill(batch.ocr_padding_mask.bool(), -1)
        has_mem = (~batch.memory_padding_mask.bool()).any(dim=1)
        open_ = ~batch.candidate_mask.bool()
        scores = torch.full((B, A), -1e4, device=ocr.device)
        scores = torch.where(open_, torch.zeros_like(scores), scores)
        for ids, type_idx, needs_memory in self.rules:
            fired = torch.isin(ocr, ids.to(ocr.device)).any(dim=1)
            if needs_memory:
                fired = fired & has_mem
            hit = fired.unsqueeze(1) & open_ & (batch.candidate_types == type_idx)
            scores = torch.where(hit, torch.full_like(scores, 2.0), scores)
        abstain = torch.ones(B, 1, device=ocr.device)
        return torch.cat([abstain, scores], dim=1)


# ---------------------------------------------------------------- online-ranker helpers

def app_key(app_id: int) -> str:
    """Ranker app key from a hashed app bucket (batches carry buckets, not names)."""
    return f"bucket{int(app_id)}"


def _row_types(batch: SomaBatch, b: int) -> list[ActionType | None]:
    types = batch.candidate_types[b].tolist()
    mask = batch.candidate_mask[b].tolist()
    return [None if m else ACTION_TYPES[t] for t, m in zip(types, mask)]


def ranker_hook(ranker: OnlinePreferenceRanker) -> Callable[[SomaBatch, torch.Tensor], torch.Tensor]:
    """``logits_hook`` for :func:`soma.learning.evaluator.evaluate` adding ranker biases."""

    def hook(batch: SomaBatch, logits: torch.Tensor) -> torch.Tensor:
        bias = torch.stack([ranker.bias_for_types(_row_types(batch, b), app_key(int(batch.app_ids[b])))
                            for b in range(batch.batch_size)])
        bias = torch.nn.functional.pad(bias, (0, logits.shape[1] - bias.shape[1]))
        return OnlinePreferenceRanker.apply(logits, bias)

    return hook


@torch.no_grad()
def simulate_feedback(model: nn.Module, items: Sequence[SomaItem], ranker: OnlinePreferenceRanker,
                      device: torch.device | str = "cpu", batch_size: int = 64) -> int:
    """Feed simulated *explicit* feedback on the frozen model's suggestions to ``ranker``.

    Per item: if the model spoke, the shown suggestion is ACCEPTED when it is
    the label and REJECTED otherwise; when the label is a different speaking
    action, the user is simulated as picking that alternative (ACCEPTED).
    Silent predictions on silent labels produce no feedback (nothing shown).
    Returns the number of updates.
    """
    model.eval()
    hook = ranker_hook(ranker)
    n = 0
    for batch, targets, _ in make_loader(list(items), batch_size):
        batch = batch.to(device)
        logits = hook(batch, model(batch)["action_logits"].float().cpu())
        preds = logits.argmax(-1)
        for b in range(batch.batch_size):
            types, app = _row_types(batch, b), app_key(int(batch.app_ids[b]))
            p, t = int(preds[b]), int(targets[b])
            if p != 0 and types[p - 1] is not None:
                outcome = FeedbackOutcome.ACCEPTED if p == t else FeedbackOutcome.REJECTED
                n += ranker.update(app, types[p - 1], outcome)
            if t != 0 and p != t and types[t - 1] is not None:
                n += ranker.update(app, types[t - 1], FeedbackOutcome.ACCEPTED)
    return n


# ---------------------------------------------------------------- runner

@dataclass
class AblationBudget:
    """Identical training budget for every variant (small config, CPU-friendly)."""

    n_train: int = 1500
    n_eval: int = 500
    epochs: int = 6
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    eval_seed: int = 1
    n_personal: int = 200
    personal_epochs: int = 6
    device: str = "cpu"

    def train_cfg(self, epochs: int | None = None) -> TrainConfig:
        return TrainConfig(seed=self.seed, batch_size=self.batch_size,
                           epochs=self.epochs if epochs is None else epochs, lr=self.lr,
                           use_amp=False, val_fraction=0.0)


@dataclass(frozen=True)
class AblationSpec:
    group: str
    variant: str
    transform: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    note: str = ""


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("memory", "none", "mask_memory", note="memory masked at train and eval"),
    AblationSpec("memory", "retrieval_only", "retrieval_only",
                 note="memory only gates/features compare candidates; fusion sees NO_MEMORY"),
    AblationSpec("memory", "gated", None, note="full gated memory cross-attention"),
    AblationSpec("temporal", "current_frame", "mask_history", note="history fully masked"),
    AblationSpec("temporal", "temporal_context", None, note="history available"),
    AblationSpec("ocr", "visual_only", "mask_ocr", note="OCR tokens fully masked"),
    AblationSpec("ocr", "ocr_visual", None, note="OCR + visual tokens"),
    AblationSpec("temporal_kind", "gru", None, {"temporal_kind": "gru"}),
    AblationSpec("temporal_kind", "transformer", None, {"temporal_kind": "transformer"}),
)
#: Online-ranker settings compared in the ``adaptation`` group. The defaults are
#: deliberately gentle (bias ~ clipped Beta log-odds, at most ~log(n) in size);
#: ``scaled`` multiplies the log-odds so they can outweigh a confident base.
RANKER_VARIANTS: dict[str, dict[str, float]] = {
    "frozen_base_online_ranker": {},
    "frozen_base_online_ranker_scaled": {"scale": 4.0, "abstain_scale": 4.0, "max_bias": 20.0},
}
GROUPS: tuple[str, ...] = ("memory", "temporal", "ocr", "temporal_kind", "adaptation")


def _pick(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(metrics[k]) for k in REPORT_KEYS if k in metrics}


def run_variant(spec: AblationSpec, budget: AblationBudget,
                base_cfg: ModelConfig | None = None) -> dict[str, Any]:
    """Train one variant from scratch on the shared budget and evaluate it."""
    cfg = replace(base_cfg or small_model_config(), **spec.overrides)
    transform = TRANSFORMS[spec.transform] if spec.transform else None
    seed_everything(budget.seed)
    train_items = generate_synthetic(budget.n_train, cfg, seed=budget.seed)
    eval_items = generate_synthetic(budget.n_eval, cfg, seed=budget.eval_seed)
    seed_everything(budget.seed)
    model = build_model(cfg, pretrained=False)
    start = time.perf_counter()
    trainer = Trainer(model, budget.train_cfg(), budget.device, patience=10**9)
    loader = TransformedLoader(make_loader(train_items, budget.batch_size, shuffle=True,
                                           seed=budget.seed), transform)
    trainer.fit(loader)
    train_s = time.perf_counter() - start
    metrics = evaluate(model, TransformedLoader(make_loader(eval_items, 128), transform),
                       device=budget.device)
    return {"group": spec.group, "variant": spec.variant, "note": spec.note,
            **_pick(metrics), "train_s": round(train_s, 2)}


def run_adaptation(budget: AblationBudget, base_cfg: ModelConfig | None = None) -> list[dict[str, Any]]:
    """(e) frozen base + online ranker vs fine-tuned policy head, same personal data.

    Both start from the same generically trained base. Metrics are on a
    held-out personalized set; ``prior_top1`` is the generic holdout
    (forgetting = base prior_top1 - variant prior_top1).
    """
    cfg = base_cfg or small_model_config()
    seed_everything(budget.seed)
    generic = generate_synthetic(budget.n_train, cfg, seed=budget.seed)
    prior_eval = generate_synthetic(budget.n_eval, cfg, seed=budget.eval_seed)
    personal_train = generate_personalized(budget.n_personal, cfg, seed=budget.seed + 100)
    personal_eval = generate_personalized(budget.n_eval, cfg, seed=budget.eval_seed + 100)
    seed_everything(budget.seed)
    base = build_model(cfg, pretrained=False)
    Trainer(base, budget.train_cfg(), budget.device, patience=10**9).fit(
        make_loader(generic, budget.batch_size, shuffle=True, seed=budget.seed))

    def row(variant: str, model: nn.Module, hook: Any = None, note: str = "",
            train_s: float = 0.0) -> dict[str, Any]:
        pers = evaluate(model, make_loader(personal_eval, 128), device=budget.device, logits_hook=hook)
        prior = evaluate(model, make_loader(prior_eval, 128), device=budget.device, logits_hook=hook)
        return {"group": "adaptation", "variant": variant, "note": note, **_pick(pers),
                "prior_top1": float(prior["top1"]), "train_s": round(train_s, 2)}

    rows = [row("base_no_adaptation", base, note="generic base, no personal data")]

    for variant, kwargs in RANKER_VARIANTS.items():
        start = time.perf_counter()
        ranker = OnlinePreferenceRanker(**kwargs)
        n_updates = simulate_feedback(base, personal_train, ranker, budget.device)
        rows.append(row(variant, base, ranker_hook(ranker),
                        note=f"base weights frozen; ranker {kwargs or 'defaults'}; "
                             f"{n_updates} simulated explicit feedback updates",
                        train_s=time.perf_counter() - start))

    start = time.perf_counter()
    tuned = copy_model(base)
    replay = generate_synthetic(len(personal_train), cfg, seed=budget.seed + 200)
    Trainer(tuned, budget.train_cfg(budget.personal_epochs), budget.device,
            trainable="policy_head", patience=10**9).fit(
        make_loader(list(personal_train) + replay, budget.batch_size, shuffle=True, seed=budget.seed))
    rows.append(row("finetuned_policy_head", tuned,
                    note="policy head only; personal data + equal generic replay",
                    train_s=time.perf_counter() - start))
    return rows


def run_ablations(groups: Sequence[str] = GROUPS, budget: AblationBudget | None = None,
                  base_cfg: ModelConfig | None = None) -> dict[str, Any]:
    """Run the selected ablation groups; returns ``{notice, budget, rows}``."""
    budget = budget or AblationBudget()
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown ablation groups {sorted(unknown)}; choose from {GROUPS}")
    rows: list[dict[str, Any]] = []
    for spec in ABLATIONS:
        if spec.group in groups:
            rows.append(run_variant(spec, budget, base_cfg))
    if "adaptation" in groups:
        rows.extend(run_adaptation(budget, base_cfg))
    cfg = base_cfg or small_model_config()
    return {"notice": SYNTHETIC_NOTICE, "data": "synthetic", "budget": asdict(budget),
            "model_config": asdict(cfg), "rows": rows}


def format_table(report: dict[str, Any]) -> str:
    """Markdown table of an :func:`run_ablations` report, headed by the synthetic notice."""
    lines = [f"> {report['notice']}", "",
             "| group | variant | top1 | unwanted rate | ECE | NLL | prior top1 | train s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        prior = f"{r['prior_top1']:.3f}" if "prior_top1" in r else "-"
        lines.append(f"| {r['group']} | {r['variant']} | {r['top1']:.3f} | "
                     f"{r['unwanted_suggestion_rate']:.3f} | {r['ece']:.3f} | {r['nll']:.3f} | "
                     f"{prior} | {r['train_s']:.1f} |")
    return "\n".join(lines)
