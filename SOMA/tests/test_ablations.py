"""Scripted baselines, ablation transforms and the tiny-budget ablation runner (synthetic)."""

import json
import math
from dataclasses import replace

import pytest
import torch

from scripts import ablations as ablations_script
from soma.config import small_model_config
from soma.learning import collate_soma, evaluate, generate_synthetic, make_loader
from soma.learning.ablations import (ABLATIONS, GROUPS, SYNTHETIC_NOTICE, AblationBudget,
                                     KeywordRulesBaseline, TypePriorityBaseline, format_table,
                                     mask_history, mask_memory, mask_ocr, retrieval_only,
                                     run_ablations)
from soma.learning.synthetic_data import TYPE_INDEX
from soma.models import build_model
from soma.types import ActionType

TINY = AblationBudget(n_train=48, n_eval=32, epochs=1, batch_size=16, n_personal=24,
                      personal_epochs=1)


@pytest.fixture(scope="module")
def batch():
    cfg = small_model_config()
    return cfg, collate_soma(generate_synthetic(64, cfg, seed=3))


def test_masking_transforms(batch):
    cfg, (b, targets, _) = batch
    assert mask_memory(b).memory_padding_mask.all() and not mask_memory(b).memory_embeddings.any()
    assert mask_history(b).history_padding_mask.all()
    assert mask_ocr(b).ocr_padding_mask.all() and not mask_ocr(b).ocr_ids.any()
    model = build_model(cfg).eval()
    with torch.no_grad():
        for fn in (mask_memory, mask_history, mask_ocr, retrieval_only):
            logits = model(fn(b, targets))["action_logits"]
            assert torch.isfinite(logits).all()


def test_retrieval_only_gates_compare_candidates_on_retrieval(batch):
    _, (b, targets, _) = batch
    out = retrieval_only(b, targets)
    assert out.memory_padding_mask.all()                       # fusion never sees memory
    compare = b.candidate_types == TYPE_INDEX[ActionType.COMPARE_WITH_PRIOR_WORK]
    no_mem = b.memory_padding_mask.all(dim=1, keepdim=True)
    was_open = ~b.candidate_mask
    assert (out.candidate_mask[compare & no_mem & was_open]).all()
    # Other candidates and every labeled slot stay open.
    assert torch.equal(out.candidate_mask[~compare], b.candidate_mask[~compare])
    for i, t in enumerate(targets.tolist()):
        if t > 0:
            assert not out.candidate_mask[i, t - 1]


def test_type_priority_baseline_always_speaks_when_possible(batch):
    _, (b, targets, _) = batch
    logits = TypePriorityBaseline()(b)["action_logits"]
    pred = logits.argmax(-1)
    has_open = (~b.candidate_mask).any(dim=1)
    assert torch.equal(pred != 0, has_open)
    order = [TYPE_INDEX[a] for a in (ActionType.EXPLAIN_ERROR, ActionType.OPEN_RELEVANT_FILE)]
    for i in range(b.batch_size):
        open_types = b.candidate_types[i][~b.candidate_mask[i]].tolist()
        if order[0] in open_types:
            assert int(b.candidate_types[i, pred[i] - 1]) == order[0]


def test_keyword_rules_mirror_the_generator_without_hash_collisions():
    cfg = replace(small_model_config(), vocab_size=8192)
    items = generate_synthetic(200, cfg, seed=5)
    m = evaluate(KeywordRulesBaseline(cfg), make_loader(items, 64), device="cpu")
    assert m["top1"] > 0.95                                     # rule-generated labels
    blind = evaluate(TypePriorityBaseline(), make_loader(items, 64), device="cpu")
    assert blind["unwanted_suggestion_rate"] > 0.9 and blind["top1"] < m["top1"]


def test_tiny_ablation_run_covers_every_group_and_is_deterministic():
    report = run_ablations(GROUPS, TINY)
    assert report["notice"] == SYNTHETIC_NOTICE and report["data"] == "synthetic"
    variants = {(r["group"], r["variant"]) for r in report["rows"]}
    assert {(s.group, s.variant) for s in ABLATIONS} <= variants
    assert {"base_no_adaptation", "frozen_base_online_ranker",
            "finetuned_policy_head"} <= {v for g, v in variants if g == "adaptation"}
    for r in report["rows"]:
        for key in ("top1", "unwanted_suggestion_rate", "ece"):
            assert math.isfinite(r[key]) and 0.0 <= r[key] <= 1.0 + 1e-6
    # The frozen-base ranker never changes generic (other-app) behaviour.
    rows = {r["variant"]: r for r in report["rows"] if r["group"] == "adaptation"}
    assert rows["frozen_base_online_ranker"]["prior_top1"] == rows["base_no_adaptation"]["prior_top1"]
    table = format_table(report)
    assert table.startswith("> SYNTHETIC DATA ONLY")
    again = run_ablations(["memory"], TINY)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "train_s"} for r in rows]  # noqa: E731
    first = [r for r in report["rows"] if r["group"] == "memory"]
    assert strip(again["rows"]) == strip(first)


def test_multi_seed_aggregation():
    report = run_ablations(["temporal_kind"], TINY, seeds=[0, 1])
    assert report["seeds"] == [0, 1] and len(report["per_seed"]) == 2
    row = report["rows"][0]
    assert row["n_seeds"] == 2 and "top1_std" in row
    assert "+/-" in format_table(report)


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        run_ablations(["multiscale"], TINY)


def test_ablations_script_writes_json(tmp_path):
    out = tmp_path / "abl.json"
    ablations_script.main(["--groups", "ocr", "--n-train", "32", "--n-eval", "16", "--epochs", "1",
                           "--json", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["data"] == "synthetic"
    assert [r["variant"] for r in data["rows"]] == ["visual_only", "ocr_visual"]
