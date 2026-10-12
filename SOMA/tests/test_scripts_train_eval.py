"""Offline train/eval scripts on synthetic data (synthetic != real-world quality)."""

import json

import pytest
import torch

from scripts import evaluate as evaluate_script
from scripts import train_ranker
from soma.models import load_checkpoint

ARGS = ["--small", "--n", "200", "--epochs", "2", "--seed", "0"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("train") / "ranker.pt"
    summary = train_ranker.main([*ARGS, "--out", str(out), "--json", str(out.with_suffix(".json"))])
    return out, summary


def test_train_script_writes_loadable_checkpoint_with_metadata(trained):
    out, summary = trained
    assert out.exists() and summary["checkpoint"] == str(out)
    model, cfg, extra = load_checkpoint(out)
    assert cfg.d_model == 64                      # --small config
    assert extra["data_kind"] == "synthetic" and "SYNTHETIC" in extra["notice"]
    assert extra["seed"] == 0 and extra["n_train"] + extra["n_val"] == 200
    assert extra["epochs_run"] == 2
    for split in ("train", "val"):
        m = extra["metrics"][split]
        assert 0.0 <= m["top1"] <= 1.0 and m["nll"] == m["nll"]
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["data_kind"] == "synthetic"
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_train_script_is_seeded(trained, tmp_path):
    out, _ = trained
    again = tmp_path / "again.pt"
    train_ranker.main([*ARGS, "--out", str(again)])
    a, _, ea = load_checkpoint(out)
    b, _, eb = load_checkpoint(again)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert ea["metrics"] == eb["metrics"]


def test_evaluate_script_reports_full_and_both_baselines(trained, tmp_path):
    out, _ = trained
    js = tmp_path / "eval.json"
    report = evaluate_script.main(["--checkpoint", str(out), "--n", "120", "--seed", "1",
                                   "--device", "cpu", "--json", str(js)])
    on_disk = json.loads(js.read_text(encoding="utf-8"))
    assert on_disk["data"] == "synthetic" and "SYNTHETIC" in on_disk["notice"]
    assert set(on_disk["results"]) == {"full", "no_memory", "rules_type_priority", "rules_keyword"}
    for name, m in on_disk["results"].items():
        for key in ("top1", "unwanted_suggestion_rate", "ece", "nll"):
            assert key in m, (name, key)
        assert m["n"] == 120
    assert on_disk["holdout"]["seed"] == 1 and not on_disk["holdout"]["same_seed_as_training"]
    assert "top1" in on_disk["full_vs_no_memory"]
    # The screen-blind rule never abstains while a candidate is open.
    assert on_disk["results"]["rules_type_priority"]["abstain_recall"] < 0.5
    # Reproducible: same checkpoint + seed -> identical numbers.
    again = evaluate_script.run(out, 120, 1, device="cpu")
    assert again["results"] == report["results"]


def test_evaluate_warns_when_holdout_seed_equals_training_seed(trained):
    out, _ = trained
    report = evaluate_script.run(out, 40, 0, device="cpu")
    assert report["holdout"]["same_seed_as_training"]
    assert "not a holdout" in evaluate_script.format_report(report)


def test_benchmark_script_profiles_every_stage(tmp_path):
    from scripts import benchmark
    from soma.runtime.profiling import StageTimer, percentile

    t = StageTimer()
    for ms in (1.0, 2.0, 3.0, 4.0):
        t.add("x", ms)
    s = t.summary()["x"]
    assert s["n"] == 4 and s["p50_ms"] == 2.5 and percentile([], 50) != percentile([], 50)

    out = tmp_path / "bench.json"
    report = benchmark.main(["--configs", "small", "--iters", "2", "--warmup", "1", "--memories", "50",
                             "--no-rapidocr", "--json", str(out)])
    stages = report["results"]["small"]["stages"]
    for name in ("render", "ocr_synthetic", "parse", "visual_encode", "text_encode", "screen_fusion",
                 "temporal", "retrieval_N50", "memory_fusion", "policy", "gate", "model_forward"):
        assert stages[name]["n"] == 2 and stages[name]["p95_ms"] >= stages[name]["p50_ms"] >= 0
    assert "ocr_rapidocr" not in stages
    hw = json.loads(out.read_text(encoding="utf-8"))["hardware"]
    assert {"cpu", "torch", "cuda_available"} <= set(hw)
