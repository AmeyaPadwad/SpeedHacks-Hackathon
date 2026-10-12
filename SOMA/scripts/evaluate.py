"""Reproducible offline evaluation of a SOMA checkpoint on a fixed SYNTHETIC holdout.

    python -m scripts.evaluate --checkpoint checkpoints/ranker.pt [--n 500] [--seed 1]
                               [--json out.json]

Compares (build plan section 12):

* ``full``             the checkpoint as trained (screen + temporal + gated memory).
* ``no_memory``        baseline (1): same weights, every memory slot masked
                       (screen-only policy, :func:`soma.learning.evaluate_no_memory`).
* ``rules_type_priority`` baseline (2): scripted, screen-blind candidate-type
                       heuristic (always speaks with the highest-priority type).
* ``rules_keyword``    scripted OCR-keyword rules that mirror the synthetic
                       generator (reference upper bound on rule-generated data).

The holdout is regenerated deterministically from ``--seed`` and the
checkpoint's model config. Use a seed different from training (training uses
``train.seed``, default 0). All numbers are synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from soma.device import select_device
from soma.learning import evaluate, evaluate_no_memory, generate_synthetic, make_loader
from soma.learning.ablations import SYNTHETIC_NOTICE, KeywordRulesBaseline, TypePriorityBaseline
from soma.learning.evaluator import compare
from soma.learning.synthetic_data import label_distribution
from soma.models import load_checkpoint

REPORT_ORDER = ("full", "no_memory", "rules_type_priority", "rules_keyword")
TABLE_KEYS = ("top1", "topk", "abstain_precision", "abstain_recall", "unwanted_suggestion_rate",
              "missed_suggestion_rate", "ece", "nll")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.evaluate", description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="checkpoint written by scripts.train_ranker")
    p.add_argument("--n", type=int, default=500, help="holdout size")
    p.add_argument("--seed", type=int, default=1, help="holdout seed (differs from training seed)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", default=None, help="write the full report to this JSON file")
    return p


def run(checkpoint: str | Path, n: int = 500, seed: int = 1, *, batch_size: int = 128,
        device: str = "auto") -> dict[str, Any]:
    """Evaluate ``checkpoint`` and both baselines; returns a JSON-serializable report."""
    dev = select_device(device)
    model, cfg, extra = load_checkpoint(checkpoint, map_location=dev)
    items = generate_synthetic(int(n), cfg, seed=int(seed))
    loader = make_loader(items, batch_size)
    results = {
        "full": evaluate(model, loader, device=dev),
        "no_memory": evaluate_no_memory(model, loader, device=dev),
        "rules_type_priority": evaluate(TypePriorityBaseline().to(dev), loader, device=dev),
        "rules_keyword": evaluate(KeywordRulesBaseline(cfg).to(dev), loader, device=dev),
    }
    results = {name: {k: float(v) for k, v in m.items()} for name, m in results.items()}
    train_seed = extra.get("seed")
    return {
        "notice": SYNTHETIC_NOTICE,
        "data": "synthetic",
        "checkpoint": str(checkpoint),
        "checkpoint_data_kind": extra.get("data_kind", "unknown"),
        "train_seed": train_seed,
        "holdout": {"n": int(n), "seed": int(seed), **label_distribution(items),
                    "same_seed_as_training": train_seed is not None and int(train_seed) == int(seed)},
        "device": str(dev),
        "results": results,
        "full_vs_no_memory": compare(results["no_memory"], results["full"]),
        "full_vs_rules_type_priority": compare(results["rules_type_priority"], results["full"]),
    }


def format_report(report: dict[str, Any]) -> str:
    res = report["results"]
    cols = [c for c in REPORT_ORDER if c in res]
    width = max(len(k) for k in TABLE_KEYS) + 2
    head = "metric".ljust(width) + "".join(c.rjust(22) for c in cols)
    lines = [report["notice"],
             f"checkpoint: {report['checkpoint']}  holdout n={report['holdout']['n']} "
             f"seed={report['holdout']['seed']} silence={report['holdout']['silence']:.3f}",
             head, "-" * len(head)]
    for k in TABLE_KEYS:
        lines.append(k.ljust(width) + "".join(f"{res[c][k]:22.4f}" for c in cols))
    if report["holdout"]["same_seed_as_training"]:
        lines.append("WARNING: holdout seed equals the training seed; this is not a holdout.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = run(args.checkpoint, args.n, args.seed, batch_size=args.batch_size, device=args.device)
    print(format_report(report))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
