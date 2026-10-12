"""Offline Stage B training of the SOMA ranker on SYNTHETIC data.

    python -m scripts.train_ranker [--config configs/train.yaml] [--n 2000] [--epochs N]
                                   [--out checkpoints/ranker.pt] [--seed 0] [--small]

Data comes from :func:`soma.learning.generate_synthetic` (non-sensitive,
rule-generated GUI scenarios). The resulting numbers are synthetic and are
**not** evidence of real-world quality. The checkpoint stores the state dict,
model config, and metadata (seed, data kind, split sizes, metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from soma.config import load_config, small_model_config
from soma.device import seed_everything, select_device
from soma.learning import (SomaDataset, Trainer, evaluate, generate_synthetic, make_loader,
                           seeded_split)
from soma.learning.ablations import SYNTHETIC_NOTICE
from soma.learning.synthetic_data import label_distribution
from soma.models import build_model, save_checkpoint

DEFAULT_CONFIG = "configs/train.yaml"
DEFAULT_OUT = "checkpoints/ranker.pt"
TABLE_KEYS = ("top1", "topk", "abstain_precision", "abstain_recall", "unwanted_suggestion_rate",
              "missed_suggestion_rate", "ece", "nll")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.train_ranker", description=__doc__.split("\n")[0])
    p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config (model + train sections)")
    p.add_argument("--n", type=int, default=2000, help="number of synthetic items (train + val)")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--out", default=DEFAULT_OUT, help="checkpoint path (.pt, gitignored)")
    p.add_argument("--seed", type=int, default=None, help="override train.seed")
    p.add_argument("--small", action="store_true", help="tiny model config for a fast CPU run")
    p.add_argument("--device", default=None, help="auto | cpu | cuda (default: config)")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")
    p.add_argument("--json", default=None, help="also write the metrics summary to this JSON file")
    return p


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def metrics_table(rows: dict[str, dict[str, float]]) -> str:
    """Plain-text table: one column per split/model, one row per metric."""
    cols = list(rows)
    width = max(len(k) for k in TABLE_KEYS) + 2
    head = "metric".ljust(width) + "".join(c.rjust(12) for c in cols)
    lines = [head, "-" * len(head)]
    for k in TABLE_KEYS:
        lines.append(k.ljust(width) + "".join(f"{rows[c].get(k, float('nan')):12.4f}" for c in cols))
    return "\n".join(lines)


def train(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    tcfg = cfg.train
    overrides: dict[str, Any] = {"use_amp": tcfg.use_amp}
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.lr is not None:
        overrides["lr"] = float(args.lr)
    tcfg = replace(tcfg, **overrides)
    model_cfg = small_model_config() if args.small else cfg.model
    device = select_device(args.device or cfg.device)

    seed_everything(tcfg.seed)
    items = generate_synthetic(int(args.n), model_cfg, seed=tcfg.seed)
    dataset = SomaDataset(items)
    train_set, val_set = seeded_split(dataset, tcfg.val_fraction, seed=tcfg.seed)
    train_loader = make_loader(train_set, tcfg.batch_size, shuffle=True, seed=tcfg.seed)
    val_loader = make_loader(val_set, max(64, tcfg.batch_size)) if len(val_set) else None

    seed_everything(tcfg.seed)
    model = build_model(model_cfg)
    out = _resolve(args.out)
    run_dir = out.parent / "runs" / f"{out.stem}_seed{tcfg.seed}"
    trainer = Trainer(model, tcfg, device, run_dir=run_dir, model_cfg=model_cfg,
                      patience=max(3, tcfg.epochs))
    result = trainer.fit(train_loader, val_loader)

    metrics = {"train": evaluate(model, make_loader(train_set, 128), device=device)}
    if val_loader is not None:
        metrics["val"] = evaluate(model, val_loader, device=device)
    extra = {
        "data_kind": "synthetic",
        "notice": SYNTHETIC_NOTICE,
        "seed": int(tcfg.seed),
        "n_items": int(args.n),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "label_distribution": label_distribution(items),
        "small": bool(args.small),
        "train_config": asdict(tcfg),
        "device": str(device),
        "epochs_run": int(result.epochs_run),
        "best_epoch": int(result.best_epoch),
        "stopped_reason": result.stopped_reason,
        "metrics": {split: {k: float(v) for k, v in m.items()} for split, m in metrics.items()},
    }
    save_checkpoint(model, out, model_cfg, extra=extra)
    summary = {"checkpoint": str(out), "run_dir": str(run_dir), "elapsed_s": result.elapsed_s,
               **{k: v for k, v in extra.items() if k != "train_config"}}
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    summary = train(args)
    print(SYNTHETIC_NOTICE)
    print(f"checkpoint: {summary['checkpoint']}  (seed={summary['seed']}, n_train={summary['n_train']}, "
          f"n_val={summary['n_val']}, epochs_run={summary['epochs_run']}, "
          f"{summary['elapsed_s']:.1f}s on {summary['device']})")
    print(metrics_table(summary["metrics"]))
    if args.json:
        p = _resolve(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
