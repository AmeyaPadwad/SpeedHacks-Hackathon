"""M5 one-variable-at-a-time ablations on SYNTHETIC data with the small model config.

    python -m scripts.ablations [--groups memory temporal ocr temporal_kind adaptation]
                                [--n-train 1500] [--n-eval 500] [--epochs 6] [--seed 0 | --seeds 0 1 2]
                                [--json runs/ablations.json]

Every variant trains from the same seed on the same budget and changes one
thing (see :mod:`soma.learning.ablations`). The printed table is headed by a
notice that the results are synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from soma.learning.ablations import GROUPS, AblationBudget, format_table, run_ablations


def build_parser() -> argparse.ArgumentParser:
    d = AblationBudget()
    p = argparse.ArgumentParser(prog="python -m scripts.ablations", description=__doc__.split("\n")[0])
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-eval", type=int, default=d.n_eval)
    p.add_argument("--n-personal", type=int, default=d.n_personal)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="several training seeds; report mean +/- std (overrides --seed)")
    p.add_argument("--eval-seed", type=int, default=d.eval_seed)
    p.add_argument("--json", default=None, help="write the report to this JSON file")
    return p


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    budget = AblationBudget(n_train=args.n_train, n_eval=args.n_eval, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                            eval_seed=args.eval_seed, n_personal=args.n_personal,
                            personal_epochs=args.epochs)
    report = run_ablations(args.groups, budget, seeds=args.seeds)
    print(format_table(report))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
