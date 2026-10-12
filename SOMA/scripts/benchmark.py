"""Measure per-stage pipeline latency (P50/P95) on this machine.

    python -m scripts.benchmark [--configs dev small] [--iters 30] [--warmup 3]
                                [--memories 1000 10000] [--no-rapidocr] [--rapidocr-iters 5]
                                [--torch-profiler] [--json runs/benchmark.json]

``dev`` is ``configs/dev.yaml``; ``small`` is ``small_model_config()``. The
numbers describe this hardware only and are not latency promises.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from soma.config import ModelConfig, load_config, small_model_config
from soma.runtime.profiling import (format_stages, hardware_info, profile_pipeline,
                                    torch_profile_forward)

CONFIGS = ("dev", "small", "train")


def model_config(name: str) -> ModelConfig:
    if name == "small":
        return small_model_config()
    if name in ("dev", "train"):
        return load_config(f"configs/{name}.yaml").model
    return load_config(name).model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.benchmark", description=__doc__.split("\n")[0])
    p.add_argument("--configs", nargs="+", default=["dev", "small"],
                   help="dev | small | train | path/to/config.yaml")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--memories", type=int, nargs="+", default=[1_000, 10_000])
    p.add_argument("--no-rapidocr", action="store_true", help="skip RapidOCR even if installed")
    p.add_argument("--rapidocr-iters", type=int, default=5)
    p.add_argument("--device", default="cpu", help="cpu | auto | cuda")
    p.add_argument("--torch-profiler", action="store_true", help="also print a torch.profiler table")
    p.add_argument("--json", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {"date": date.today().isoformat(), "hardware": hardware_info(),
                              "results": {}}
    hw = report["hardware"]
    print(f"{hw['cpu']} | {hw['logical_cpus']} logical CPUs | torch {hw['torch']} "
          f"({hw['torch_threads']} threads) | CUDA available: {hw['cuda_available']} | {hw['os']}")
    for name in args.configs:
        cfg = model_config(name)
        res = profile_pipeline(cfg, iters=args.iters, warmup=args.warmup, n_memories=args.memories,
                               rapidocr=False if args.no_rapidocr else None,
                               rapidocr_iters=args.rapidocr_iters, device=args.device)
        report["results"][name] = res
        print(f"\n## {name}: d_model={cfg.d_model} image={cfg.image_size}/{cfg.patch_size} "
              f"L={cfg.max_ocr_tokens} T={cfg.history_len} K={cfg.max_memories} "
              f"A={cfg.max_candidates} temporal={cfg.temporal_kind} params={res['n_params']:,} "
              f"device={res['device']}")
        print(format_stages(res))
        if args.torch_profiler:
            table = torch_profile_forward(cfg, device=args.device)
            report["results"][name]["torch_profiler"] = table
            print(table)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
