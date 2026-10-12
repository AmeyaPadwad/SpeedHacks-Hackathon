"""Train a runtime-compatible demo checkpoint on synthetic screens.

``scripts.train_ranker`` trains on abstract synthetic tensors. This script
instead runs the *real* runtime pipeline (synthetic screen -> synthetic OCR ->
parser -> retrieval -> candidates) and labels each scenario with its expected
action, so the desktop app with ``--synthetic --checkpoint`` makes visible
suggestions and stays silent on low-value screens.

Synthetic data only: the result says nothing about real-world quality.

    python -m scripts.train_demo --data-dir data/demo --grant-demo-consent
    python -m scripts.run_desktop --synthetic --data-dir data/demo --checkpoint checkpoints/demo_runtime.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soma.config import SomaConfig, TrainConfig, small_model_config  # noqa: E402
from soma.consent import ConsentManager, Scope  # noqa: E402
from soma.learning.datasets import collate_soma  # noqa: E402
from soma.learning.trainer import Trainer  # noqa: E402
from soma.perception.ocr import SyntheticOCREngine  # noqa: E402
from soma.perception.synthetic import SCENARIOS, expected_action, render_scenario  # noqa: E402
from soma.runtime.factory import open_memory, save_runtime_checkpoint  # noqa: E402
from soma.runtime.pipeline import SomaRuntime  # noqa: E402
from soma.types import ActionType, ScreenFrame  # noqa: E402

DEMO_NOTE = "My notes on HTTP client sessions, retries and timeouts from the last project."
DEMO_PREFERENCE = "When a traceback appears, inspect the last frame first"


def target_index(candidates, name: str) -> int:
    """Slot of the scenario's expected action (0 = NO_SUGGESTION)."""
    want = expected_action(name)
    if want is ActionType.NO_SUGGESTION:
        return 0
    for i, c in enumerate(candidates):
        if c.action is want:
            return i + 1
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data/demo")
    p.add_argument("--out", default="checkpoints/demo_runtime.pt")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--grant-demo-consent", action="store_true",
                   help="explicitly grant capture, indexing of the demo notes folder, and feedback "
                        "storage in <data-dir>/consent.json (required)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.grant_demo_consent:
        print("Refusing to run without --grant-demo-consent (SOMA never grants consent implicitly).")
        return 2
    data_dir = (ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    data_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    cfg = SomaConfig(device="cpu", model=small_model_config())
    consent = ConsentManager(data_dir / "consent.json")
    for s in (Scope.SCREEN_CAPTURE, Scope.FOLDER_INDEXING, Scope.FEEDBACK_STORAGE):
        consent.grant(s)
    notes = data_dir / "demo_notes"
    notes.mkdir(exist_ok=True)
    (notes / "http_notes.md").write_text(DEMO_NOTE, encoding="utf-8")
    consent.add_folder(notes)

    mem = open_memory(cfg, consent, data_dir)
    mem.ingestor.ingest_folder(notes)
    if not any(DEMO_PREFERENCE in (i.text_summary or "") for i in mem.store.list_items()):
        mem.ingestor.add_preference(DEMO_PREFERENCE)

    rt = SomaRuntime(cfg, consent, ocr=SyntheticOCREngine(), memory=mem, model_seed=0)
    items, version, rng = [], 0, random.Random(0)
    for seed in range(args.seeds):
        order = list(SCENARIOS)
        rng.shuffle(order)
        for name in order:
            version += 1
            image, obs, _ = render_scenario(name, seed=seed)
            p = rt.prepare(ScreenFrame(image=image, window=obs.window, version=version))
            items.append(p.to_item(target_index(p.candidates, name), task=name, keep_images=True))
            rt.history.push(p.screen_state[0])
    rt.history.clear()

    trainer = Trainer(rt.model, TrainConfig(lr=3e-3, weight_decay=0.0, use_amp=False, grad_clip=1.0),
                      device="cpu")
    g = random.Random(1)
    for step in range(args.steps):
        batch, targets, weights = collate_soma(g.sample(items, min(16, len(items))))
        parts = trainer.train_step(batch, targets, weights)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {parts.get('total', float('nan')):.4f}")
    rt.model.eval()
    save_runtime_checkpoint(rt, out, extra={"data_kind": "synthetic-runtime-demo",
                                            "seeds": args.seeds, "steps": args.steps})
    mem.close()
    print(f"Saved {out}  (synthetic demo only; not real-world quality)")
    print(f"Run: python -m scripts.run_desktop --synthetic --data-dir {args.data_dir} --checkpoint {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
