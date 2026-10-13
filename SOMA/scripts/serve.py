"""Run SOMA headless for a companion app (Pette): runtime + scheduler + localhost API.

Usage (from ``SOMA/``)::

    python -m scripts.serve [--config configs/pette.yaml] [--data-dir data]
                            [--checkpoint path] [--interval 2.0]
                            [--endpoint-file path] [--grant-screen] [--once]

* The API binds to 127.0.0.1 on a free port. ``--endpoint-file`` receives
  ``{"url": ..., "token": ...}`` with owner-only permissions (removed on exit);
  Pette reads it via ``PETTE_SOMA_ENDPOINT``.
* ``--grant-screen`` records consent for screen capture and app monitoring in
  the data dir's consent file (the same opt-in the desktop consent dialog
  writes). Without it, SOMA only runs if consent was granted before.
* ``--once`` runs one forced step, prints why SOMA spoke or stayed silent, and
  exits (a diagnostic; nothing is served).

Nothing is executed: the API only reports suggestions and takes feedback.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soma.api.local_api import LocalAPIServer, write_token  # noqa: E402
from soma.config import load_config  # noqa: E402
from soma.consent import ConsentManager, Scope  # noqa: E402
from soma.runtime.factory import build_runtime  # noqa: E402
from soma.runtime.scheduler import InferenceScheduler  # noqa: E402

CONSENT_FILE = "consent.json"
SCREEN_SCOPES = (Scope.SCREEN_CAPTURE, Scope.APP_MONITORING)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m scripts.serve", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/pette.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--interval", type=float, default=2.0, help="seconds between steps (adaptive back-off)")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--endpoint-file", default=None)
    p.add_argument("--grant-screen", action="store_true",
                   help="opt in to screen capture + app monitoring (stored in the consent file)")
    p.add_argument("--synthetic", action="store_true", help="synthetic screens instead of the real one")
    p.add_argument("--once", action="store_true", help="one forced step, print the decision, exit")
    return p.parse_args(argv)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def open_consent(data_dir: Path, grant_screen: bool) -> ConsentManager:
    consent = ConsentManager(data_dir / CONSENT_FILE)
    if grant_screen:
        for scope in SCREEN_SCOPES:
            consent.grant(scope)
    if consent.paused:
        consent.resume()
    return consent


def describe(result) -> str:
    """One readable block: app, candidates, decision and distribution (no screen text)."""
    s = result.suggestion
    lines = [f"app: {result.window.app_name or '?'}  skipped: {result.skipped_reason or 'no'}",
             f"candidates: {[c.action.value + (':' + c.template if c.template else '') for c in result.candidates]}",
             f"decision: {'SPEAK ' + s.candidate.text if s.candidate else 'silent'}  ({s.reason})",
             "distribution: " + ", ".join(f"{k} {v:.2f}" for k, v in s.distribution.items())]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, stop: threading.Event | None = None) -> int:
    """Serve until SIGINT/SIGTERM (or until ``stop`` is set, for tests)."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    consent = open_consent(data_dir, args.grant_screen)
    if not consent.allowed(Scope.SCREEN_CAPTURE):
        print("SOMA: screen capture is not granted; pass --grant-screen to opt in.", file=sys.stderr)
    runtime = build_runtime(cfg, consent, data_dir=data_dir,
                            checkpoint=resolve(args.checkpoint) if args.checkpoint else None,
                            synthetic=args.synthetic)

    if args.once:
        print(describe(runtime.step(force=True)), flush=True)
        runtime.memory.close()
        return 0

    scheduler = InferenceScheduler(runtime, args.interval)
    api = LocalAPIServer(runtime, scheduler=scheduler, port=args.port)
    if stop is None:
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
            if sig is not None:
                signal.signal(sig, lambda *_: stop.set())

    endpoint = resolve(args.endpoint_file) if args.endpoint_file else None
    api.start()
    scheduler.start()
    try:
        if endpoint is not None:
            write_token(endpoint, json.dumps({"url": api.url, "token": api.token}))
        print(f"SOMA serving on {api.url} (config {args.config}, "
              f"{'synthetic' if args.synthetic else 'real'} screen)", flush=True)
        while not stop.wait(0.5):
            pass
    finally:
        scheduler.stop()
        api.stop()
        if endpoint is not None:
            endpoint.unlink(missing_ok=True)
        runtime.memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
