"""Build and manage SOMA's local memory from one user-selected folder.

Usage::

    python -m scripts.build_memory --folder <dir> --grant [--data-dir data]
    python -m scripts.build_memory --list
    python -m scripts.build_memory --forget <memory-id>
    python -m scripts.build_memory --export <path.json>

Indexing requires the explicit ``--grant`` flag, which records FOLDER_INDEXING
consent and allows exactly that folder (home directories and drive roots are
refused). Without ``--grant`` a folder is only indexed if it was allowed
before. Files stay where they are; only redacted summaries, provenance and
embeddings are stored under ``--data-dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from soma.config import SomaConfig, load_config
from soma.consent import ConsentError, ConsentManager, Scope
from soma.memory.deletion import forget, verify_deleted
from soma.memory.export import export_memories, list_for_viewer
from soma.runtime.factory import open_memory

CONSENT_FILE = "consent.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.build_memory",
                                description="Index a user-selected folder into SOMA's local memory.")
    p.add_argument("--folder", type=Path, help="folder to index (requires --grant the first time)")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="where memory is stored")
    p.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    p.add_argument("--grant", action="store_true",
                   help="explicitly grant folder-indexing consent for --folder")
    p.add_argument("--list", action="store_true", help="list stored memories")
    p.add_argument("--forget", metavar="ID", help="delete one memory everywhere and verify")
    p.add_argument("--export", metavar="PATH", type=Path, help="export memories as JSON")
    return p


def main(argv: list[str] | None = None, out=None) -> int:
    out = out or sys.stdout
    args = _parser().parse_args(argv)
    if not (args.folder or args.list or args.forget or args.export):
        _parser().print_usage(out)
        return 2
    cfg: SomaConfig = load_config(args.config) if args.config else SomaConfig()
    data_dir = args.data_dir
    consent = ConsentManager(data_dir / CONSENT_FILE)
    mem = open_memory(cfg, consent, data_dir)
    try:
        if args.folder is not None:
            folder = args.folder.expanduser().resolve()
            if args.grant:
                try:
                    consent.add_folder(folder)
                except ConsentError as exc:
                    print(f"refused: {exc}", file=out)
                    return 2
                consent.grant(Scope.FOLDER_INDEXING)
                print(f"granted folder indexing for {folder}", file=out)
            if not consent.path_allowed(folder):
                print(f"refused: {folder} is not an allowed folder; re-run with --grant to allow it",
                      file=out)
                return 2
            try:
                report = mem.ingestor.ingest_folder(folder)
            except (ConsentError, NotADirectoryError) as exc:
                print(f"refused: {exc}", file=out)
                return 2
            print(f"indexed {folder}: added={len(report.added)} updated={len(report.updated)} "
                  f"unchanged={len(report.unchanged)} skipped={len(report.skipped)} "
                  f"errors={len(report.errors)}", file=out)
            for err in report.errors[:20]:
                print(f"  error: {err}", file=out)
        if args.list:
            rows = list_for_viewer(mem.store)
            print(f"{len(rows)} memories", file=out)
            for r in rows:
                summary = (r["summary"] or "").replace("\n", " ")[:80]
                print(f"  {r['id']}  [{r['type']}]  {summary}  ({r['source']})", file=out)
        if args.forget:
            removed = forget(mem.store, mem.index, args.forget)
            ok = verify_deleted(mem.store, mem.index, args.forget)
            print(f"forget {args.forget}: removed={removed} verified={ok}", file=out)
            if not ok:
                return 1
        if args.export:
            n = export_memories(mem.store, args.export)
            print(f"exported {n} memories to {args.export}", file=out)
        return 0
    finally:
        mem.close()


if __name__ == "__main__":
    raise SystemExit(main())
