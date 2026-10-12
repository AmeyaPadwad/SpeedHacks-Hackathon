"""Forget and verify deletion of memories across store, vector index and derived files.

Deletion removes, for each item: the ``memory_items`` row, its vector entry
(then persists the index so the on-disk ``.npz`` no longer holds it),
derived files such as thumbnails (only files inside ``store.derived_dir`` are
ever deleted; original user files are never touched), and feedback events
whose ``context_ref`` points at the item.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from soma.memory.sqlite_store import MemoryStore, parse_iso, path_within
from soma.memory.vector_index import VectorIndex
from soma.types import MemoryItem


def _derived_files(store: MemoryStore, item_id: str, item: MemoryItem | None = None) -> list[Path]:
    """Derived files for ``item_id`` that live inside the store's derived directory."""
    found: list[Path] = []
    d = store.derived_dir
    if item is not None:
        for key in ("thumbnail",):
            p = item.provenance.get(key)
            if p and path_within(p, d):
                found.append(Path(p))
        for p in item.provenance.get("derived_files", []) or []:
            if path_within(p, d):
                found.append(Path(p))
    if d.is_dir():
        found.extend(p for p in d.glob(f"{item_id}.*") if p.is_file())
        found.extend(p for p in d.glob(f"{item_id}_*") if p.is_file())
    uniq: dict[str, Path] = {}
    for p in found:
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def _forget_one(store: MemoryStore, index: VectorIndex, item_id: str) -> bool:
    item = store.get_item(item_id)
    for p in _derived_files(store, item_id, item):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    removed_row = store.delete_item(item_id)
    removed_vec = index.remove(item_id)
    store.delete_feedback_referencing(item_id)
    return removed_row or removed_vec


def forget(store: MemoryStore, index: VectorIndex, item_id: str) -> bool:
    """Delete one memory everywhere and persist the index. True if anything was removed."""
    removed = _forget_one(store, index, item_id)
    index.persist()
    return removed


def forget_source(store: MemoryStore, index: VectorIndex, folder: str | Path) -> int:
    """Delete every memory whose ``source_uri`` lies inside ``folder``. Returns the count."""
    ids = [i.id for i in store.items_under(folder)]
    for item_id in ids:
        _forget_one(store, index, item_id)
    index.persist()
    return len(ids)


def purge_expired(store: MemoryStore, index: VectorIndex, now: datetime | None = None) -> int:
    """Delete memories whose ``expires_at`` has passed. Returns the count."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ids = [i.id for i in store.list_items()
           if (exp := parse_iso(i.expires_at)) is not None and exp <= now]
    for item_id in ids:
        _forget_one(store, index, item_id)
    if ids:
        index.persist()
    return len(ids)


def verify_deleted(store: MemoryStore, index: VectorIndex, item_id: str) -> bool:
    """True when ``item_id`` is absent from the store, the live index, the persisted
    index file (if any), derived files, and feedback references."""
    if store.get_item(item_id) is not None:
        return False
    if item_id in index:
        return False
    if index.path is not None and Path(index.path).exists():
        if item_id in VectorIndex.load(index.path):
            return False
    if _derived_files(store, item_id):
        return False
    if any(f.context_ref == item_id for f in store.list_feedback()):
        return False
    return True
