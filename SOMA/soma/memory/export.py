"""User controls: inspect, export and correct memories (plan section 6, item 6).

Exports never include embeddings or raw file contents: only the metadata,
short summaries and provenance already stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from soma.memory.embedder import TextEmbedder
from soma.memory.sqlite_store import MemoryStore, utc_now_iso
from soma.memory.vector_index import VectorIndex
from soma.types import MemoryItem, SourceType


def _item_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "source_type": SourceType(item.source_type).value,
        "source_uri": item.source_uri,
        "user_approved": item.user_approved,
        "text_summary": item.text_summary,
        "tags": list(item.tags),
        "provenance": dict(item.provenance),
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "expires_at": item.expires_at,
    }


def export_memories(store: MemoryStore, path: str | Path) -> int:
    """Write all memories, feedback outcomes and sources to a JSON file. Returns item count."""
    items = store.list_items()
    payload = {
        "exported_at": utc_now_iso(),
        "memories": [_item_dict(i) for i in items],
        "feedback": [{"id": f.id, "suggestion_id": f.suggestion_id, "outcome": f.outcome.value,
                      "observed_at": f.observed_at, "context_ref": f.context_ref,
                      "consented_for_training": f.consented_for_training}
                     for f in store.list_feedback()],
        "sources": store.list_sources(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return len(items)


def list_for_viewer(store: MemoryStore, include_unapproved: bool = True) -> list[dict[str, Any]]:
    """Compact rows for a memory-viewer UI, most recently updated first."""
    rows = []
    for item in store.list_items(approved_only=not include_unapproved):
        rows.append({
            "id": item.id,
            "type": SourceType(item.source_type).value,
            "summary": item.text_summary,
            "source": item.source_uri or item.provenance.get("origin", ""),
            "approved": item.user_approved,
            "tags": list(item.tags),
            "updated_at": item.updated_at,
            "expires_at": item.expires_at,
            "has_thumbnail": bool(item.provenance.get("thumbnail")),
        })
    rows.sort(key=lambda r: r["updated_at"] or "", reverse=True)
    return rows


def correct(store: MemoryStore, index: VectorIndex, text_embedder: TextEmbedder, item_id: str,
            new_summary: str, reembed_images: bool = False) -> MemoryItem:
    """Replace a memory's summary with the user's correction and re-embed it.

    Image memories keep their image embedding unless ``reembed_images``.
    Raises ``KeyError`` for an unknown id.
    """
    item = store.get_item(item_id)
    if item is None:
        raise KeyError(item_id)
    item.text_summary = new_summary
    item.provenance = {**item.provenance, "corrected_at": utc_now_iso(), "corrected_by": "user"}
    store.upsert_item(item)
    if SourceType(item.source_type) is not SourceType.IMAGE or reembed_images:
        index.add(item_id, text_embedder.embed_text(item.text_summary))
        index.persist()
    return store.get_item(item_id)  # type: ignore[return-value]
