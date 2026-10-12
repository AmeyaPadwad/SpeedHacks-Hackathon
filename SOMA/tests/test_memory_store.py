"""MemoryStore: schema, CRUD, feedback, sources, suggestion log, thread safety."""

from __future__ import annotations

import threading

from soma.memory.sqlite_store import SUMMARY_CAP, MemoryStore, path_within
from soma.types import FeedbackEvent, FeedbackOutcome, MemoryItem, SourceType


def _item(i: str = "m1", **kw) -> MemoryItem:
    base = dict(id=i, source_type=SourceType.DOCUMENT, user_approved=True,
                text_summary="hello", source_uri="/tmp/x.txt", tags=["a", "b"],
                provenance={"path": "/tmp/x.txt", "sha256": "abc"})
    base.update(kw)
    return MemoryItem(**base)


def test_schema_matches_plan(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    cols = [c for c, _ in s.table_columns("memory_items")]
    assert cols == ["id", "source_uri", "source_type", "user_approved", "text_summary",
                    "embedding_ref", "created_at", "updated_at", "expires_at", "tags_json",
                    "provenance_json"]
    fb = [c for c, _ in s.table_columns("feedback_events")]
    assert fb == ["id", "suggestion_id", "outcome", "observed_at", "context_ref",
                  "consented_for_training"]
    assert {c for c, _ in s.table_columns("sources")} == {"root", "allowed", "trust", "blocked"}
    assert {c for c, _ in s.table_columns("suggestion_log")} == {
        "id", "action", "candidate_key", "screen_version", "shown_at", "confidence"}


def test_schema_has_no_raw_content_or_blob_columns(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    banned = ("content", "raw", "blob", "bytes", "image", "pixels", "screen_text", "ocr", "body")
    for table in ("memory_items", "feedback_events", "sources", "suggestion_log"):
        for name, decl in s.table_columns(table):
            assert decl.upper() != "BLOB", (table, name)
            assert not any(b in name.lower() for b in banned), (table, name)


def test_wal_mode(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    mode = s._query("PRAGMA journal_mode")[0][0]
    assert mode.lower() == "wal"


def test_crud_roundtrip_with_json(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    s.upsert_item(_item())
    got = s.get_item("m1")
    assert got is not None
    assert got.tags == ["a", "b"]
    assert got.provenance["sha256"] == "abc"
    assert got.source_type is SourceType.DOCUMENT
    assert got.created_at and got.updated_at
    assert got.embedding_ref == "m1"
    s.update_summary("m1", "changed")
    assert s.get_item("m1").text_summary == "changed"
    s.set_approved("m1", False)
    assert s.list_items(approved_only=True) == []
    assert s.delete_item("m1")
    assert s.get_item("m1") is None
    assert not s.delete_item("m1")


def test_summary_is_capped(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    s.upsert_item(_item(text_summary="x" * 5000))
    assert len(s.get_item("m1").text_summary) <= SUMMARY_CAP
    s.update_summary("m1", "y" * 5000)
    assert len(s.get_item("m1").text_summary) <= SUMMARY_CAP


def test_list_by_source_type(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    s.upsert_item(_item("d"))
    s.upsert_item(_item("p", source_type=SourceType.PREFERENCE, source_uri=None))
    assert [i.id for i in s.list_items(source_types=[SourceType.PREFERENCE])] == ["p"]
    assert s.list_items(source_types=[]) == []
    assert s.count_items() == 2


def test_feedback_roundtrip(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    s.add_feedback(FeedbackEvent("sug_1", FeedbackOutcome.ACCEPTED, consented_for_training=True))
    s.add_feedback(FeedbackEvent("sug_2", FeedbackOutcome.REJECTED, context_ref="m1"))
    assert len(s.list_feedback()) == 2
    only = s.list_feedback(consented_only=True)
    assert [f.suggestion_id for f in only] == ["sug_1"]
    assert only[0].outcome is FeedbackOutcome.ACCEPTED
    assert s.delete_feedback_referencing("m1") == 1
    assert len(s.list_feedback()) == 1


def test_sources_block_and_trust(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    root = tmp_path / "notes"
    root.mkdir()
    s.set_source(root, trust=0.9)
    assert s.source_trust(root / "a.md") == 0.9
    assert not s.is_source_blocked(root / "a.md")
    s.set_source_blocked(root)
    assert s.is_source_blocked(root / "sub" / "a.md")
    assert not s.is_source_blocked(tmp_path / "other.md")
    s.set_source_blocked(root, False)
    assert not s.is_source_blocked(root / "a.md")
    s.set_source(root, trust=7)
    assert s.get_source(root)["trust"] == 1.0


def test_suggestion_log(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")
    s.log_suggestion("EXPLAIN_ERROR", "EXPLAIN_ERROR|x||", 3, 0.8)
    rows = s.list_suggestions()
    assert rows[0]["screen_version"] == 3 and rows[0]["confidence"] == 0.8


def test_path_within(tmp_path):
    assert path_within(tmp_path / "a" / "b", tmp_path / "a")
    assert path_within(tmp_path / "a", tmp_path / "a")
    assert not path_within(tmp_path / "ab", tmp_path / "a")


def test_concurrent_writes(tmp_path):
    s = MemoryStore(tmp_path / "m.sqlite")

    def work(n: int) -> None:
        for j in range(20):
            s.upsert_item(_item(f"t{n}_{j}"))

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert s.count_items() == 80


# ---------------------------------------------------------------- export / viewer / correct


def test_export_viewer_and_correct(tmp_path):
    import json

    import numpy as np

    from soma.memory.embedder import TextEmbedder
    from soma.memory.export import correct, export_memories, list_for_viewer
    from soma.memory.vector_index import VectorIndex

    te = TextEmbedder(dim=32, vocab_size=512)
    s = MemoryStore(tmp_path / "m.sqlite")
    idx = VectorIndex(32, path=tmp_path / "i.npz")
    s.upsert_item(_item("m1", text_summary="wrong fact"))
    idx.add("m1", te.embed_text("wrong fact"))
    s.add_feedback(FeedbackEvent("sug_1", FeedbackOutcome.EDITED, context_ref="m1"))

    out = tmp_path / "export" / "memories.json"
    assert export_memories(s, out) == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["memories"][0]["text_summary"] == "wrong fact"
    assert "embedding" not in json.dumps(data["memories"][0]).replace("embedding_ref", "")
    assert data["feedback"][0]["outcome"] == "edited"

    rows = list_for_viewer(s)
    assert rows[0]["id"] == "m1" and rows[0]["summary"] == "wrong fact"

    fixed = correct(s, idx, te, "m1", "right fact")
    assert fixed.text_summary == "right fact" and "corrected_at" in fixed.provenance
    np.testing.assert_allclose(idx.get("m1"), te.embed_text("right fact"), atol=1e-6)
    assert "m1" in VectorIndex.load(tmp_path / "i.npz")
    import pytest
    with pytest.raises(KeyError):
        correct(s, idx, te, "missing", "x")


def test_in_memory_store_keeps_derived_files_out_of_cwd():
    from pathlib import Path

    s = MemoryStore(":memory:")
    s.upsert_item(_item())
    assert s.get_item("m1") is not None
    assert not str(s.derived_dir.resolve()).startswith(str(Path.cwd().resolve()))
    s.close()
