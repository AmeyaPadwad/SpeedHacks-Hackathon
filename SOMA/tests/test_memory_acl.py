"""Access control: permission filtering happens before scoring and never leaks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from soma.config import MemoryConfig
from soma.consent import ConsentError, ConsentManager, Scope
from soma.memory.embedder import TextEmbedder
from soma.memory.ingestion import Ingestor
from soma.memory.retrieval import Retriever
from soma.memory.sqlite_store import MemoryStore
from soma.memory.vector_index import VectorIndex
from soma.types import MemoryItem, SourceType

DIM = 64


@pytest.fixture(scope="module")
def te():
    return TextEmbedder(dim=DIM, vocab_size=4096)


@pytest.fixture
def env(tmp_path, te):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "pytest.md").write_text("pytest traceback assertion error notes", encoding="utf-8")
    (b / "pytest2.md").write_text("pytest traceback in folder b", encoding="utf-8")
    consent = ConsentManager()
    consent.grant(Scope.FOLDER_INDEXING)
    consent.add_folder(a)
    consent.add_folder(b)
    store = MemoryStore(tmp_path / "m.sqlite")
    index = VectorIndex(DIM)
    ing = Ingestor(store, index, consent, te, None, MemoryConfig())
    ing.ingest_folder(a)
    ing.ingest_folder(b)
    ret = Retriever(store, index, consent, MemoryConfig())
    return dict(a=a, b=b, consent=consent, store=store, index=index, ing=ing, ret=ret,
                q=te.embed_text("pytest traceback"))


def _uris(results) -> set[str]:
    return {Path(r.item.source_uri).parent.name for r in results if r.item.source_uri}


def test_baseline_returns_both(env):
    assert _uris(env["ret"].retrieve(env["q"], k=5)) == {"a", "b"}


def test_blocked_source_never_returned(env):
    env["store"].set_source_blocked(env["b"])
    res = env["ret"].retrieve(env["q"], k=10)
    assert _uris(res) == {"a"}
    assert env["ret"].permission_violations == 0


def test_blocked_items_are_not_scored(env, monkeypatch):
    env["store"].set_source_blocked(env["b"])
    seen: list[set[str] | None] = []
    orig = env["index"].search

    def spy(query, k, allowed_ids=None):
        seen.append(set(allowed_ids) if allowed_ids is not None else None)
        return orig(query, k, allowed_ids)

    monkeypatch.setattr(env["index"], "search", spy)
    env["ret"].retrieve(env["q"], k=10)
    assert seen and seen[0] is not None
    blocked_ids = {i.id for i in env["store"].items_under(env["b"])}
    assert not (seen[0] & blocked_ids)


def test_items_outside_allowed_folders_excluded(env):
    env["consent"].remove_folder(env["b"])
    assert _uris(env["ret"].retrieve(env["q"], k=10)) == {"a"}


def test_revoking_folder_indexing_excludes_documents(env):
    env["ing"].add_preference("I like pytest traceback summaries")
    env["consent"].revoke(Scope.FOLDER_INDEXING)
    res = env["ret"].retrieve(env["q"], k=10)
    assert res and all(r.item.source_type is SourceType.PREFERENCE for r in res)


def test_unapproved_excluded(env):
    for item in env["store"].items_under(env["a"]):
        env["store"].set_approved(item.id, False)
    assert _uris(env["ret"].retrieve(env["q"], k=10)) == {"b"}


def test_expired_excluded(env):
    now = datetime.now(timezone.utc)
    past = (now - timedelta(days=1)).isoformat()
    env["ing"].add_preference("pytest traceback preference", expires_at=past)
    res = env["ret"].retrieve(env["q"], k=10, now=now)
    assert all(r.item.source_type is not SourceType.PREFERENCE for r in res)


def test_feedback_memories_need_feedback_consent(env):
    env["consent"].grant(Scope.FEEDBACK_STORAGE)
    env["ing"].add_feedback_memory("pytest traceback explain accepted", suggestion_id="s1")
    kinds = {r.item.source_type for r in env["ret"].retrieve(env["q"], k=10)}
    assert SourceType.FEEDBACK in kinds
    env["consent"].revoke(Scope.FEEDBACK_STORAGE)
    kinds = {r.item.source_type for r in env["ret"].retrieve(env["q"], k=10)}
    assert SourceType.FEEDBACK not in kinds


def test_document_item_with_foreign_uri_excluded(env, te, tmp_path):
    """A document row pointing outside the allowed folders is never returned."""
    rogue = MemoryItem(id="rogue", source_type=SourceType.DOCUMENT, user_approved=True,
                       text_summary="pytest traceback", source_uri=str(tmp_path / "elsewhere.md"))
    env["store"].upsert_item(rogue)
    env["index"].add("rogue", te.embed_text("pytest traceback"))
    ids = {r.item.id for r in env["ret"].retrieve(env["q"], k=10)}
    assert "rogue" not in ids


def test_no_ingestion_without_consent(tmp_path, te):
    folder = tmp_path / "x"
    folder.mkdir()
    ing = Ingestor(MemoryStore(tmp_path / "m.sqlite"), VectorIndex(DIM), ConsentManager(), te)
    with pytest.raises(ConsentError):
        ing.ingest_folder(folder)


def test_home_dir_refused(tmp_path, te):
    consent = ConsentManager()
    consent.grant(Scope.FOLDER_INDEXING)
    with pytest.raises(ConsentError):
        consent.add_folder(Path.home())
    ing = Ingestor(MemoryStore(tmp_path / "m.sqlite"), VectorIndex(DIM), consent, te)
    with pytest.raises(ConsentError):
        ing.ingest_folder(Path.home())
