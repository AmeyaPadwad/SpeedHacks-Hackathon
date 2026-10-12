"""Deletion removes memories from the store, the persisted index and derived files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from soma.config import MemoryConfig
from soma.consent import ConsentManager, Scope
from soma.memory.deletion import forget, forget_source, purge_expired, verify_deleted
from soma.memory.embedder import ImageEmbedder, TextEmbedder
from soma.memory.ingestion import Ingestor, item_id_for
from soma.memory.retrieval import Retriever
from soma.memory.sqlite_store import MemoryStore
from soma.memory.vector_index import VectorIndex
from soma.types import FeedbackEvent, FeedbackOutcome

DIM = 64


@pytest.fixture(scope="module")
def embedders():
    return TextEmbedder(dim=DIM, vocab_size=4096), ImageEmbedder(dim=DIM, input_size=32)


@pytest.fixture
def env(tmp_path, embedders):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "trace.md").write_text("pytest traceback notes", encoding="utf-8")
    (folder / "other.md").write_text("weekly planning notes", encoding="utf-8")
    Image.fromarray(np.full((64, 64, 3), 90, np.uint8)).save(folder / "pic.png")
    consent = ConsentManager()
    consent.grant(Scope.FOLDER_INDEXING)
    consent.add_folder(folder)
    store = MemoryStore(tmp_path / "db" / "m.sqlite")
    idx_path = tmp_path / "db" / "idx.npz"
    index = VectorIndex(DIM, path=idx_path)
    te, ie = embedders
    ing = Ingestor(store, index, consent, te, ie, MemoryConfig(), save_thumbnails=True)
    ing.ingest_folder(folder)
    return dict(folder=folder, consent=consent, store=store, index=index, idx_path=idx_path,
                ing=ing, te=te, ret=Retriever(store, index, consent))


def test_forget_removes_from_store_index_and_retrieval(env):
    trace = item_id_for(env["folder"] / "trace.md")
    env["store"].add_feedback(FeedbackEvent("s1", FeedbackOutcome.ACCEPTED, context_ref=trace))
    q = env["te"].embed_text("pytest traceback")
    assert trace in {r.item.id for r in env["ret"].retrieve(q, k=5)}
    assert not verify_deleted(env["store"], env["index"], trace)

    assert forget(env["store"], env["index"], trace)
    assert verify_deleted(env["store"], env["index"], trace)
    assert trace not in {r.item.id for r in env["ret"].retrieve(q, k=5)}
    reloaded = VectorIndex.load(env["idx_path"])
    assert trace not in reloaded and len(reloaded) == 2
    assert env["store"].list_feedback() == []
    assert (env["folder"] / "trace.md").exists()          # the original file is untouched
    assert not forget(env["store"], env["index"], trace)  # idempotent


def test_forget_image_removes_thumbnail(env):
    pic = item_id_for(env["folder"] / "pic.png")
    thumb = Path(env["store"].get_item(pic).provenance["thumbnail"])
    assert thumb.exists()
    forget(env["store"], env["index"], pic)
    assert not thumb.exists()
    assert verify_deleted(env["store"], env["index"], pic)
    assert (env["folder"] / "pic.png").exists()


def test_verify_detects_stale_persisted_index(env):
    other = item_id_for(env["folder"] / "other.md")
    env["store"].delete_item(other)
    env["index"].remove(other)           # removed in memory but not yet persisted
    assert not verify_deleted(env["store"], env["index"], other)
    env["index"].persist()
    assert verify_deleted(env["store"], env["index"], other)


def test_verify_detects_leftover_derived_file(env):
    item_id = "mem_leftover"
    env["store"].derived_dir.mkdir(parents=True, exist_ok=True)
    (env["store"].derived_dir / f"{item_id}.png").write_bytes(b"x")
    assert not verify_deleted(env["store"], env["index"], item_id)


def test_forget_source(env):
    sub = env["folder"] / "sub"
    sub.mkdir()
    (sub / "a.md").write_text("sub note", encoding="utf-8")
    env["ing"].ingest_folder(env["folder"])
    assert forget_source(env["store"], env["index"], sub) == 1
    assert env["store"].items_under(sub) == []
    assert forget_source(env["store"], env["index"], env["folder"]) == 3
    assert env["store"].count_items() == 0
    assert len(VectorIndex.load(env["idx_path"])) == 0
    assert not env["store"].derived_dir.exists() or not any(env["store"].derived_dir.iterdir())


def test_purge_expired(env):
    now = datetime.now(timezone.utc)
    old = env["ing"].add_preference("temporary", expires_at=(now - timedelta(minutes=1)).isoformat())
    keep = env["ing"].add_preference("lasting", expires_at=(now + timedelta(days=1)).isoformat())
    assert purge_expired(env["store"], env["index"], now) == 1
    assert verify_deleted(env["store"], env["index"], old.id)
    assert env["store"].get_item(keep.id) is not None
