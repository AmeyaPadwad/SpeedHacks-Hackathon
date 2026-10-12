"""Ingestor: consent gating, walk rules, redaction, idempotency, images, explicit memories."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from soma.config import MemoryConfig
from soma.consent import ConsentError, ConsentManager, Scope
from soma.memory.embedder import ImageEmbedder, TextEmbedder
from soma.memory.ingestion import Ingestor, item_id_for, redact
from soma.memory.sqlite_store import SUMMARY_CAP, MemoryStore
from soma.memory.vector_index import VectorIndex
from soma.types import SourceType

DIM = 64


@pytest.fixture(scope="module")
def embedders():
    return TextEmbedder(dim=DIM, vocab_size=4096), ImageEmbedder(dim=DIM, input_size=32)


def _setup(tmp_path: Path, embedders, grant: bool = True, **kw):
    folder = tmp_path / "notes"
    folder.mkdir(exist_ok=True)
    consent = ConsentManager()
    if grant:
        consent.grant(Scope.FOLDER_INDEXING)
        consent.add_folder(folder)
    store = MemoryStore(tmp_path / "db" / "m.sqlite")
    index = VectorIndex(DIM, path=tmp_path / "db" / "idx.npz")
    te, ie = embedders
    ing = Ingestor(store, index, consent, te, ie, MemoryConfig(max_file_bytes=10_000), **kw)
    return folder, consent, store, index, ing


def test_no_ingestion_without_consent(tmp_path, embedders):
    folder, consent, store, index, ing = _setup(tmp_path, embedders, grant=False)
    (folder / "a.md").write_text("hello", encoding="utf-8")
    with pytest.raises(ConsentError):
        ing.ingest_folder(folder)
    consent.grant(Scope.FOLDER_INDEXING)          # scope granted but folder not added
    with pytest.raises(ConsentError):
        ing.ingest_folder(folder)
    assert store.count_items() == 0 and len(index) == 0


def test_home_directory_refused(tmp_path, embedders):
    *_, ing = _setup(tmp_path, embedders)
    with pytest.raises(ConsentError):
        ing.ingest_folder(Path.home())
    with pytest.raises(ConsentError):
        ing.consent.add_folder(Path.home())


def test_folder_outside_allowed_refused(tmp_path, embedders):
    *_, ing = _setup(tmp_path, embedders)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ConsentError):
        ing.ingest_folder(other)


def test_walk_rules(tmp_path, embedders):
    folder, _, store, index, ing = _setup(tmp_path, embedders)
    (folder / "keep.md").write_text("pytest traceback notes", encoding="utf-8")
    (folder / "code.py").write_text("print('hi')", encoding="utf-8")
    (folder / "bin.exe").write_bytes(b"MZ....")
    (folder / "big.txt").write_text("x" * 20_000, encoding="utf-8")
    (folder / ".env").write_text("TOKEN=abc", encoding="utf-8")
    hidden = folder / ".git"
    hidden.mkdir()
    (hidden / "config.txt").write_text("secret", encoding="utf-8")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("nested note", encoding="utf-8")
    report = ing.ingest_folder(folder)
    names = sorted(Path(p).name for p in report.added)
    assert names == ["code.py", "deep.txt", "keep.md"]
    assert store.count_items() == 3 and len(index) == 3
    assert (tmp_path / "db" / "idx.npz").exists()       # index persisted after ingest
    for item in store.list_items():
        assert item.source_type is SourceType.DOCUMENT
        prov = item.provenance
        assert {"path", "mtime", "sha256", "ingested_at", "size"} <= set(prov)
        assert item.source_uri == prov["path"]


def test_symlink_escaping_folder_skipped(tmp_path, embedders):
    folder, _, store, *_rest = _setup(tmp_path, embedders)
    ing = _rest[-1]
    outside = tmp_path / "secret.txt"
    outside.write_text("outside data", encoding="utf-8")
    try:
        os.symlink(outside, folder / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    report = ing.ingest_folder(folder)
    assert report.added == []
    assert store.count_items() == 0


def test_reingest_unchanged_is_noop_and_change_updates(tmp_path, embedders):
    folder, _, store, index, ing = _setup(tmp_path, embedders)
    f = folder / "a.md"
    f.write_text("first version", encoding="utf-8")
    r1 = ing.ingest_folder(folder)
    item = store.get_item(item_id_for(f))
    r2 = ing.ingest_folder(folder)
    assert len(r1.added) == 1 and r2.changed == 0 and len(r2.unchanged) == 1
    assert store.get_item(item.id).updated_at == item.updated_at
    f.write_text("second version, different", encoding="utf-8")
    r3 = ing.ingest_folder(folder)
    assert len(r3.updated) == 1
    again = store.get_item(item.id)
    assert again.provenance["sha256"] != item.provenance["sha256"]
    assert again.created_at == item.created_at
    assert store.count_items() == 1 and len(index) == 1


def test_summary_is_short_and_redacted(tmp_path, embedders):
    folder, _, store, *_r = _setup(tmp_path, embedders)
    ing = _r[-1]
    body = "api_key = sk-ABCDEFGHIJKLMNOPQRSTUVWX\n" + "lorem ipsum " * 300
    (folder / "s.txt").write_text(body, encoding="utf-8")
    ing.ingest_folder(folder)
    item = store.list_items()[0]
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in item.text_summary
    assert len(item.text_summary) <= SUMMARY_CAP
    assert len(item.text_summary) < len(body)


def test_redact_fallback():
    assert "hunter2" not in redact("password: hunter2")


def test_images_embedded_without_thumbnail_by_default(tmp_path, embedders):
    folder, _, store, index, ing = _setup(tmp_path, embedders)
    Image.fromarray(np.full((40, 30, 3), 200, np.uint8)).save(folder / "shot.png")
    ing.ingest_folder(folder)
    item = store.list_items()[0]
    assert item.source_type is SourceType.IMAGE
    assert "shot.png" in item.text_summary and "30x40" in item.text_summary
    assert "thumbnail" not in item.provenance
    assert not store.derived_dir.exists() or not any(store.derived_dir.iterdir())
    assert index.get(item.id).shape == (DIM,)


def test_thumbnail_only_when_flag_allows(tmp_path, embedders):
    folder, _, store, index, ing = _setup(tmp_path, embedders, save_thumbnails=True)
    Image.fromarray(np.zeros((300, 300, 3), np.uint8)).save(folder / "big.png")
    ing.ingest_folder(folder)
    item = store.list_items()[0]
    thumb = Path(item.provenance["thumbnail"])
    assert thumb.exists() and thumb.parent == store.derived_dir
    with Image.open(thumb) as im:
        assert max(im.size) <= 128


def test_preferences_and_ui_events(tmp_path, embedders):
    _, consent, store, index, ing = _setup(tmp_path, embedders)
    pref = ing.add_preference("I prefer pytest -x", tags=["testing"], conflict_key="test-runner")
    assert pref.source_type is SourceType.PREFERENCE
    assert "conflict:test-runner" in pref.tags
    assert pref.id in index
    with pytest.raises(ConsentError):
        ing.add_ui_event("opened terminal", app="Code.exe")
    with pytest.raises(ConsentError):
        ing.add_feedback_memory("accepted explain-error", suggestion_id="sug_1")
    consent.grant(Scope.FEEDBACK_STORAGE)
    ev = ing.add_ui_event("opened terminal", app="Code.exe")
    fb = ing.add_feedback_memory("accepted explain-error", suggestion_id="sug_1")
    assert ev.source_type is SourceType.UI_EVENT and ev.provenance["app"] == "Code.exe"
    assert fb.source_type is SourceType.FEEDBACK
    assert store.count_items() == 3
