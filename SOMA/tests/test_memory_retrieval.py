"""Retrieval relevance, reranking, embedder determinism, index persistence, tensor packing."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from soma.config import MemoryConfig
from soma.consent import ConsentManager, Scope
from soma.memory.embedder import ImageEmbedder, TextEmbedder
from soma.memory.ingestion import Ingestor
from soma.memory.retrieval import Retriever, conflict_key, to_tensors
from soma.memory.sqlite_store import MemoryStore
from soma.memory.vector_index import VectorIndex
from soma.types import MemoryItem, RetrievedMemory, SourceType

DIM = 64
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def te():
    return TextEmbedder(dim=DIM, vocab_size=4096)


def _env(tmp_path, te):
    folder = tmp_path / "notes"
    folder.mkdir()
    consent = ConsentManager()
    consent.grant(Scope.FOLDER_INDEXING)
    consent.add_folder(folder)
    store = MemoryStore(tmp_path / "m.sqlite")
    index = VectorIndex(DIM, path=tmp_path / "idx.npz")
    ing = Ingestor(store, index, consent, te)
    return folder, consent, store, index, ing, Retriever(store, index, consent, MemoryConfig())


# ---------------------------------------------------------------- relevance


def test_relevant_note_ranks_above_unrelated(tmp_path, te):
    folder, _, _, _, ing, ret = _env(tmp_path, te)
    (folder / "debugging.md").write_text(
        "When pytest shows a traceback, read the last frame first. The traceback points at the "
        "failing assertion and the test file line.", encoding="utf-8")
    (folder / "recipe.md").write_text(
        "Banana bread: mash bananas, mix flour, sugar, eggs and butter, bake for an hour.",
        encoding="utf-8")
    ing.ingest_folder(folder)
    res = ret.retrieve(te.embed_text("pytest traceback"), k=2)
    assert [Path(r.item.source_uri).name for r in res] == ["debugging.md", "recipe.md"]
    assert res[0].score > res[1].score
    assert res[0].embedding.dtype == np.float32 and res[0].embedding.shape == (DIM,)
    assert ret.permission_violations == 0


def test_empty_memory_returns_empty(tmp_path, te):
    *_, ret = _env(tmp_path, te)
    assert ret.retrieve(te.embed_text("anything"), k=4) == []
    assert ret.retrieve(te.embed_text("anything"), k=0) == []


# ---------------------------------------------------------------- reranking


def _pref(store, index, te, iid, text, updated, tags=()):
    store.upsert_item(MemoryItem(id=iid, source_type=SourceType.PREFERENCE, user_approved=True,
                                 text_summary=text, tags=list(tags), created_at=updated,
                                 updated_at=updated))
    index.add(iid, te.embed_text(text))


def test_recency_breaks_ties(tmp_path, te):
    _, _, store, index, _, ret = _env(tmp_path, te)
    now = datetime.now(timezone.utc)
    _pref(store, index, te, "old", "use black formatter", (now - timedelta(days=365)).isoformat())
    _pref(store, index, te, "new", "use black formatter", (now - timedelta(hours=1)).isoformat())
    res = ret.retrieve(te.embed_text("formatter"), k=2, now=now)
    assert [r.item.id for r in res] == ["new", "old"]


def test_conflicting_preferences_most_recent_wins(tmp_path, te):
    _, _, store, index, _, ret = _env(tmp_path, te)
    now = datetime.now(timezone.utc)
    _pref(store, index, te, "tabs", "indent python with tabs", (now - timedelta(days=10)).isoformat(),
          tags=["conflict:indent"])
    _pref(store, index, te, "spaces", "use four spaces for indentation",
          (now - timedelta(days=1)).isoformat(), tags=["conflict:indent"])
    # Query matches the older one best lexically; it is still superseded.
    res = ret.retrieve(te.embed_text("indent python with tabs"), k=5, now=now)
    ids = [r.item.id for r in res]
    assert "spaces" in ids and "tabs" not in ids
    assert conflict_key(store.get_item("spaces")) == "indent"


def test_source_trust_affects_rank(tmp_path, te):
    folder, consent, store, index, ing, ret = _env(tmp_path, te)
    hi, lo = folder / "hi", folder / "lo"
    hi.mkdir()
    lo.mkdir()
    (hi / "n.md").write_text("deploy checklist", encoding="utf-8")
    (lo / "n.md").write_text("deploy checklist", encoding="utf-8")
    ing.ingest_folder(folder)
    store.set_source(hi, trust=1.0)
    store.set_source(lo, trust=0.0)
    res = ret.retrieve(te.embed_text("deploy checklist"), k=2)
    assert Path(res[0].item.source_uri).parent.name == "hi"


# ---------------------------------------------------------------- embedders


def test_text_embedder_deterministic_unit_vectors(te):
    fresh = TextEmbedder(dim=DIM, vocab_size=4096)
    a, b = te.embed_text("pytest traceback"), fresh.embed_text("pytest traceback")
    assert a.dtype == np.float32 and a.shape == (DIM,)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert te.dim == DIM
    assert te.embed_text("").shape == (DIM,)
    assert not np.allclose(a, te.embed_text("banana bread"))


def test_embedders_do_not_disturb_global_rng():
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    TextEmbedder(dim=16, vocab_size=128)
    ImageEmbedder(dim=16, input_size=16)
    assert torch.equal(torch.rand(3), expected)


def test_text_embedder_deterministic_across_processes(te):
    code = ("import sys; sys.path.insert(0, r'%s');"
            "from soma.memory.embedder import TextEmbedder;"
            "print(','.join('%%.6f' %% x for x in TextEmbedder(dim=%d, vocab_size=4096)"
            ".embed_text('pytest traceback')[:8]))" % (ROOT, DIM))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, timeout=120).stdout.strip()
    other = np.array([float(x) for x in out.split(",")], dtype=np.float32)
    np.testing.assert_allclose(other, te.embed_text("pytest traceback")[:8], atol=1e-5)


def test_image_embedder_deterministic():
    img = np.random.RandomState(0).randint(0, 255, (50, 70, 3)).astype(np.uint8)
    a = ImageEmbedder(dim=DIM, input_size=32).embed_image(img)
    b = ImageEmbedder(dim=DIM, input_size=32).embed_image(img)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert a.dtype == np.float32 and a.shape == (DIM,)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    gray = ImageEmbedder(dim=DIM, input_size=32).embed_image(img[..., 0])
    assert gray.shape == (DIM,)


# ---------------------------------------------------------------- index


def test_index_save_load_roundtrip(tmp_path):
    rng = np.random.RandomState(1)
    idx = VectorIndex(8)
    for i in range(5):
        idx.add(f"id{i}", rng.randn(8))
    p = idx.save(tmp_path / "sub" / "idx.npz")
    loaded = VectorIndex.load(p)
    assert loaded.ids() == idx.ids() and len(loaded) == 5
    q = rng.randn(8)
    a, b = loaded.search(q, 3), idx.search(q, 3)
    assert [i for i, _ in a] == [i for i, _ in b]
    np.testing.assert_allclose([s for _, s in a], [s for _, s in b], atol=1e-6)
    np.testing.assert_allclose(loaded.get("id2"), idx.get("id2"), atol=1e-6)


def test_index_filtered_search_and_remove(tmp_path):
    idx = VectorIndex(3)
    idx.add("a", [1, 0, 0])
    idx.add("b", [0.9, 0.1, 0])
    idx.add("c", [0, 1, 0])
    assert [i for i, _ in idx.search([1, 0, 0], 3, allowed_ids={"c"})] == ["c"]
    assert idx.search([1, 0, 0], 3, allowed_ids=set()) == []
    assert idx.search([1, 0, 0], 3, allowed_ids={"missing"}) == []
    assert idx.remove("a") and not idx.remove("a")
    assert idx.search([1, 0, 0], 1)[0][0] == "b"
    with pytest.raises(ValueError):
        idx.add("bad", [1, 0])


def test_index_open_creates_or_loads(tmp_path):
    p = tmp_path / "i.npz"
    idx = VectorIndex.open(p, dim=4)
    idx.add("x", [1, 0, 0, 0])
    idx.persist()
    assert VectorIndex.open(p, dim=4).ids() == ["x"]
    with pytest.raises(ValueError):
        VectorIndex.open(p, dim=5)


# ---------------------------------------------------------------- tensors


def test_to_tensors_padding():
    mems = [RetrievedMemory(MemoryItem(id=f"m{i}", source_type=SourceType.PREFERENCE,
                                       user_approved=True),
                            np.full(DIM, i + 1, np.float32), 1.0) for i in range(2)]
    emb, mask = to_tensors(mems, K=4, D=DIM)
    assert emb.shape == (4, DIM) and emb.dtype == torch.float32
    assert mask.tolist() == [False, False, True, True]
    assert torch.all(emb[2:] == 0) and float(emb[1, 0]) == 2.0
    emb, mask = to_tensors([], K=3, D=DIM)
    assert mask.all() and not torch.isnan(emb).any()
    emb, mask = to_tensors(mems, K=1, D=DIM)
    assert mask.tolist() == [False]
    with pytest.raises(ValueError):
        to_tensors(mems, K=2, D=DIM + 1)


# ---------------------------------------------------------------- M2 acceptance


def test_m2_acceptance_screen_query_block_and_delete(tmp_path, te):
    """A screen-text query retrieves the relevant allowed memory, never a blocked
    source, and deletion removes it from retrieval and the persisted index."""
    from soma.memory.deletion import forget, verify_deleted

    folder, consent, store, index, ing, ret = _env(tmp_path, te)
    work, private = folder / "work", folder / "private"
    work.mkdir()
    private.mkdir()
    (work / "flaky_tests.md").write_text(
        "Flaky pytest failures: the traceback shows a timeout in test_api. Rerun with -x.",
        encoding="utf-8")
    (work / "meeting.md").write_text("Quarterly planning meeting agenda and budget.",
                                     encoding="utf-8")
    (private / "pytest_diary.md").write_text("pytest traceback private diary entry",
                                             encoding="utf-8")
    ing.ingest_folder(folder)
    store.set_source_blocked(private)

    screen_text = "FAILED tests/test_api.py::test_list - Traceback (most recent call last) pytest"
    res = ret.retrieve(te.embed_text(screen_text), k=3)
    names = [Path(r.item.source_uri).name for r in res]
    assert names[0] == "flaky_tests.md"
    assert "pytest_diary.md" not in names
    assert ret.permission_violations == 0

    top = res[0].item.id
    forget(store, index, top)
    assert verify_deleted(store, index, top)
    again = [r.item.id for r in Retriever(store, VectorIndex.load(tmp_path / "idx.npz"), consent)
             .retrieve(te.embed_text(screen_text), k=3)]
    assert top not in again
