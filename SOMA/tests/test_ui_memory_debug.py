"""Memory viewer (forget/export) and debug overlay (in-memory only)."""

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from soma.memory import MemoryStore, VectorIndex
from soma.perception import render_scenario
from soma.types import MemoryItem, SourceType
from soma.ui.desktop.debug_overlay import MAX_W, PLACEHOLDER, DebugOverlay
from soma.ui.desktop.memory_viewer import NOT_OPEN, MemoryViewer


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path / "m.sqlite")
    index = VectorIndex(8, path=tmp_path / "idx.npz")
    for i in range(3):
        store.upsert_item(MemoryItem(id=f"mem{i}", source_type=SourceType.PREFERENCE,
                                     user_approved=True, text_summary=f"prefers option {i}"))
        index.add(f"mem{i}", np.ones(8, dtype=np.float32) * (i + 1))
    index.persist()
    yield store, index
    store.close()


def test_viewer_without_store(qapp):
    v = MemoryViewer()
    assert v.table.rowCount() == 0 and v.status.text() == NOT_OPEN
    assert not v.forget_btn.isEnabled()


def test_viewer_lists_rows(qapp, memory):
    v = MemoryViewer(*memory)
    assert v.table.rowCount() == 3
    assert sorted(v.ids) == ["mem0", "mem1", "mem2"]
    summaries = {v.table.item(r, 1).text() for r in range(3)}
    assert "prefers option 1" in summaries


def test_forget_removes_row_and_verifies(qapp, memory):
    store, index = memory
    v = MemoryViewer(store, index)
    row = v.ids.index("mem1")
    v.table.selectRow(row)
    assert v.selected_ids() == ["mem1"]
    v.forget_btn.click()
    assert "mem1" not in v.ids and v.table.rowCount() == 2
    assert "verified" in v.status.text()
    assert store.get_item("mem1") is None and "mem1" not in index


def test_forget_requires_selection(qapp, memory):
    v = MemoryViewer(*memory)
    v.table.clearSelection()
    v.forget_selected()
    assert v.table.rowCount() == 3 and "Select" in v.status.text()


def test_export(qapp, memory, tmp_path):
    out = tmp_path / "export.json"
    v = MemoryViewer(*memory, export_picker=lambda _p: str(out))
    assert v.export() == str(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["memories"]) == 3
    assert "embedding" not in json.dumps(data["memories"])


def test_debug_overlay_downscales_and_clears(qapp, tmp_path):
    image, obs, _ = render_scenario("editor_error", seed=1, size=(1920, 1080)) \
        if "editor_error" in __import__("soma.perception", fromlist=["SCENARIOS"]).SCENARIOS \
        else render_scenario(__import__("soma.perception", fromlist=["SCENARIOS"]).SCENARIOS[0],
                             seed=1, size=(1920, 1080))
    d = DebugOverlay()
    before = set(tmp_path.iterdir())
    d.show_frame(image, obs)
    assert d.has_image
    w, h = d.pixmap_size
    assert 0 < w <= MAX_W and h > 0
    assert "elements" in d.caption.text()
    d.clear()
    assert not d.has_image and d.view.text() == PLACEHOLDER
    assert set(tmp_path.iterdir()) == before  # nothing written


def test_debug_overlay_without_frame_uses_layout(qapp):
    _, obs, _ = render_scenario("terminal_traceback")
    d = DebugOverlay()
    d.show_frame(None, obs)
    assert d.has_image and "layout only" in d.caption.text()
