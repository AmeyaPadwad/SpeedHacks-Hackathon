"""Memory tab: inspect, forget (with verified deletion) and export local memories."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from soma.memory import export_memories, forget, list_for_viewer, verify_deleted

COLUMNS = ("Type", "Summary", "Source", "Approved", "Updated")
NOT_OPEN = "Memory is not open yet."
ExportPicker = Callable[[QWidget], str]


def default_export_picker(parent: QWidget) -> str:
    path, _ = QFileDialog.getSaveFileName(parent, "Export SOMA memories", "soma_memories.json",
                                          "JSON (*.json)")
    return path


class MemoryViewer(QWidget):
    """Table of ``list_for_viewer`` rows with Forget and Export."""

    def __init__(self, store: Any = None, index: Any = None,
                 export_picker: ExportPicker | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store, self.index = store, index
        self._pick = export_picker or default_export_picker
        self._ids: list[str] = []

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 3, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)

        self.refresh_btn = QPushButton("Refresh")
        self.forget_btn = QPushButton("Forget selected")
        self.export_btn = QPushButton("Export…")
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)

        row = QHBoxLayout()
        for b in (self.refresh_btn, self.forget_btn, self.export_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        info = QLabel("Everything SOMA remembers lives on this computer. Forgetting removes the "
                      "record, its vectors and any derived files; your original files are never "
                      "touched.")
        info.setObjectName("muted")
        info.setWordWrap(True)
        lay.addWidget(info)
        lay.addLayout(row)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.status)

        self.refresh_btn.clicked.connect(self.refresh)
        self.forget_btn.clicked.connect(self.forget_selected)
        self.export_btn.clicked.connect(self.export)
        self.refresh()

    def set_memory(self, store: Any, index: Any) -> None:
        self.store, self.index = store, index
        self.status.setText("")
        self.refresh()

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def refresh(self) -> None:
        enabled = self.store is not None
        for b in (self.refresh_btn, self.forget_btn, self.export_btn):
            b.setEnabled(enabled)
        self.table.setRowCount(0)
        self._ids = []
        if not enabled:
            self.status.setText(NOT_OPEN)
            return
        rows = list_for_viewer(self.store)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._ids.append(row["id"])
            vals = (row["type"], row["summary"], row["source"],
                    "yes" if row["approved"] else "no", (row["updated_at"] or "")[:19])
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v or ""))
                item.setData(Qt.ItemDataRole.UserRole, row["id"])
                if c == 1:
                    item.setToolTip(str(v or ""))
                self.table.setItem(r, c, item)
        if not self.status.text() or self.status.text() == NOT_OPEN:
            self.status.setText(f"{len(rows)} memories.")

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return [self._ids[r] for r in rows if r < len(self._ids)]

    def forget_ids(self, ids: list[str]) -> tuple[int, int]:
        """Forget each id and verify deletion. Returns (forgotten, verified)."""
        done = verified = 0
        for item_id in ids:
            if forget(self.store, self.index, item_id):
                done += 1
            if verify_deleted(self.store, self.index, item_id):
                verified += 1
        noun = "memory" if done == 1 else "memories"
        if verified == len(ids):
            msg = f"Forgot {done} {noun}; deletion verified."
        else:
            msg = f"Forgot {done} {noun}; {len(ids) - verified} could not be verified as deleted."
        self.status.setText(msg)
        self.refresh()
        return done, verified

    def forget_selected(self) -> None:
        if self.store is None:
            return
        ids = self.selected_ids()
        if not ids:
            self.status.setText("Select one or more rows to forget.")
            return
        self.forget_ids(ids)

    def export(self) -> str | None:
        if self.store is None:
            return None
        path = self._pick(self)
        if not path:
            return None
        n = export_memories(self.store, path)
        self.status.setText(f"Exported {n} memories (no embeddings or file contents) to {path}.")
        return path
