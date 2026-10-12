"""SOMA main window: state + pause, active window, OCR text, scores, suggestion, timings.

All widgets are updated on the Qt main thread from :class:`RuntimeBridge`
signals. Feedback goes to ``runtime.feedback``; nothing is ever executed.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from soma.capture import redact_text
from soma.consent import ConsentManager, Scope
from soma.policy import distribution_labels
from soma.types import FeedbackOutcome, Suggestion
from soma.ui.desktop import theme
from soma.ui.desktop.bridge import RuntimeBridge
from soma.ui.desktop.debug_overlay import DebugOverlay
from soma.ui.desktop.memory_viewer import ExportPicker, MemoryViewer
from soma.ui.desktop.score_chart import NO_SUGGESTION, ScoreChart
from soma.ui.desktop.suggestion_card import SuggestionCard

TIMING_WINDOW = 50
STAGE_ORDER = ("capture", "ocr", "parse", "encode", "retrieve", "policy", "gate", "total")


def chosen_label(result: Any) -> str:
    """Distribution label of the suggested candidate, or ``NO_SUGGESTION``."""
    s: Suggestion = result.suggestion
    if s.is_silent:
        return NO_SUGGESTION
    labels = distribution_labels(result.candidates)
    for i, c in enumerate(result.candidates):
        if c.id == s.candidate.id or c is s.candidate:
            return labels[i + 1]
    return s.candidate.action.value


class MainWindow(QMainWindow):
    permissions_requested = Signal()
    _ingest_done = Signal(str)

    def __init__(self, bridge: RuntimeBridge, consent: ConsentManager, *,
                 memory: Any = None, frame_tap: Any = None,
                 export_picker: ExportPicker | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SOMA")
        self.resize(1080, 720)
        self.setStyleSheet(theme.STYLESHEET)
        self.bridge = bridge
        self.consent = consent
        self.memory = memory
        self.frame_tap = frame_tap
        self._timings: dict[str, deque[float]] = {}
        self._last_result: Any = None
        self.feedback_errors: list[str] = []

        # -- header: state + pause
        self.state_badge = QLabel("Off")
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("pause")
        self.pause_btn.setToolTip("Stop capturing immediately")
        self.perm_btn = QPushButton("Permissions…")
        self.app_label = QLabel("Active app: –")
        self.title_label = QLabel("")
        self.title_label.setObjectName("muted")
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        win_box = QVBoxLayout()
        win_box.addWidget(self.app_label)
        win_box.addWidget(self.title_label)
        header = QHBoxLayout()
        header.addWidget(self.state_badge)
        header.addLayout(win_box, 1)
        header.addWidget(self.perm_btn)
        header.addWidget(self.pause_btn)

        # -- live tab
        self.ocr_table = QTableWidget(0, 3)
        self.ocr_table.setHorizontalHeaderLabels(("Text", "Role", "Confidence"))
        self.ocr_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ocr_table.verticalHeader().setVisible(False)
        hh = self.ocr_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self._heading("Recognized text"))
        lv.addWidget(self.ocr_table, 1)

        self.chart = ScoreChart()
        self.card = SuggestionCard()
        self.skip_label = QLabel("")
        self.skip_label.setObjectName("muted")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self._heading("Suggestion"))
        rv.addWidget(self.card)
        rv.addWidget(self._heading("Candidate scores (PyTorch policy)"))
        rv.addWidget(self.chart)
        rv.addWidget(self.skip_label)
        rv.addStretch(1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([480, 560])

        self.memory_view = MemoryViewer(getattr(memory, "store", None),
                                        getattr(memory, "index", None),
                                        export_picker=export_picker)
        self.debug = DebugOverlay()
        self.tabs = QTabWidget()
        self.tabs.addTab(split, "Live")
        self.tabs.addTab(self.memory_view, "Memory")
        self.tabs.addTab(self.debug, "Debug")

        self.timing_label = QLabel("Timings: –")
        self.timing_label.setObjectName("muted")
        self.ingest_label = QLabel("")
        self.ingest_label.setObjectName("muted")

        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addLayout(header)
        cv.addWidget(self.tabs, 1)
        footer = QHBoxLayout()
        footer.addWidget(self.timing_label, 1)
        footer.addWidget(self.ingest_label)
        cv.addLayout(footer)
        self.setCentralWidget(central)

        # -- wiring
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.perm_btn.clicked.connect(self.permissions_requested.emit)
        self.card.feedback.connect(self.send_feedback)
        bridge.result.connect(self.on_result)
        bridge.state_changed.connect(self.on_state)
        bridge.error.connect(lambda m: self.skip_label.setText(f"Error: {m}"))
        self._ingest_done.connect(self._on_ingest_done, Qt.ConnectionType.QueuedConnection)
        self.on_state(bridge.state)

    @staticmethod
    def _heading(text: str) -> QLabel:
        lab = QLabel(text.upper())
        lab.setObjectName("muted")
        return lab

    # ------------------------------------------------------------ state
    @property
    def state(self) -> str:
        return self.state_badge.text()

    @Slot(str)
    def on_state(self, state: str) -> None:
        self.state_badge.setText(state)
        self.state_badge.setStyleSheet(theme.state_badge_style(state))
        self.pause_btn.setText("Resume" if state == "Paused" else "Pause")
        self.pause_btn.setEnabled(state != "Off")
        if state != "Capturing":
            self._drop_frames()
            self.card.mark_stale()
            self.chart.set_stale(True)

    def _drop_frames(self) -> None:
        self.debug.clear()
        if self.frame_tap is not None:
            self.frame_tap.clear()

    @Slot()
    def toggle_pause(self) -> None:
        """Instant: the bridge bumps its generation before the runtime is told."""
        self.bridge.toggle()

    def set_memory(self, memory: Any) -> None:
        self.memory = memory
        self.memory_view.set_memory(getattr(memory, "store", None), getattr(memory, "index", None))

    # ------------------------------------------------------------ results
    @Slot(object)
    def on_result(self, res: Any) -> None:
        if res is None:
            return
        self._last_result = res
        self._record_timings(res.timings_ms)
        if res.skipped_reason:
            self.skip_label.setText(f"Skipped: {res.skipped_reason}")
            return
        self.skip_label.setText("")
        self._show_window(res.window)
        self._show_ocr(res.observation)
        self.chart.set_distribution(res.suggestion.distribution, chosen_label(res))
        self._show_suggestion(res)
        if self.bridge.state == "Capturing":
            image = self.frame_tap.latest(res.frame_version) if self.frame_tap is not None else None
            self.debug.show_frame(image, res.observation)

    def _show_window(self, window: Any) -> None:
        self.app_label.setText(f"Active app: {window.app_name or 'unknown'}")
        if self.consent.allowed(Scope.APP_MONITORING):
            self.title_label.setText(redact_text(window.title or ""))
        else:
            self.title_label.setText("Window title hidden (app monitoring is off)")

    def _show_ocr(self, obs: Any) -> None:
        els = list(obs.elements)
        self.ocr_table.setRowCount(len(els))
        for r, e in enumerate(els):
            vals = (redact_text(e.text), e.ui_role or "", f"{float(e.confidence):.2f}")
            for c, v in enumerate(vals):
                self.ocr_table.setItem(r, c, QTableWidgetItem(v))

    def _show_suggestion(self, res: Any) -> None:
        s: Suggestion = res.suggestion
        if not s.is_silent:
            self.card.show_suggestion(s)
            return
        shown = self.card.suggestion
        if self.card.active and shown is not None:
            if shown.screen_version != res.frame_version:
                self.card.mark_stale()     # screen changed under an unanswered suggestion
            return
        self.card.show_suggestion(s)

    def _record_timings(self, timings: dict[str, float] | None) -> None:
        for stage, ms in (timings or {}).items():
            self._timings.setdefault(stage, deque(maxlen=TIMING_WINDOW)).append(float(ms))
        self.timing_label.setText(self.timing_text())

    def timing_p50(self) -> dict[str, float]:
        return {k: statistics.median(v) for k, v in self._timings.items() if v}

    def timing_text(self) -> str:
        p50 = self.timing_p50()
        if not p50:
            return "Timings: –"
        keys = [k for k in STAGE_ORDER if k in p50] + sorted(k for k in p50 if k not in STAGE_ORDER)
        n = max(len(v) for v in self._timings.values())
        return "P50 ms: " + " · ".join(f"{k} {p50[k]:.0f}" for k in keys) + f"  (n={n})"

    # ------------------------------------------------------------ feedback
    @Slot(str, object, object)
    def send_feedback(self, suggestion_id: str, outcome: FeedbackOutcome,
                      edited_text: str | None) -> None:
        rt = self.bridge.runtime
        if rt is None:
            return
        try:
            rt.feedback(suggestion_id, outcome, edited_text)
        except Exception as exc:  # feedback must never crash the UI
            self.feedback_errors.append(type(exc).__name__)
            self.card.status.setText(f"Feedback not recorded: {exc}")

    # ------------------------------------------------------------ ingestion
    def ingest_folders(self, folders: list[str], ingestor: Any,
                       spawn: Callable[[Callable[[], None]], None] | None = None) -> None:
        """Index user-chosen folders off the main thread; report counts only."""
        if not folders or ingestor is None:
            return
        self.ingest_label.setText(f"Indexing {len(folders)} folder(s)…")

        def work() -> None:
            added = errors = 0
            msgs = []
            for f in folders:
                try:
                    rep = ingestor.ingest_folder(f)
                    added += rep.changed
                    errors += len(rep.errors)
                except Exception as exc:
                    msgs.append(f"{type(exc).__name__}: {exc}")
            text = f"Indexed {added} file(s)" + (f", {errors} error(s)" if errors else "")
            if msgs:
                text += " · " + "; ".join(msgs)
            self._ingest_done.emit(text)

        (spawn or (lambda fn: threading.Thread(target=fn, daemon=True).start()))(work)

    @Slot(str)
    def _on_ingest_done(self, text: str) -> None:
        self.ingest_label.setText(text)
        self.memory_view.refresh()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.bridge.stop()
        self._drop_frames()
        super().closeEvent(event)
