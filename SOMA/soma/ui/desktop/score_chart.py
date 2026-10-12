"""Horizontal bar chart of the policy's candidate score distribution (QPainter only)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from soma.ui.desktop import theme

NO_SUGGESTION = "NO_SUGGESTION"


class ScoreChart(QWidget):
    """One bar per label (probability 0..1); the chosen label is drawn in the accent colour.

    ``NO_SUGGESTION`` is always listed first so abstention is visible.
    """

    ROW_H = 22
    LABEL_W = 210

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, float]] = []
        self._chosen: str | None = None
        self._stale = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.ROW_H * 2)
        self.setToolTip("Policy probabilities per candidate, including staying silent.")

    # -- data
    def set_distribution(self, distribution: dict[str, float], chosen: str | None = None) -> None:
        items = [(str(k), max(0.0, float(v))) for k, v in (distribution or {}).items()]
        items.sort(key=lambda kv: (kv[0] != NO_SUGGESTION, -kv[1]))
        self._items = items
        self._chosen = chosen if chosen is not None else (max(items, key=lambda kv: kv[1])[0]
                                                         if items else None)
        self._stale = False
        self.setFixedHeight(max(2, len(items)) * self.ROW_H + 8)
        self.update()

    def set_stale(self, stale: bool) -> None:
        self._stale = stale
        self.update()

    def clear(self) -> None:
        self.set_distribution({})
        self._chosen = None

    @property
    def items(self) -> list[tuple[str, float]]:
        return list(self._items)

    @property
    def chosen(self) -> str | None:
        return self._chosen

    def total(self) -> float:
        return sum(v for _, v in self._items)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(420, max(2, len(self._items)) * self.ROW_H + 8)

    # -- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        if not self._items:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No scores yet")
            p.end()
            return
        label_w = min(self.LABEL_W, int(w * 0.45))
        bar_x = label_w + 8
        bar_w = max(10, w - bar_x - 56)
        for i, (label, value) in enumerate(self._items):
            y = 4 + i * self.ROW_H
            chosen = label == self._chosen
            p.setPen(QColor(theme.TEXT if chosen else theme.MUTED))
            p.drawText(QRectF(0, y, label_w, self.ROW_H - 4),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.BACKGROUND))
            p.drawRoundedRect(QRectF(bar_x, y + 3, bar_w, self.ROW_H - 10), 3, 3)
            color = theme.ACCENT if chosen and not self._stale else theme.BAR
            if self._stale:
                color = theme.STALE if chosen else theme.BORDER
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(bar_x, y + 3, bar_w * min(1.0, value), self.ROW_H - 10), 3, 3)
            p.setPen(QPen(QColor(theme.TEXT if chosen else theme.MUTED)))
            p.drawText(QRectF(bar_x + bar_w + 6, y, 50, self.ROW_H - 4),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{value:.2f}")
        p.end()
