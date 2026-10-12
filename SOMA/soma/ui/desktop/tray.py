"""System tray icon with Pause/Resume and Quit; the tooltip shows the active state."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from soma.ui.desktop import theme
from soma.ui.desktop.bridge import RuntimeBridge


def state_icon(state: str, size: int = 32) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(theme.STATE_COLORS.get(state, theme.OFF)))
    p.drawEllipse(3, 3, size - 6, size - 6)
    p.end()
    return QIcon(pm)


class SomaTray(QObject):
    """Global Pause/Resume. Works without a system tray (actions still exist)."""

    def __init__(self, bridge: RuntimeBridge, app: QApplication | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.menu = QMenu()
        self.toggle_action = QAction("Pause", self.menu)
        self.show_action = QAction("Show SOMA", self.menu)
        self.quit_action = QAction("Quit", self.menu)
        self.menu.addAction(self.toggle_action)
        self.menu.addAction(self.show_action)
        self.menu.addSeparator()
        self.menu.addAction(self.quit_action)
        self.toggle_action.triggered.connect(bridge.toggle)
        self.quit_action.triggered.connect(self._quit)
        self._app = app
        self.icon: QSystemTrayIcon | None = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.icon = QSystemTrayIcon(state_icon(bridge.state), self)
            self.icon.setContextMenu(self.menu)
            self.icon.show()
        bridge.state_changed.connect(self.on_state)
        self.on_state(bridge.state)

    @property
    def tooltip(self) -> str:
        return self._tooltip

    def on_state(self, state: str) -> None:
        self._tooltip = f"SOMA: {state}"
        self.toggle_action.setText("Resume" if state == "Paused" else "Pause")
        self.toggle_action.setEnabled(state != "Off")
        if self.icon is not None:
            self.icon.setToolTip(self._tooltip)
            self.icon.setIcon(state_icon(state))

    def _quit(self) -> None:
        self.bridge.stop()
        app = self._app or QApplication.instance()
        if app is not None:
            app.quit()

    def hide(self) -> None:
        if self.icon is not None:
            self.icon.hide()
