"""Floating overlay: Pette sprite, speech bubble, confirmation buttons, tray."""
import math
import uuid
from dataclasses import dataclass, field

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QSystemTrayIcon,
    QVBoxLayout, QWidget,
)

from .sprites import ANIMATIONS, SpriteSheet

MINT, CREAM, INK = "#7CF2C4", "#FCF5E8", "#1E1E1E"
SPRITE_SCALE = 0.9
FPS = 15
FRAME_MS = 450  # how long each animation frame holds


@dataclass
class Proposal:
    """A drafted action shown for approval. Nothing runs until the user clicks the action button."""
    title: str
    body: str
    action_label: str = "Send"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class PetWidget(QWidget):
    """Paints the animated sprite; distinguishes a click from a drag."""
    clicked = Signal()

    def __init__(self, sheet: SpriteSheet):
        super().__init__()
        self.sheet = sheet
        self.state = "idle"
        self._tick = 0
        self._press = None
        self._moved = False
        w, h = int(sheet.cell_w * SPRITE_SCALE), int(sheet.cell_h * SPRITE_SCALE)
        self.setFixedSize(w + 8, h + 14)
        self.setCursor(Qt.OpenHandCursor)
        self._timer = QTimer(self, interval=1000 // FPS, timeout=self._advance)
        self._timer.start()

    def set_state(self, state: str):
        if state != self.state:
            self.state, self._tick = state, 0
            self.update()

    def _advance(self):
        self._tick += 1
        self.update()

    def paintEvent(self, _):
        seq = ANIMATIONS[self.state]
        frame = seq[(self._tick * 1000 // FPS // FRAME_MS) % len(seq)]
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        t = self._tick / FPS
        sleeping = self.state == "sleeping"
        bob = 0 if sleeping else round(math.sin(t * 2.2) * 2)
        if sleeping:
            p.setOpacity(0.7)
        p.drawPixmap(QRect(4, 6 + bob, self.width() - 8, self.height() - 14), self.sheet.frames[frame])
        if sleeping:
            p.setOpacity(1)
            p.setPen(QColor(MINT))
            f = p.font()
            f.setPointSize(13)
            f.setItalic(True)
            p.setFont(f)
            off = int(t * 8) % 14
            p.drawText(self.width() - 40, 34 - off // 2, "z")
            p.drawText(self.width() - 26, 22 - off // 2, "Z")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            self._moved = False
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._press is not None:
            target = e.globalPosition().toPoint() - self._press
            if (target - self.window().pos()).manhattanLength() > 3:
                self._moved = True
            self.window().move(target)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._press is not None:
            self._press = None
            self.setCursor(Qt.OpenHandCursor)
            if not self._moved:
                self.clicked.emit()


class Bubble(QFrame):
    """Speech bubble: message text, action/Cancel buttons for a proposal, or a typed-command field."""
    approved = Signal(str)
    cancelled = Signal(str)
    submitted = Signal(str)
    input_cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("bubble")
        self.setFixedWidth(300)
        self.setStyleSheet(f"""
            #bubble {{ background: {CREAM}; border: 2px solid {INK}; border-radius: 14px; }}
            QLabel {{ color: {INK}; font-family: 'Segoe UI'; background: transparent; }}
            #title {{ font-size: 10pt; font-weight: 600; color: #3C7A62; }}
            #body {{ font-size: 11pt; }}
            QPushButton {{ font-family: 'Segoe UI'; font-size: 10pt; padding: 6px 14px;
                          border-radius: 8px; border: 1px solid {INK}; background: white; color: {INK}; }}
            QPushButton:hover {{ background: #EFE6D2; }}
            QPushButton#primary {{ background: {INK}; color: {MINT}; font-weight: 600; }}
            QPushButton#primary:hover {{ background: #333; }}
            QPushButton:disabled {{ background: #D8D2C4; color: #8A8A8A; border-color: #8A8A8A; }}
            QLineEdit {{ font-family: 'Segoe UI'; font-size: 11pt; padding: 5px 8px; border-radius: 8px;
                        border: 1px solid {INK}; background: white; color: {INK}; }}
        """)
        self.title = QLabel(objectName="title")
        self.body = QLabel(objectName="body", wordWrap=True, textInteractionFlags=Qt.TextSelectableByMouse)
        self.input = QLineEdit()
        self.input.returnPressed.connect(self._submit)
        self.input.hide()
        self.primary = QPushButton(objectName="primary")
        self.cancel = QPushButton("Cancel")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch()
        row.addWidget(self.cancel)
        row.addWidget(self.primary)
        self.buttons = QWidget()
        self.buttons.setLayout(row)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(6)
        for w in (self.title, self.body, self.input, self.buttons):
            lay.addWidget(w)
        self.proposal: Proposal | None = None
        self.asking = False
        self.primary.clicked.connect(self._approve)
        self.cancel.clicked.connect(self._cancel)

    def show_text(self, text: str, title: str = ""):
        self.proposal = None
        self.asking = False
        self.title.setText(title)
        self.title.setVisible(bool(title))
        self.body.setText(text)
        self.input.hide()
        self.buttons.hide()
        self.show()

    def show_input(self, prompt: str):
        self.proposal = None
        self.asking = True
        self.title.hide()
        self.body.setText(prompt)
        self.input.clear()
        self.input.setPlaceholderText("e.g. Slack Sarah asking her to review this file")
        self.input.show()
        self.primary.setText("Go")
        self.primary.setEnabled(True)
        self.cancel.setEnabled(True)
        self.buttons.show()
        self.show()
        self.input.setFocus()

    def show_proposal(self, p: Proposal):
        self.proposal = p
        self.asking = False
        self.input.hide()
        self.title.setText(p.title)
        self.title.show()
        self.body.setText(p.body)
        self.primary.setText(p.action_label)
        self.primary.setEnabled(True)
        self.cancel.setEnabled(True)
        self.buttons.show()
        self.show()

    def _consume(self) -> str | None:
        """Single-use approval bound to the displayed proposal id."""
        p, self.proposal = self.proposal, None
        self.primary.setEnabled(False)
        self.cancel.setEnabled(False)
        return p.id if p else None

    def _submit(self):
        text = self.input.text().strip()
        if self.asking and text:
            self.asking = False
            self.input.hide()
            self.submitted.emit(text)

    def _approve(self):
        if self.asking:
            self._submit()
        elif (pid := self._consume()) is not None:
            self.approved.emit(pid)

    def _cancel(self):
        if self.asking:
            self.asking = False
            self.input.hide()
            self.input_cancelled.emit()
        elif (pid := self._consume()) is not None:
            self.cancelled.emit(pid)


class Overlay(QWidget):
    """Frameless always-on-top window: bubble above the pet, bottom edge anchored."""
    approved = Signal(str)
    cancelled = Signal(str)
    ask_requested = Signal()
    pause_toggled = Signal(bool)
    text_submitted = Signal(str)
    input_cancelled = Signal()

    def __init__(self, sheet: SpriteSheet | None = None, tray: bool = True):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.pet = PetWidget(sheet or SpriteSheet())
        self.bubble = Bubble()
        self.bubble.hide()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)
        lay.addWidget(self.bubble, 0, Qt.AlignHCenter)
        lay.addWidget(self.pet, 0, Qt.AlignHCenter)
        self.paused = False
        self.bubble.approved.connect(self._decided(self.approved))
        self.bubble.cancelled.connect(self._decided(self.cancelled))
        self.bubble.submitted.connect(self.text_submitted)
        self.bubble.input_cancelled.connect(self._on_input_cancelled)
        self.pet.clicked.connect(self._on_pet_click)
        self._hide_timer = QTimer(self, singleShot=True, timeout=self.dismiss)
        self._pause_action = None
        self._tray = self._build_tray() if tray else None

    # --- public API -------------------------------------------------------
    def set_state(self, state: str):
        self.pet.set_state("sleeping" if self.paused else state)

    def say(self, text: str, title: str = "", seconds: float | None = 6):
        if self.paused:
            return
        self._reanchor(lambda: self.bubble.show_text(text, title))
        self.set_state("speaking")
        self._arm_hide(seconds)

    def propose(self, p: Proposal):
        if self.paused:
            return
        self._reanchor(lambda: self.bubble.show_proposal(p))
        self.set_state("speaking")
        self._arm_hide(None)  # a pending approval never times out silently

    def ask_text(self, prompt: str):
        """Typed-command fallback; the window takes focus so the user can type."""
        if self.paused:
            return
        self._reanchor(lambda: self.bubble.show_input(prompt))
        self.set_state("idle")
        self._arm_hide(None)
        self.activateWindow()
        self.raise_()
        self.bubble.input.setFocus()

    def dismiss(self):
        self._hide_timer.stop()
        self.bubble.proposal = None
        self.bubble.asking = False
        self._reanchor(self.bubble.hide)
        self.set_state("idle")

    def set_paused(self, paused: bool):
        self.paused = paused
        if paused:
            self.dismiss()  # also drops any pending proposal so it can't be approved later
        self.pet.set_state("sleeping" if paused else "idle")
        if self._pause_action and self._pause_action.isChecked() != paused:
            self._pause_action.setChecked(paused)
        self.pause_toggled.emit(paused)

    def place_default(self):
        self.adjustSize()
        a = QGuiApplication.primaryScreen().availableGeometry()
        self.move(a.right() - self.width() - 24, a.bottom() - self.height() - 24)

    # --- internals --------------------------------------------------------
    def _arm_hide(self, seconds):
        self._hide_timer.stop()
        if seconds:
            self._hide_timer.start(int(seconds * 1000))

    def _decided(self, signal):
        def handler(pid):
            signal.emit(pid)
            self.set_state("idle")
        return handler

    def _on_input_cancelled(self):
        self.input_cancelled.emit()
        self.dismiss()

    def _on_pet_click(self):
        if self.bubble.isVisible():
            if not (self.bubble.proposal or self.bubble.asking):
                self.dismiss()
        elif not self.paused:
            self.ask_requested.emit()

    def _reanchor(self, change):
        """Apply a bubble change, then resize keeping the pet fixed on screen."""
        old = self.geometry()
        change()
        self.adjustSize()
        g = self.geometry()
        g.moveBottom(old.bottom())
        g.moveLeft(old.center().x() - g.width() // 2)
        self.setGeometry(self._clamp(g))

    @staticmethod
    def _clamp(g: QRect) -> QRect:
        screen = QGuiApplication.screenAt(g.center()) or QGuiApplication.primaryScreen()
        a = screen.availableGeometry()
        g.moveLeft(max(a.left(), min(g.left(), a.right() - g.width() + 1)))
        g.moveTop(max(a.top(), min(g.top(), a.bottom() - g.height() + 1)))
        return g

    def _build_tray(self) -> QSystemTrayIcon:
        icon = QIcon(self.pet.sheet.frames[0].scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("Pette")
        menu = QMenu()
        menu.addAction(QAction("Ask Pette", menu, triggered=self.ask_requested.emit))
        self._pause_action = QAction("Pause all", menu, checkable=True)
        self._pause_action.toggled.connect(lambda c: c != self.paused and self.set_paused(c))
        menu.addAction(self._pause_action)
        menu.addAction(QAction("Bring to front", menu, triggered=self._bring_front))
        menu.addAction(QAction("Quit", menu, triggered=QApplication.quit))
        tray.setContextMenu(menu)
        self._menu = menu
        tray.show()
        return tray

    def _bring_front(self):
        self.setGeometry(self._clamp(self.geometry()))
        self.show()
        self.raise_()
