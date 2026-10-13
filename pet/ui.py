"""Pette on screen: the camera orb (her body, bottom-right, draggable) and SOMA's speech bubble, docked
at the bottom centre of the screen and toggled with Ctrl+Alt+B or the tray.

Styling follows shadcn/ui (zinc palette, hairline borders, soft shadow); the bubble is frosted glass
(translucent gradient fill, top sheen, bright inner edge) so the screen shows through. Two
Aceternity-style touches painted in Qt: a rotating gradient "moving border" on the bubble while a send
is counting down, and a glowing conic ring around the camera orb that changes colour with the status.
"""
import uuid
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction, QBrush, QColor, QConicalGradient, QFont, QGuiApplication, QIcon, QImage, QLinearGradient, QPainter,
    QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QMenu, QPushButton, QSystemTrayIcon,
    QVBoxLayout, QWidget,
)

# shadcn/ui zinc tokens
BG, FG, MUTED, BORDER, RING = "#ffffff", "#09090b", "#71717a", "#e4e4e7", "#18181b"
SECONDARY = "#f4f4f5"
# Glass (frosted, translucent) bubble: alpha 0..255 for the fill's top/bottom, the sheen and the edges.
GLASS_TOP, GLASS_BOTTOM, GLASS_SHEEN, GLASS_EDGE, GLASS_HAIRLINE = 182, 142, 110, 190, 34
FONT ="'Inter', 'Segoe UI Variable Text', 'Segoe UI', sans-serif"
# status -> (ring colour, label)
STATUS = {
    "idle": ("#a1a1aa", ""),
    "looking": ("#3b82f6", "Looking"),
    "suggesting": ("#f59e0b", "Suggestion"),
    "done": ("#22c55e", "Done ✓"),
    "dismissed": ("#71717a", "Not now"),
    "indexing": ("#8b5cf6", "Indexing"),
    "failed": ("#ef4444", "{d}"),
}
ORB = 116       # camera orb diameter
BAR_MARGIN = 18 # gap between the speech bubble and the bottom of the screen (above the taskbar)
TAIL = 10       # speech-bubble tail height
MOODS = ("idle", "away", "blink", "sleeping")   # webcam presence / pause, drawn on the orb


@dataclass
class Proposal:
    """A card in the bubble: title, text and two buttons. `glow` adds the moving gradient border."""
    title: str
    body: str
    action_label: str = "Do it"
    glow: bool = False
    cancel_label: str = "Cancel"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


def tray_icon(size: int = 64) -> QIcon:
    """Pette's tray icon, drawn in code: a violet ring around a soft glass dot (no image files)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QPointF(size / 2, size / 2)
    grad = QConicalGradient(c, 90)
    for stop, col in ((0.0, "#8b5cf6"), (0.5, "#3b82f6"), (1.0, "#8b5cf6")):
        grad.setColorAt(stop, QColor(col))
    p.setPen(QPen(QBrush(grad), size * 0.12))
    p.setBrush(QColor(250, 250, 252, 230))
    p.drawEllipse(c, size * 0.38, size * 0.38)
    p.end()
    return QIcon(pm)


def _font(px: int, weight=QFont.Normal) -> QFont:
    f = QFont(["Inter", "Segoe UI Variable Text", "Segoe UI"])
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


class Orb(QWidget):
    """Pette's body: you (mirrored webcam, local only), your hand skeleton and gesture, a glowing status
    ring, and her mood (dozing while you're away or paused, eyes shut when yours are). Click = look now;
    drag = move her."""
    clicked = Signal()
    HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                  (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

    def __init__(self):
        super().__init__()
        self.setFixedSize(ORB + 28, ORB + 40)
        self.image: QImage | None = None
        self.hand: list[tuple[float, float]] = []
        self.gesture = ""
        self.camera_on = False
        self.feed_visible = True      # Ctrl+Alt+C: hide the camera picture, keep Pette (ring, moods, clicks)
        self.state, self.detail = "idle", ""
        self.level = 0.0
        self.mood = "idle"
        self._phase = 0.0
        self._press = None
        self._moved = False
        self._timer = QTimer(self, interval=33, timeout=self._spin)
        self._timer.start()
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click: look at my screen now · drag to move\nCamera: processed on this PC, never stored")

    def set_frame(self, frame: dict):
        """{'image': QImage, 'hand': [(x, y)...] 0..1, 'gesture': str} from the camera thread (queued)."""
        self.image = frame.get("image")
        self.hand = frame.get("hand") or []
        self.gesture = frame.get("gesture") or ""
        self.update()

    def set_camera(self, on: bool):
        self.camera_on = on
        if not on:
            self.image, self.hand, self.gesture = None, [], ""
        self.update()

    def set_status(self, state: str, detail: str = ""):
        self.state, self.detail = state, detail
        self.update()

    def set_mood(self, mood: str):
        if mood in MOODS and mood != self.mood:
            self.mood = mood
            self.update()

    # --- click vs drag (moves the whole window) -----------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            self._moved = False

    def mouseMoveEvent(self, e):
        if self._press is not None:
            target = e.globalPosition().toPoint() - self._press
            if (target - self.window().pos()).manhattanLength() > 3:
                self._moved = True
                self.setCursor(Qt.ClosedHandCursor)
            self.window().move(target)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._press is not None:
            self._press = None
            self.setCursor(Qt.PointingHandCursor)
            if not self._moved:
                self.clicked.emit()

    def label(self) -> str:
        text = STATUS.get(self.state, STATUS["idle"])[1]
        return text.format(d=self.detail) if "{d}" in text else text

    def _spin(self):
        speed = 6 if self.state in ("looking", "indexing", "suggesting") else 1.2
        self._phase = (self._phase + speed) % 360
        self.level *= 0.85
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        dozing = self.mood in ("away", "sleeping")
        if dozing:
            p.setOpacity(0.6)
        color = QColor(STATUS.get(self.state, STATUS["idle"])[0])
        cx, cy, r = self.width() / 2, ORB / 2 + 12, ORB / 2
        # glow + rotating conic ring (Aceternity "moving border")
        glow = QColor(color)
        for i, a in enumerate((26, 16, 8)):
            glow.setAlpha(a + int(self.level * 60))
            p.setPen(QPen(glow, 6 + i * 5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r + 2, r + 2)
        grad = QConicalGradient(QPointF(cx, cy), self._phase)
        light = QColor(color).lighter(150)
        grad.setColorAt(0.0, color)
        grad.setColorAt(0.35, QColor(color.red(), color.green(), color.blue(), 40))
        grad.setColorAt(0.6, light)
        grad.setColorAt(1.0, color)
        p.setPen(QPen(QBrush(grad), 3.5 + self.level * 3))
        p.drawEllipse(QPointF(cx, cy), r, r)
        # inner: webcam or placeholder
        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), r - 3, r - 3)
        p.save()
        p.setClipPath(clip)
        inner = QRectF(cx - r + 3, cy - r + 3, 2 * r - 6, 2 * r - 6)
        if not self.feed_visible:
            p.fillRect(inner, QColor("#ede9fe"))
            p.setPen(QPen(QColor("#8b5cf6"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy - 6), 20, 11)                 # a closed eye: camera hidden
            p.drawLine(QPointF(cx - 24, cy + 8), QPointF(cx + 24, cy - 20))
            p.setFont(_font(10))
            p.setPen(QColor(MUTED))
            p.drawText(inner.adjusted(0, 0, 0, -12), Qt.AlignHCenter | Qt.AlignBottom, "camera hidden")
        elif self.image is not None and not self.image.isNull():
            p.drawImage(inner, self.image)
            if self.hand:
                p.setPen(QPen(QColor(255, 255, 255, 220), 1.6))
                pts = [QPointF(inner.left() + x * inner.width(), inner.top() + y * inner.height()) for x, y in self.hand]
                for a, b in self.HAND_EDGES:
                    if a < len(pts) and b < len(pts):
                        p.drawLine(pts[a], pts[b])
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#a78bfa"))
                for pt in pts:
                    p.drawEllipse(pt, 2.2, 2.2)
        else:
            p.fillRect(inner, QColor("#f4f4f5"))
            p.setPen(QPen(QColor("#a1a1aa"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy - 10), 15, 15)
            p.drawArc(QRectF(cx - 28, cy + 10, 56, 44), 0, 180 * 16)
            p.setFont(_font(10))
            p.setPen(QColor(MUTED))
            p.drawText(inner.adjusted(0, 0, 0, -12), Qt.AlignHCenter | Qt.AlignBottom,
                       "camera off" if not self.camera_on else "starting…")
        if self.mood == "blink":   # your eyes are shut: a soft lid comes down over the orb
            lid = QLinearGradient(inner.topLeft(), inner.bottomLeft())
            lid.setColorAt(0.0, QColor(24, 24, 27, 200))
            lid.setColorAt(0.55, QColor(24, 24, 27, 150))
            lid.setColorAt(0.56, QColor(24, 24, 27, 0))
            p.fillRect(inner, QBrush(lid))
        p.restore()
        if dozing:                 # away or paused: Pette dozes
            p.setOpacity(1)
            p.setPen(QColor("#8b5cf6"))
            p.setFont(_font(16, QFont.DemiBold))
            off = int(self._phase / 6) % 14
            p.drawText(int(cx + r - 10), int(cy - r + 22 - off // 2), "z")
            p.drawText(int(cx + r + 2), int(cy - r + 10 - off // 2), "Z")
        # gesture chip (top) and status pill (bottom), shadcn Badge style
        if self.gesture and self.feed_visible:
            self._pill(p, self.gesture, cx, cy - r + 4, QColor(255, 255, 255, 235), QColor(FG), QColor(BORDER))
        text = self.label()
        if text:
            self._pill(p, text, cx, cy + r + 2, color, QColor("white"), color)

    @staticmethod
    def _pill(p: QPainter, text: str, cx: float, cy: float, bg: QColor, fg: QColor, border: QColor):
        p.setFont(_font(11, QFont.DemiBold))
        w = p.fontMetrics().horizontalAdvance(text) + 16
        rect = QRectF(cx - w / 2, cy - 10, w, 20)
        p.setPen(QPen(border, 1))
        p.setBrush(bg)
        p.drawRoundedRect(rect, 10, 10)
        p.setPen(fg)
        p.drawText(rect, Qt.AlignCenter, text)


class Bubble(QWidget):
    """Speech bubble (frosted-glass card): title, text, and the card's two buttons."""
    approved = Signal(str)
    cancelled = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(320)
        self.setStyleSheet(f"""
            QLabel {{ color: {FG}; font-family: {FONT}; background: transparent; }}
            QLabel#title {{ font-size: 12px; font-weight: 600; color: #3f3f46; letter-spacing: 0.2px; }}
            QLabel#body {{ font-size: 14px; }}
            QPushButton {{ font-family: {FONT}; font-size: 13px; font-weight: 500; padding: 6px 12px;
                          border-radius: 8px; border: 1px solid rgba(255, 255, 255, 170);
                          background: rgba(255, 255, 255, 95); color: {FG}; }}
            QPushButton:hover {{ background: rgba(255, 255, 255, 150); }}
            QPushButton#primary {{ background: rgba(24, 24, 27, 215); color: #fafafa;
                                  border-color: rgba(24, 24, 27, 230); }}
            QPushButton#primary:hover {{ background: rgba(39, 39, 42, 235); }}
            QPushButton:disabled {{ background: rgba(244, 244, 245, 90); color: #a1a1aa;
                                   border-color: rgba(255, 255, 255, 120); }}
        """)
        shadow = QGraphicsDropShadowEffect(self, blurRadius=32, offset=QPointF(0, 8), color=QColor(0, 0, 0, 38))
        self.setGraphicsEffect(shadow)
        self.title = QLabel(objectName="title")
        self.body = QLabel(objectName="body", wordWrap=True, textInteractionFlags=Qt.TextSelectableByMouse)
        self.primary = QPushButton(objectName="primary")
        self.cancel = QPushButton("Cancel")
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.addStretch()
        row.addWidget(self.cancel)
        row.addWidget(self.primary)
        self.buttons = QWidget()
        self.buttons.setLayout(row)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 14)
        lay.setSpacing(6)
        for w in (self.title, self.body, self.buttons):
            lay.addWidget(w)
        self.proposal: Proposal | None = None
        self.progress = None          # 0..1 time left on a card, drawn as a bar under it
        self.tail = False             # draw a speech-bubble tail under the card
        self._phase = 0.0
        self._border = QTimer(self, interval=33, timeout=self._spin)
        self.primary.clicked.connect(self._approve)
        self.cancel.clicked.connect(self._cancel)

    # --- content ----------------------------------------------------------
    def show_text(self, text: str, title: str = ""):
        self.proposal = None
        self.progress = None          # a plain message never carries a card's time-left bar
        self._set_title(title)
        self.body.setText(text)
        self.buttons.hide()
        self._glow(False)
        self.show()

    def show_proposal(self, p: Proposal):
        self.proposal = p
        self._set_title(p.title)
        self.body.setText(p.body)
        self.primary.setText(p.action_label)
        self.cancel.setText(p.cancel_label)
        self.primary.setEnabled(True)
        self.cancel.setEnabled(True)
        self.buttons.show()
        self._glow(p.glow)
        self.show()

    def set_progress(self, fraction: float | None):
        """Time left on the card (1 -> 0), drawn as a thin bar; None hides it."""
        self.progress = None if fraction is None else max(0.0, min(1.0, fraction))
        self.update()

    def _set_title(self, title: str):
        self.title.setText(title)
        self.title.setVisible(bool(title))

    # --- actions ----------------------------------------------------------
    def consume(self) -> str | None:
        """Single-use decision bound to the displayed proposal id."""
        p, self.proposal = self.proposal, None
        self.primary.setEnabled(False)
        self.cancel.setEnabled(False)
        self._glow(False)
        return p.id if p else None

    _consume = consume

    def _approve(self):
        if (pid := self.consume()) is not None:
            self.approved.emit(pid)

    def _cancel(self):
        if (pid := self.consume()) is not None:
            self.cancelled.emit(pid)

    # --- paint (card + moving border + time-left bar) ---------------------
    def _glow(self, on: bool):
        if on:
            self._border.start()
        else:
            self._border.stop()
        self.update()

    def _spin(self):
        self._phase = (self._phase + 4) % 360
        self.update()

    def set_tail(self, on: bool):
        if on != self.tail:
            self.tail = on
            m = self.layout().contentsMargins()
            self.layout().setContentsMargins(m.left(), m.top(), m.right(), 14 + (TAIL if on else 0))
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5 - (TAIL if self.tail else 0))
        path = QPainterPath()
        path.addRoundedRect(r, 14, 14)
        if self.tail:                 # a speech-bubble tail pointing down, slightly left of centre
            tip = QPointF(r.center().x() - 18, r.bottom() + TAIL)
            tail = QPainterPath(QPointF(tip.x() - 9, r.bottom() - 1))
            tail.lineTo(tip)
            tail.lineTo(QPointF(tip.x() + 13, r.bottom() - 1))
            tail.closeSubpath()
            path = path.united(tail)
        # frosted glass: translucent white body, a soft sheen across the top, bright inner edge
        body = QLinearGradient(r.topLeft(), r.bottomLeft())
        body.setColorAt(0.0, QColor(255, 255, 255, GLASS_TOP))
        body.setColorAt(1.0, QColor(250, 250, 252, GLASS_BOTTOM))
        p.fillPath(path, QBrush(body))
        sheen = QLinearGradient(r.topLeft(), QPointF(r.left(), r.top() + min(48.0, r.height() * 0.5)))
        sheen.setColorAt(0.0, QColor(255, 255, 255, GLASS_SHEEN))
        sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillPath(path, QBrush(sheen))
        if self._border.isActive():
            grad = QConicalGradient(r.center(), self._phase)
            for stop, c in ((0.0, "#f59e0b"), (0.25, "#ec4899"), (0.5, "#8b5cf6"), (0.75, "#3b82f6"), (1.0, "#f59e0b")):
                grad.setColorAt(stop, QColor(c))
            p.setPen(QPen(QBrush(grad), 2))
            p.drawPath(path)
        else:
            p.setPen(QPen(QColor(0, 0, 0, GLASS_HAIRLINE), 1))   # hairline so the glass reads on white
            p.drawPath(path)
            inner = QPainterPath()
            inner.addRoundedRect(r.adjusted(1, 1, -1, -1), 13, 13)
            p.setPen(QPen(QColor(255, 255, 255, GLASS_EDGE), 1))
            p.drawPath(inner)
        if self.progress is not None:
            track = QRectF(r.left() + 16, r.bottom() - 6, r.width() - 32, 3)   # r excludes the tail
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 120))
            p.drawRoundedRect(track, 1.5, 1.5)
            p.setBrush(QColor(RING))
            p.drawRoundedRect(QRectF(track.left(), track.top(), track.width() * self.progress, 3), 1.5, 1.5)


class BubbleBar(QWidget):
    """SOMA's speech bubble as its own window, docked at the bottom centre of the screen Pette is on."""

    def __init__(self, screen_of=None, avoid=None):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("Pette · SOMA suggestion")
        self.bubble = Bubble()
        self.bubble.setFixedWidth(460)
        self.bubble.set_tail(True)
        self.bubble.hide()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 8)       # room for the glass shadow
        lay.addWidget(self.bubble)
        self._screen_of = screen_of or (lambda: QGuiApplication.primaryScreen())
        self._avoid = avoid or (lambda: None)        # Pette's own window: never cover her

    def reanchor(self, change, visible: bool):
        """Apply a bubble change, resize, and dock bottom-centre; show only when there is something to say."""
        change()
        self.bubble.adjustSize()
        self.adjustSize()
        a = self._screen_of().availableGeometry()
        g = QRect(a.center().x() - self.width() // 2, a.bottom() - self.height() - BAR_MARGIN + 8,
                  self.width(), self.height())
        pette = self._avoid()
        if pette is not None and g.intersects(pette):   # small screen: slide left of Pette
            g.moveLeft(max(a.left(), pette.left() - g.width() - 8))
        self.setGeometry(g)
        self.setVisible(visible and not self.bubble.isHidden())


class Overlay(QWidget):
    """Pette: the orb in its own always-on-top window, plus the bottom-of-screen speech bubble (`bar`)."""
    approved = Signal(str)
    cancelled = Signal(str)
    look_requested = Signal()         # click Pette / tray: ask SOMA to look at the screen now
    pause_toggled = Signal(bool)
    pinch_on_proposal = Signal(str)   # proposal id; pinch = "Do it" on the card
    world_requested = Signal()
    camera_view_requested = Signal()  # hotkey: hide / show the camera picture (safe to emit from any thread)
    bubble_toggle_requested = Signal()  # hotkey: speech bubble on / off (safe to emit from any thread)
    bubble_toggled = Signal(bool)

    def __init__(self, tray: bool = True):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.orb = Orb()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 8)
        lay.addWidget(self.orb, 0, Qt.AlignHCenter)
        self.bar = BubbleBar(screen_of=self._screen, avoid=lambda: self.geometry() if self.isVisible() else None)
        self.bubble = self.bar.bubble
        self.bubble_enabled = True    # Ctrl+Alt+B / tray: the speech bubble (and SOMA's cards) on or off
        self.paused = False
        self.context_allowed = True   # False while the foreground app isn't VS Code / a terminal (FocusGate)
        self.away = self.eyes_closed = False
        self.bubble.approved.connect(self._decided(self.approved))
        self.bubble.cancelled.connect(self._decided(self.cancelled))
        self.orb.clicked.connect(self._on_orb_click)
        self.camera_view_requested.connect(self.toggle_camera_view)
        self.bubble_toggle_requested.connect(self.toggle_bubble)
        self._hide_timer = QTimer(self, singleShot=True, timeout=self.dismiss)
        self._pause_action = None
        self._bubble_action = None
        self._tray = self._build_tray() if tray else None

    # --- public API -------------------------------------------------------
    def _update_mood(self):
        self.orb.set_mood("sleeping" if self.paused else "away" if self.away
                          else "blink" if self.eyes_closed else "idle")

    def set_status(self, state: str, detail: str = ""):
        """SOMA status (pet/soma.py): drives the orb ring and pill."""
        self.orb.set_status(state, detail)

    def set_level(self, level: float):
        self.orb.level = max(self.orb.level, level)

    def set_progress(self, fraction: float | None):
        self.bubble.set_progress(fraction)

    def set_context_allowed(self, allowed: bool):
        """Hide Pette (and her bubble) outside VS Code / terminals (--focus-only)."""
        self.context_allowed = allowed
        if allowed:
            self.show()
        else:
            self.hide()
        self._reanchor(lambda: None)

    def reveal(self):
        if self.context_allowed:
            self.show()

    def set_presence(self, away: bool, eyes_closed: bool):
        """Webcam reactions: Pette dozes while you're away and shuts her eyes with yours."""
        self.away, self.eyes_closed = away, eyes_closed
        self._update_mood()

    def set_camera_indicator(self, on: bool):
        self.orb.set_camera(on)

    @property
    def camera_visible(self) -> bool:
        return self.orb.feed_visible

    def set_camera_visible(self, visible: bool):
        """Hide or show the camera picture inside the orb. Pette (ring, moods, clicks) stays, and the camera
        keeps its on/off state, so gestures still work while the picture is hidden."""
        if visible != self.orb.feed_visible:
            self.orb.feed_visible = visible
            self.orb.update()

    def toggle_camera_view(self):
        self.set_camera_visible(not self.camera_visible)

    def set_bubble_enabled(self, on: bool):
        """Speech bubble on/off. Off: nothing shows at the bottom of the screen and SOMA's cards wait."""
        if on == self.bubble_enabled:
            return
        self.bubble_enabled = on
        if not on:
            self.dismiss()
        if self._bubble_action and self._bubble_action.isChecked() != on:
            self._bubble_action.setChecked(on)
        self.bubble_toggled.emit(on)
        self.notify("Pette", "Speech bubble on." if on else "Speech bubble off: suggestions wait until it's back.")

    def toggle_bubble(self):
        self.set_bubble_enabled(not self.bubble_enabled)

    def pinch(self):
        """The only gesture: pinch on a suggestion card = "Do it". Otherwise ignored."""
        if self.paused or not self.context_allowed:
            return
        if self.bubble.proposal:
            self.pinch_on_proposal.emit(self.bubble.proposal.id)

    def nudge(self):
        """Back from being away: bring an open suggestion card to attention."""
        p = self.bubble.proposal
        if self.paused or p is None or self.bubble.isHidden():
            return
        if not self.bubble.title.text().startswith("While you were away"):
            self.bubble.title.setText(f"While you were away · {p.title}")
        self.bar.raise_()

    def add_menu_action(self, text: str, slot=None, *, checkable=False, checked=False) -> QAction | None:
        """Insert a tray action above Bring to front / Quit. No-op without a tray (tests)."""
        if not self._tray:
            return None
        act = QAction(text, self._menu, checkable=checkable)
        if checkable:
            act.setChecked(checked)
            if slot:
                act.toggled.connect(slot)
        elif slot:
            act.triggered.connect(slot)
        self._menu.insertAction(self._front_action, act)
        return act

    def add_menu(self, text: str) -> QMenu | None:
        """Insert a tray submenu above Bring to front / Quit. No-op without a tray (tests)."""
        if not self._tray:
            return None
        sub = QMenu(text, self._menu)
        self._menu.insertMenu(self._front_action, sub)
        return sub

    def notify(self, title: str, text: str):
        if self._tray:
            self._tray.showMessage(title, text, QSystemTrayIcon.Information, 4000)

    def say(self, text: str, title: str = "", seconds: float | None = 6):
        if self.paused or not self.bubble_enabled:
            return
        self._reanchor(lambda: self.bubble.show_text(text, title))
        self._arm_hide(seconds)

    def propose(self, p: Proposal):
        if self.paused or not self.bubble_enabled:
            return
        self._reanchor(lambda: self.bubble.show_proposal(p))
        self._arm_hide(None)

    def dismiss(self):
        self._hide_timer.stop()
        self.bubble.proposal = None
        self.bubble.progress = None
        self.bubble._glow(False)
        self._reanchor(self.bubble.hide)

    def set_paused(self, paused: bool):
        self.paused = paused
        if paused:
            self.dismiss()  # also drops an open card so it can't be accepted later
        self._update_mood()
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
        return handler

    def _on_orb_click(self):
        """Clicking Pette (the orb) asks SOMA to look at the screen now."""
        if self.paused:
            return
        self.look_requested.emit()

    def _reanchor(self, change):
        """Apply a bubble change; the bar docks itself at the bottom of the screen Pette is on."""
        self.bar.reanchor(change, visible=self.context_allowed and self.bubble_enabled and not self.paused)

    def _screen(self):
        return QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()

    @staticmethod
    def _clamp(g: QRect) -> QRect:
        screen = QGuiApplication.screenAt(g.center()) or QGuiApplication.primaryScreen()
        a = screen.availableGeometry()
        g.moveLeft(max(a.left(), min(g.left(), a.right() - g.width() + 1)))
        g.moveTop(max(a.top(), min(g.top(), a.bottom() - g.height() + 1)))
        return g

    def _build_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(tray_icon(), self)
        tray.setToolTip("Pette")
        menu = QMenu()
        menu.addAction(QAction("Look at my screen now", menu, triggered=self.look_requested.emit))
        menu.addAction(QAction("World model", menu, triggered=self.world_requested.emit))
        self._bubble_action = QAction("Speech bubble (Ctrl+Alt+B)", menu, checkable=True)
        self._bubble_action.setChecked(True)
        self._bubble_action.toggled.connect(lambda c: c != self.bubble_enabled and self.set_bubble_enabled(c))
        menu.addAction(self._bubble_action)
        self._pause_action = QAction("Pause all", menu, checkable=True)
        self._pause_action.toggled.connect(lambda c: c != self.paused and self.set_paused(c))
        menu.addAction(self._pause_action)
        menu.addSeparator()
        self._front_action = QAction("Bring to front", menu, triggered=self._bring_front)
        menu.addAction(self._front_action)
        menu.addAction(QAction("Quit", menu, triggered=QApplication.quit))
        tray.setContextMenu(menu)
        self._menu = menu
        tray.show()
        return tray

    def _bring_front(self):
        self.setGeometry(self._clamp(self.geometry()))
        self.show()
        self.raise_()
        self._reanchor(lambda: None)

    def closeEvent(self, e):
        self.bar.close()
        super().closeEvent(e)
