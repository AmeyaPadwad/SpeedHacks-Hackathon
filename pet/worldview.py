"""World-model window: a panel docked beside Pette showing what she has learned (db/world_model.py)
and the behaviour each fact drives, plus the live workflow status.

The UI is a React + Tailwind + shadcn/ui page (web/ -> built to pet/web/index.html as one file) shown in
Qt WebEngine. Python pushes `WorldModel.snapshot()` + live state as JSON via `window.__pette(data)`.
Facts only: no message text, code, audio or frames are ever passed to the page.
"""
import json
import time

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .sprites import asset_path

PAGE = asset_path("index.html", folder="pet/web")
WIDTH, HEIGHT = 380, 600
GAP = 10
REFRESH_MS = 1500


class _Follow(QObject):
    """Re-dock the window whenever Pette moves or resizes."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Move, QEvent.Resize, QEvent.Show) and self.win.isVisible():
            self.win.dock()
        return False


class WorldWindow(QWidget):
    def __init__(self, overlay, world, pipeline=None, watcher=None, *, web=True):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("Pette · world model")
        self.resize(WIDTH, HEIGHT)
        self.overlay, self.world, self.pipeline, self.watcher = overlay, world, pipeline, watcher
        self.state, self.detail = "idle", ""
        self.view = None
        self._ready = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if web and PAGE.exists():
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
                self.view = QWebEngineView(self)
                self.view.page().setBackgroundColor(QColor(0, 0, 0, 0))
                self.view.loadFinished.connect(self._loaded)
                self.view.titleChanged.connect(self._title)
                self.view.load(QUrl.fromLocalFile(str(PAGE)))
                lay.addWidget(self.view)
            except ImportError:
                self.view = None
        if self.view is None:   # no WebEngine / page not built: plain text fallback
            self.fallback = QLabel(wordWrap=True, alignment=Qt.AlignTop)
            self.fallback.setStyleSheet("background:#fff;border:1px solid #e4e4e7;border-radius:12px;"
                                        "padding:14px;font-family:'Segoe UI';font-size:13px;color:#09090b;")
            lay.addWidget(self.fallback)
        overlay.installEventFilter(_Follow(self))
        if pipeline is not None:
            pipeline.status.connect(self.on_status)
        self._timer = QTimer(self, interval=REFRESH_MS, timeout=self.refresh)

    # --- visibility / docking ----------------------------------------------
    def show_near(self):
        self.dock()
        self.show()
        self._timer.start()
        self.refresh()

    def toggle(self):
        if self.isVisible():
            self.hide()
            self._timer.stop()
        else:
            self.show_near()

    def dock(self):
        """Left of Pette, bottom-aligned; flips to the right if there's no room."""
        g = self.overlay.frameGeometry()
        screen = QGuiApplication.screenAt(g.center()) or QGuiApplication.primaryScreen()
        a = screen.availableGeometry()
        x = g.left() - WIDTH - GAP
        if x < a.left():
            x = min(g.right() + GAP, a.right() - WIDTH)
        y = max(a.top(), min(g.bottom() - HEIGHT, a.bottom() - HEIGHT))
        self.setGeometry(x, y, WIDTH, HEIGHT)

    # --- data --------------------------------------------------------------
    def on_status(self, state: str, detail: str):
        self.state, self.detail = state, detail
        self.refresh()

    def data(self) -> dict:
        snap = self.world.snapshot()
        orb = self.overlay.orb
        snap["live"] = {
            "state": self.state, "detail": self.detail, "label": orb.label(),
            "camera": bool(self.watcher and self.watcher.camera is not None),
            "gesture": orb.gesture,
            "present": bool(self.watcher.present) if self.watcher else True,
            "eyes_closed": bool(self.watcher.eyes_closed) if self.watcher else False,
            "paused": self.overlay.paused,
            "now": time.time(),
        }
        return snap

    def refresh(self):
        if not self.isVisible():
            return
        try:
            data = self.data()
        except Exception as e:   # a db hiccup must never take the overlay down
            data = {"error": type(e).__name__}
        if self.view is None:
            self.fallback.setText(self.world.summary() if "error" not in data else "World model unavailable.")
        elif self._ready:
            self.view.page().runJavaScript(f"window.__pette && window.__pette({json.dumps(data)})")

    def _loaded(self, ok: bool):
        self._ready = ok
        self.refresh()

    def _title(self, title: str):
        if title.startswith("close:"):   # the page's close button
            self.toggle()
