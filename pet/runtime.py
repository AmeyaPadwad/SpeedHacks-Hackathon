"""Let terminal signals stop Qt even when its native event loop is idle."""
import signal

from PySide6.QtCore import QObject, QTimer


class TerminalShutdown(QObject):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.requested = False
        self._previous = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._request)
        self.timer = QTimer(self, interval=100, timeout=self._poll)
        self.timer.start()
        app.aboutToQuit.connect(self.close)

    def _request(self, signum, frame):
        # Signal handlers only set a flag; Qt cleanup runs from its own loop.
        self.requested = True

    def _poll(self):
        if self.requested:
            self.app.quit()

    def close(self):
        self.timer.stop()
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()
