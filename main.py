"""Pette overlay entry point. Demo wiring only: the proposal below is canned, not AI output."""
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from pet.ui import Overlay, Proposal


def demo_proposal() -> Proposal:
    return Proposal(
        title="DEMO: Slack draft to Sarah",
        body="Hey Sarah, could you review my changes to the auth middleware when you get a chance? Thanks!",
        action_label="Send",
    )


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = Overlay()

    def ask():
        overlay.set_state("thinking")
        overlay.say("Thinking...", seconds=None)
        overlay.set_state("thinking")
        QTimer.singleShot(1200, lambda: None if overlay.paused else overlay.propose(demo_proposal()))

    overlay.ask_requested.connect(ask)
    overlay.approved.connect(lambda _: overlay.say("Demo only: nothing was sent.", seconds=4))
    overlay.cancelled.connect(lambda _: overlay.say("Cancelled. Nothing sent.", seconds=3))
    overlay.place_default()
    overlay.show()
    overlay.say("Hi! Click me to ask.", seconds=5)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
