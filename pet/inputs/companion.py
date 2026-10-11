"""Pette owns capture controls; the review card opens only when requested."""
import time

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QAction

from pet.ui import Overlay, Proposal
from .macos import capture_foreground, foreground_target


class CodingOverlay(Overlay):
    def _on_pet_click(self):
        if not self.paused:
            self.ask_requested.emit()


class InputCompanion(QObject):
    def __init__(self, service, panel, overlay=None):
        super().__init__()
        self.service = service
        self.panel = panel
        panel.companion_mode = True
        panel.hide()
        self.overlay = overlay or CodingOverlay()
        self._action = None
        self._proposal_id = None
        self._state = None
        self._typing = False
        self.pipeline = None
        self.timer = QTimer(self, interval=250, timeout=self._recording_tick)
        self._last_window = foreground_target()
        self.window_timer = QTimer(self, interval=250, timeout=self._remember_window)
        self.window_timer.start()
        if service._capture is capture_foreground:
            service._capture = self._capture_window
        self.overlay.ask_requested.connect(self.trigger)
        self.overlay.approved.connect(self._approved)
        self.overlay.cancelled.connect(self._cancelled)
        self.overlay.pause_toggled.connect(self._paused)
        self.overlay.text_submitted.connect(self._typed)
        self.overlay.input_cancelled.connect(service.cancel)
        service.changed.connect(self.refresh)
        service.error.connect(self._error)
        service.hotkey_pressed.disconnect(service.trigger)
        service.hotkey_pressed.connect(self.trigger)
        if self.overlay._tray:
            menu = self.overlay._menu
            menu.insertAction(menu.actions()[1], QAction("Type a task", menu, triggered=self.type_task))
            menu.insertAction(menu.actions()[2], QAction("Review context", menu, triggered=self.show_review))
            menu.insertAction(menu.actions()[3], QAction("Choose project…", menu, triggered=panel._project))
            menu.insertAction(menu.actions()[4], QAction("Cancel collection", menu, triggered=service.cancel))
        self.overlay.place_default()
        self.overlay.show()
        self.refresh()

    def bind_pipeline(self, pipeline):
        """Keep one Pette: collection owns inputs, the existing flow owns approved outcomes."""
        self.pipeline = pipeline
        self.panel.connected_flow = True
        self.panel.destination_configured = pipeline._configured
        self.panel.delivery_controls.show()
        self.panel._destination_changed()
        self.panel.handoff_note.show()
        self.service.packet_ready.connect(self._submit_context)
        self.service.progress.connect(self._collection_state)
        self.overlay.investigate_requested.connect(self.show_review)

    def _collection_state(self, state):
        if state == "idle":
            self.pipeline.cancel_context()

    def _submit_context(self, packet):
        self.panel.hide()
        self._proposal_id = self._action = None
        self.pipeline.submit_context(packet)

    def _can_collect(self):
        if self.pipeline is None or self.service.state != "ready":
            return True
        if self.pipeline.busy or self.pipeline.pending:
            self.overlay.reveal()
            return False
        self.service.cancel()
        return True

    def _propose(self, title, body, label, action, animation="speaking"):
        proposal = Proposal(title, body, label)
        self._proposal_id, self._action = proposal.id, action
        self.overlay.propose(proposal)
        self.overlay.set_state(animation)

    def _remember_window(self):
        target = foreground_target()
        if target is not None:
            self._last_window = target

    def _capture_window(self):
        return capture_foreground(self._last_window)

    def trigger(self):
        if self.overlay.paused or not self._can_collect():
            return
        if self.service.state in {"idle", "opening_microphone", "recording"}:
            if self.service.state == "idle":
                self._remember_window()
            self._typing = False
            self.service.trigger()
        elif self.service.state in {"editing", "reviewing", "ready"}:
            self.show_review()

    def type_task(self):
        if self.overlay.paused or not self._can_collect():
            return
        if self.service.state == "idle":
            self._remember_window()
            self._typing = True
            self.service.start(typed=True)
        elif self.service.state in {"editing", "reviewing"}:
            self.show_review()

    def _typed(self, task):
        if self.service.state not in {"editing", "reviewing"}:
            return
        self._typing = False
        self.service.task = self.service.transcript = task.strip()
        self.panel.task.setPlainText(task.strip())
        self._state = None
        self.refresh()
        self.show_review()

    def show_review(self):
        if self.overlay.paused:
            return
        self.panel.refresh()
        screen = self.overlay.screen().availableGeometry()
        pet = self.overlay.frameGeometry()
        x = pet.left() - self.panel.width() - 16
        if x < screen.left():
            x = min(pet.right() + 16, screen.right() - self.panel.width() + 1)
        y = max(screen.top(), min(pet.bottom() - self.panel.height(), screen.bottom() - self.panel.height() + 1))
        self.panel.move(x, y)
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def _approved(self, proposal_id):
        if proposal_id != self._proposal_id or self.overlay.paused:
            return
        action, self._action = self._action, None
        self._proposal_id = None
        if action is not None:
            action()

    def _cancelled(self, proposal_id):
        if proposal_id == self._proposal_id and not self.overlay.paused:
            self.service.cancel()

    def _paused(self, paused):
        self.panel.hide()
        if paused:
            self._typing = False
            self.service.cancel()
        else:
            self._state = None
            self.refresh()

    def refresh(self):
        state = self.service.state
        if self.overlay.paused:
            self.timer.stop()
            return
        if state == self._state:
            return
        self._state = state
        self.timer.stop()
        self.overlay.bubble.title.setStyleSheet("")
        self._proposal_id = self._action = None
        if state == "capturing":
            self.panel.hide()
            self.overlay.hide()
            return
        self.overlay.show()
        if state == "idle":
            self.panel.hide()
            self._propose("Hey, I'm Pette", "Tell me what you're building.\nClick me or press Ctrl+Shift+Space to start.", "Record a task", self.trigger, "idle")
            self.overlay.bubble.cancel.hide()
        else:
            self.overlay.bubble.cancel.show()
        if state == "opening_microphone":
            self._propose("Getting ready", "Opening your microphone…\nI'll show a red recording light when I'm listening.", "Stop", self.service.stop_recording, "listening")
        elif state == "recording":
            self._propose("● RECORDING", "", "Stop & transcribe", self.service.stop_recording, "listening")
            self.overlay.bubble.title.setStyleSheet("color: #B52A3B; font-weight: bold;")
            self._recording_tick()
            self.timer.start()
        elif state in {"stopping", "transcribing", "selecting"}:
            title, body = {
                "stopping": ("Wrapping up", "Stopping your microphone…\nThis should take at most a few seconds."),
                "transcribing": ("I heard you", "Turning your speech into text locally…"),
                "selecting": ("Finding your context", "Reading relevant saved files for your task…"),
            }[state]
            self.overlay.say(body, title, seconds=None)
            self.overlay.set_state("thinking")
        elif state in {"editing", "reviewing"}:
            if self._typing:
                self.overlay.ask_text("What would you like to build or change?")
                self.overlay.bubble.input.setPlaceholderText("e.g. Explain how pet/inputs/macos.py works")
            else:
                count = len(self.service.files)
                body = self.service.task[:180] or "You can type your task and choose its context."
                if count:
                    body += f"\n\nI found {count} file excerpt{'s' if count != 1 else ''}. Review what I'll include."
                if self.service.snapshot.warning:
                    body += "\n\nScreenshot unavailable. Open the review card for the capture error."
                self._propose("Let's review" if count else "Your task", body, "Review context", self.show_review)
        elif state == "ready":
            if self.pipeline is not None:
                self.panel.hide()
                self.overlay.say("Sending your reviewed context to the agent…", title="Working on it", seconds=None)
                self.overlay.set_state("thinking")
            else:
                self._propose("Context ready", "Your approved task and context are ready.\nOpen the review card to export JSON.", "Export JSON", self.panel._export)

    def _recording_tick(self):
        if self.service.state != "recording":
            return
        started = self.service.recording_started_at
        elapsed = min(60, int(time.monotonic() - started)) if started is not None else 0
        window_status = "Window captured." if self.service.snapshot.png else "Screenshot unavailable · see review."
        self.overlay.update_proposal_body(self._proposal_id,
            f"Listening · {elapsed // 60:02d}:{elapsed % 60:02d} / 01:00\nSpeak now. Press the hotkey again or click Stop.\n\n{window_status}")

    def _error(self, text):
        self._typing = False
        self._propose("A little help?", text, "Review / type task", self.show_review)

    def close(self):
        if self.pipeline is not None:
            self.pipeline.cancel_context()
        self.timer.stop()
        self.window_timer.stop()
        self.overlay._hide_timer.stop()
        self.overlay.pet._timer.stop()
        if self.overlay._tray:
            self.overlay._tray.hide()
        self.overlay.hide()
        self.panel.hide()
