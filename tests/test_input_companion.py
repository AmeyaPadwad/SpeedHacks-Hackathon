from types import SimpleNamespace

import pytest

from pet.inputs.companion import CodingOverlay, InputCompanion
from pet.inputs.macos import WindowCapture, WindowTarget
from pet.inputs.packet import FileExcerpt, WindowContext
from pet.inputs.selection import SelectionResult
from pet.inputs.service import CollectionService
from pet.inputs.ui import InputPanel


@pytest.fixture
def companion(qapp, tmp_path):
    jobs = []
    service = CollectionService(tmp_path, capture=lambda: WindowCapture(), spawn=jobs.append,
        recorder=SimpleNamespace(record=lambda stop, cancel, **kwargs: (kwargs["on_started"]() or b"wav")),
        transcriber=SimpleNamespace(transcribe=lambda *args: "Explain macos.py"))
    panel = InputPanel(service)
    pet = InputCompanion(service, panel, CodingOverlay(tray=False))
    yield pet, service, panel, jobs
    pet.close()
    service.close()


def test_pet_is_primary_ui_and_click_toggles_recording(companion):
    pet, service, panel, jobs = companion
    assert pet.overlay.isVisible() and not panel.isVisible()
    assert pet.overlay.pet.state == "idle"
    pet.overlay.pet.clicked.emit()
    assert service.state == "capturing" and not pet.overlay.isVisible()
    jobs.pop(0)()
    assert service.state == "opening_microphone" and pet.overlay.pet.state == "listening"
    jobs.pop(0)()
    # Fake capture completes immediately; transcription is deferred.
    assert service.state == "transcribing" and pet.overlay.pet.state == "thinking"
    jobs.pop(0)()
    assert service.state == "editing" and not panel.isVisible()
    assert pet.overlay.bubble.primary.text() == "Review context"
    pet.overlay.bubble.primary.click()
    assert panel.isVisible() and panel.task.toPlainText() == "Explain macos.py"


def test_pet_recording_timer_and_single_use_stop_action(companion):
    pet, service, panel, jobs = companion
    service.hotkey_pressed.emit()
    jobs.pop(0)()
    service._recording_started(None)
    assert pet.overlay.pet.state == "listening"
    assert "RECORDING" in pet.overlay.bubble.title.text()
    assert "00:00" in pet.overlay.bubble.body.text() and pet.timer.isActive()
    proposal_id = pet.overlay.bubble.proposal.id
    service.recording_started_at -= 3
    pet._recording_tick()
    assert "00:03" in pet.overlay.bubble.body.text()
    assert pet.overlay.bubble.proposal.id == proposal_id
    pet.overlay.bubble.primary.click()
    assert service.state == "stopping" and service.stop.is_set()
    assert pet.overlay.pet.state == "thinking" and not pet.timer.isActive()
    pet.overlay.approved.emit(proposal_id)
    assert service.state == "stopping" and service.packet is None


def test_hotkey_during_recording_stops_and_pause_cancels(companion):
    pet, service, panel, jobs = companion
    service.hotkey_pressed.emit()
    jobs.pop(0)()
    service._recording_started(None)
    service.hotkey_pressed.emit()
    assert service.state == "stopping"
    pet.overlay.set_paused(True)
    assert service.state == "idle" and pet.overlay.pet.state == "sleeping"
    jobs.pop(0)()
    assert service.state == "idle"
    service.hotkey_pressed.emit()
    assert not jobs
    pet.overlay.set_paused(False)
    assert pet.overlay.bubble.primary.text() == "Record a task"


def test_typed_task_from_pet_and_review_approval(companion):
    pet, service, panel, jobs = companion
    packets = []
    service.packet_ready.connect(packets.append)
    pet.type_task()
    jobs.pop(0)()
    assert pet.overlay.bubble.asking and not panel.isVisible()
    pet.overlay.bubble.input.setText("Fix login")
    pet.overlay.bubble.primary.click()
    assert panel.isVisible() and panel.task.toPlainText() == "Fix login"
    assert not packets
    panel.use.click()
    assert len(packets) == 1 and pet.overlay.bubble.title.text() == "Context ready"
    assert pet.overlay.bubble.primary.text() == "Export JSON"


def test_selection_progress_and_file_review_remain_with_pet(companion):
    pet, service, panel, jobs = companion
    excerpt = FileExcerpt("macos.py", 1, 1, "def capture(): pass")
    service.selector = SimpleNamespace(model="gpt-4o", select=lambda *args: SelectionResult((excerpt,)))
    pet.type_task()
    jobs.pop(0)()
    pet._typed("Explain mac-os.py")
    panel.cloud.setChecked(True)
    assert pet.overlay.pet.state == "thinking" and "Reading" in pet.overlay.bubble.body.text()
    jobs.pop(0)()
    assert panel.tabs.tabText(0) == "Files · 1"
    assert "1 file excerpt" in pet.overlay.bubble.body.text()
    assert service.packet is None


def test_closing_review_cancels_and_returns_to_pet(companion):
    pet, service, panel, jobs = companion
    pet.type_task()
    jobs.pop(0)()
    pet._typed("Explain capture")
    assert panel.isVisible()
    panel.close()
    assert service.state == "idle" and not panel.isVisible()
    assert pet.overlay.isVisible() and pet.overlay.bubble.primary.text() == "Record a task"


def test_last_external_window_survives_pet_taking_focus(qapp, tmp_path, monkeypatch):
    external = WindowTarget(123, WindowContext("Code", "auth.py", 7))
    current = [external]
    calls, jobs = [], []
    monkeypatch.setattr("pet.inputs.companion.foreground_target", lambda: current[0])

    def capture(previous):
        calls.append(previous)
        return WindowCapture(previous.window, b"\x89PNG\r\n\x1a\nfixture")

    monkeypatch.setattr("pet.inputs.companion.capture_foreground", capture)
    s = CollectionService(tmp_path, capture=capture, spawn=jobs.append)
    panel = InputPanel(s)
    pet = InputCompanion(s, panel, CodingOverlay(tray=False))
    try:
        current[0] = None  # Pette is now foreground.
        pet.type_task()
        jobs.pop(0)()
        assert calls == [external] and s.snapshot.window.window_id == 7
        assert s.snapshot.png
    finally:
        pet.close()
        s.close()
