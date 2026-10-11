from types import SimpleNamespace
import threading
import time
import pytest

from pet.inputs.audio import InputError
from pet.inputs.macos import WindowCapture
from pet.inputs.packet import WindowContext
from pet.inputs.packet import FileExcerpt
from pet.inputs.selection import SelectionError, SelectionResult
from pet.inputs.service import CollectionService
from pet.inputs.ui import InputPanel


def service(tmp_path, **kwargs):
    return CollectionService(tmp_path, capture=lambda: WindowCapture(WindowContext("Code", "auth.py", 1),
        b"\x89PNG\r\n\x1a\nfixture"), spawn=lambda fn: fn(), **kwargs)


def test_no_packet_before_approval_and_approval_is_single_use(qapp, tmp_path):
    packets = []
    s = service(tmp_path, on_packet_ready=packets.append)
    s.start(typed=True)
    assert s.state == "editing" and not packets
    assert s.approve(" ") is None
    packet = s.approve("Fix login", include_screenshot=False)
    assert packets == [packet] and packet.screenshot_png is None
    assert s.approve("Fix login") is None
    s.cancel()
    assert s.state == "idle" and s.packet is None and s.snapshot.png is None


def test_hotkey_toggle_and_cancel_drop_queued_capture(qapp, tmp_path):
    jobs = []
    s = CollectionService(tmp_path, capture=lambda: WindowCapture(), spawn=jobs.append)
    s.trigger()
    s.trigger()
    assert len(jobs) == 1
    s.cancel()
    jobs.pop()()
    assert s.state == "idle" and s.snapshot.png is None


def test_capture_happens_before_recording_and_local_transcription(qapp, tmp_path):
    order = []
    jobs = []
    s = CollectionService(tmp_path,
        capture=lambda: (order.append("capture") or WindowCapture()),
        recorder=SimpleNamespace(record=lambda stop, cancel, **kwargs: (order.append("record") or kwargs["on_started"]() or b"wav")),
        transcriber=SimpleNamespace(transcribe=lambda wav, cancel: (order.append("transcribe") or "fix login")),
        spawn=jobs.append)
    s.trigger()
    jobs.pop(0)()
    assert s.state == "opening_microphone"
    s.trigger()
    assert s.stop.is_set() and s.state == "stopping"
    jobs.pop(0)()
    jobs.pop(0)()
    assert order == ["capture", "record", "transcribe"]
    assert s.task == s.transcript == "fix login" and s.state == "editing"


def test_mic_failure_offers_typed_input(qapp, tmp_path):
    def fail(*args, **kwargs):
        raise InputError("Microphone unavailable")
    s = service(tmp_path, recorder=SimpleNamespace(record=fail))
    s.start()
    assert s.state == "editing" and "Microphone unavailable" in s.warnings
    assert s.approve("typed task").task == "typed task"


def test_cloud_selection_requires_opt_in_and_cancel_drops_late_results(qapp, tmp_path):
    jobs, calls, packets = [], [], []
    s = service(tmp_path, selector=SimpleNamespace(select=lambda *args: (calls.append(args) or SelectionResult())))
    s.packet_ready.connect(packets.append)
    s.start(typed=True)
    s.select_files("fix")
    assert not calls and s.state == "editing"
    s._spawn = jobs.append
    s.select_files("fix", cloud_allowed=True)
    s.cancel()
    jobs.pop()()
    assert s.state == "idle" and not packets


def test_manual_files_and_exclusion_are_reflected_in_packet(qapp, tmp_path):
    (tmp_path / "auth.py").write_text("login = False")
    s = service(tmp_path)
    s.start(typed=True)
    s.manual_files(["auth.py"], "fix login")
    assert s.state == "reviewing" and s.files[0].path == "auth.py"
    packet = s.approve("fix login", included_files=[])
    assert not packet.files


def test_voice_clarification_preserves_task_and_original_transcript(qapp, tmp_path):
    s = service(tmp_path, recorder=SimpleNamespace(record=lambda *args, **kwargs: b"wav"),
                transcriber=SimpleNamespace(transcribe=lambda *args: "look in src/auth.py"))
    s.start(typed=True)
    s.task = "Fix login"
    s.clarify_by_voice()
    assert s.task == "Fix login" and "src/auth.py" in s.hints
    assert "Clarification:" in s.transcript


def test_review_panel_does_not_emit_until_button_click(qapp, tmp_path):
    packets = []
    s = service(tmp_path, on_packet_ready=packets.append)
    panel = InputPanel(s)
    s.start(typed=True)
    panel.task.setPlainText("Fix login")
    panel.include_image.setChecked(False)
    assert panel.use.isEnabled() and not packets
    panel.use.click()
    assert len(packets) == 1 and packets[0].task == "Fix login"
    assert packets[0].screenshot_png is None
    assert not panel.use.isEnabled() and panel.export.isEnabled()
    s.cancel()
    panel.hide()


def test_hotkey_recording_indicator_and_stop_transcribe_across_real_threads(qapp, tmp_path):
    ready = threading.Event()

    def record(stop, cancel, *, on_started):
        ready.wait(2)
        on_started()
        stop.wait(2)
        return b"wav"

    s = CollectionService(tmp_path, capture=lambda: WindowCapture(),
        recorder=SimpleNamespace(record=record),
        transcriber=SimpleNamespace(transcribe=lambda *args: "Fix login"))
    panel = InputPanel(s)
    states = []
    s.progress.connect(states.append)

    def wait_for(state):
        deadline = time.monotonic() + 3
        while s.state != state and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert s.state == state

    try:
        s.hotkey_pressed.emit()
        wait_for("opening_microphone")
        assert "Opening microphone" in panel.status.text()
        assert not panel.recording_timer.isActive()
        ready.set()
        wait_for("recording")
        assert "● RECORDING — 00:00" in panel.status.text()
        assert "Speak now" in panel.status.text() and panel.recording_timer.isActive()
        assert panel.stop.isEnabled()
        s.recording_started_at -= 3
        panel._recording_status()
        assert "00:03" in panel.status.text()
        s.hotkey_pressed.emit()
        assert s.state == "stopping" and not panel.recording_timer.isActive()
        wait_for("editing")
        assert "transcribing" in states and s.task == "Fix login"
        assert panel.task.toPlainText() == "Fix login"
    finally:
        ready.set()
        s.close()
        panel.hide()


def test_late_microphone_ready_cannot_override_stop_or_cancel(qapp, tmp_path):
    s = CollectionService(tmp_path, capture=lambda: WindowCapture(), spawn=lambda fn: None)
    s.start()
    s._captured(WindowCapture(), False)
    generation = s.generation
    s.stop_recording()
    s._receive(generation, s._recording_started, None)
    assert s.state == "stopping"
    s.cancel()
    s._receive(generation, s._recording_started, None)
    assert s.state == "idle" and s.recording_started_at is None


def test_enabling_ai_starts_selection_and_task_changes_require_review(qapp, tmp_path):
    calls = []
    excerpt = FileExcerpt("macos.py", 1, 1, "def capture(): pass", "Explains capture")
    selector = SimpleNamespace(provider="groq", provider_name="Groq", model="openai/gpt-oss-20b", select=lambda *args: (calls.append(args) or SelectionResult((excerpt,))))
    s = service(tmp_path, selector=selector)
    panel = InputPanel(s)
    s.start(typed=True)
    panel.task.setPlainText("Describe mac-os.py")
    assert "Groq" in panel.cloud.text() and "Free plan" in panel.cloud_note.text()
    panel.cloud.setChecked(True)
    assert len(calls) == 1 and s.state == "reviewing"
    assert panel.files.count() == 1 and s.file_selection_status == "selected"
    panel.task.setPlainText("Describe the hotkey listener")
    panel.use.click()
    assert len(calls) == 2 and s.packet is None  # Selection results need review.
    panel.use.click()
    assert s.packet.files == (excerpt,)
    assert s.packet.file_selection_status == "selected" and s.packet.file_selection_model == "openai/gpt-oss-20b"
    assert s.packet.file_selection_provider == "groq"
    s.cancel()
    panel.hide()


@pytest.mark.parametrize("result, status", [
    (SelectionResult(question="Which exact macos.py path do you mean?"), "needs_clarification"),
    (SelectionResult(), "empty"),
])
def test_empty_selection_exports_include_status_and_explanation(qapp, tmp_path, result, status):
    s = service(tmp_path, selector=SimpleNamespace(select=lambda *args: result))
    s.start(typed=True)
    s.select_files("Describe mac-os.py", cloud_allowed=True)
    packet = s.approve("Describe mac-os.py")
    assert packet.file_selection_status == status and not packet.files and packet.warnings
    if result.question:
        assert result.question in packet.warnings


def test_export_distinguishes_skipped_selection_from_excluded_files(qapp, tmp_path):
    s = service(tmp_path)
    s.start(typed=True)
    packet = s.approve("Explain this code")
    assert packet.file_selection_status == "not_requested"
    assert any("not requested" in warning for warning in packet.warnings)
    s.cancel()
    s.start(typed=True)
    (tmp_path / "macos.py").write_text("def capture(): pass")
    s.manual_files(["macos.py"], "Explain this code")
    packet = s.approve("Explain this code", included_files=[])
    assert packet.file_selection_status == "excluded" and not packet.files
    assert any("excluded" in warning for warning in packet.warnings)


def test_api_failure_is_visible_in_export(qapp, tmp_path):
    def fail(*args):
        raise SelectionError("AI selection failed. Check the API key/connection or choose files manually.")

    s = service(tmp_path, selector=SimpleNamespace(model="gpt-4o", select=fail))
    s.start(typed=True)
    s.select_files("Describe macos.py", cloud_allowed=True)
    packet = s.approve("Describe macos.py")
    assert packet.file_selection_status == "failed" and packet.file_selection_model == "gpt-4o"
    assert any("AI selection failed" in warning for warning in packet.warnings)


def test_screenshot_error_and_settings_are_visible_without_png(qapp, tmp_path, monkeypatch):
    warning = "macOS has not granted Screen Recording to this process."
    s = CollectionService(tmp_path, capture=lambda: WindowCapture(warning=warning), spawn=lambda fn: fn())
    panel = InputPanel(s)
    opened = []
    monkeypatch.setattr("pet.inputs.ui.QDesktopServices.openUrl", lambda url: opened.append(url.toString()))
    s.start(typed=True)
    panel.tabs.setCurrentIndex(1)
    assert panel.image.text() == warning and panel.screen_settings.isVisible()
    assert not panel.preview_image.isEnabled() and not panel.include_image.isEnabled()
    panel.screen_settings.click()
    assert opened == ["x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"]
    # PNG identity stays None; a new warning must still replace the displayed message.
    s.snapshot = WindowCapture(warning="No capturable foreground window.")
    panel.refresh()
    assert panel.image.text() == s.snapshot.warning and not panel.screen_settings.isVisible()
    s.close()
    panel.hide()
