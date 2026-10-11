"""Qt collection service; approved packets feed the agent flow or a standalone consumer."""
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .audio import InputError, LocalWhisper, Recorder
from .macos import WindowCapture, capture_foreground
from .packet import ContextPacket, MAX_FILES
from .project import ProjectError, ProjectReader, inventory
from .selection import FileSelector, SelectionError, bounded_excerpt


class CollectionService(QObject):
    progress = Signal(str)
    changed = Signal()
    error = Signal(str)
    packet_ready = Signal(object)
    hotkey_pressed = Signal()
    cancel_pressed = Signal()
    _delivery = Signal(int, object, object)  # generation, callback, result

    def __init__(self, project: Path, *, capture=capture_foreground, recorder=None,
                 transcriber=None, selector=None, on_packet_ready=None, spawn=None):
        super().__init__()
        self.project = project.expanduser().resolve()
        self._capture = capture
        self.recorder = recorder or Recorder()
        self.transcriber = transcriber or LocalWhisper()
        self.selector = selector or FileSelector()
        self._spawn = spawn or self._start_thread
        self._threads = []
        self.generation = 0
        self.state = "idle"
        self.stop = threading.Event()
        self.cancel_event = threading.Event()
        self.snapshot = WindowCapture()
        self.transcript = ""
        self.task = ""
        self.hints = ""
        self.files = ()
        self.warnings = []
        self.catalog = None
        self.file_selection_status = "not_requested"
        self.file_selection_model = ""
        self.file_selection_provider = ""
        self.file_selection_input = None
        self.packet = None
        self.session_id = None
        self.captured_at = None
        self.transcription_seconds = None
        self.recording_started_at = None
        self._clarifying = False
        self._delivery.connect(self._receive)
        self.hotkey_pressed.connect(self.trigger)
        self.cancel_pressed.connect(self.cancel)
        if on_packet_ready is not None:
            self.packet_ready.connect(on_packet_ready)

    def _state(self, state):
        self.state = state
        self.progress.emit(state)
        self.changed.emit()

    def _start_thread(self, fn):
        self._threads = [t for t in self._threads if t.is_alive()]
        thread = threading.Thread(target=fn, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _work(self, work, done):
        generation = self.generation
        event = self.cancel_event

        def run():
            try:
                result = work()
                if event.is_set():
                    return
                self._delivery.emit(generation, done, result)
            except Exception as e:
                message = str(e) if isinstance(e, (InputError, ProjectError, SelectionError)) else "Collection failed; retry or use manual input."
                if not event.is_set():
                    self._delivery.emit(generation, self._failed, message)

        self._spawn(run)

    def _receive(self, generation, callback, result):
        if generation == self.generation and not self.cancel_event.is_set():
            callback(result)

    def trigger(self):
        if self.state in {"opening_microphone", "recording"}:
            self.stop_recording()
        elif self.state == "idle":
            self.start()

    def start(self, *, typed=False):
        if self.state != "idle":
            return
        self.generation += 1
        self.cancel_event = threading.Event()
        self.stop = threading.Event()
        self._clarifying = False
        self.transcript = self.task = self.hints = ""
        self.files, self.warnings, self.catalog, self.packet = (), [], None, None
        self._reset_file_selection()
        self.transcription_seconds = None
        from datetime import datetime, timezone
        from uuid import uuid4
        self.session_id = str(uuid4())
        self.captured_at = datetime.now(timezone.utc).isoformat()
        self._state("capturing")
        # UI must wait for this result before showing/raising its capture panel.
        self._work(self._capture, lambda value: self._captured(value, typed))

    def _captured(self, value, typed):
        self.snapshot = value
        if value.warning:
            self.warnings.append(value.warning)
        if typed:
            self._state("editing")
        else:
            self._record()

    def _record(self):
        self.stop = threading.Event()
        stop, cancel = self.stop, self.cancel_event
        generation = self.generation
        self.recording_started_at = None
        self._state("opening_microphone")

        def started():
            self._delivery.emit(generation, self._recording_started, None)

        self._work(lambda: self.recorder.record(stop, cancel, on_started=started), self._recorded)

    def _recording_started(self, _):
        if self.state == "opening_microphone":
            self.recording_started_at = time.monotonic()
            self._state("recording")

    def clarify_by_voice(self):
        if self.state not in {"editing", "reviewing"}:
            return
        self._clarifying = True
        self._record()

    def stop_recording(self):
        if self.state in {"opening_microphone", "recording"}:
            self.stop.set()
            self._state("stopping")

    def _recorded(self, wav):
        self._state("transcribing")
        cancel = self.cancel_event

        def transcribe():
            start = time.monotonic()
            text = self.transcriber.transcribe(wav, cancel)
            return text, time.monotonic() - start

        self._work(transcribe, self._heard)

    def _heard(self, result):
        text, self.transcription_seconds = result
        if self._clarifying:
            self.hints = (self.hints + "\n" + text).strip()
            self.transcript = (self.transcript + "\nClarification: " + text).strip()
        else:
            self.transcript = self.task = text
        self._clarifying = False
        self._state("editing")

    def _failed(self, message):
        if self.state == "selecting":
            self.file_selection_status = "failed"
        self.warnings.append(message)
        self._state("editing")
        self.error.emit(message)

    def set_project(self, project):
        if self.state in {"idle", "editing", "reviewing"}:
            self.project = Path(project).expanduser().resolve()
            self.catalog = None
            self.files = ()
            self.packet = None
            self._reset_file_selection()
            if self.state != "idle":
                self._state("editing")
            else:
                self.changed.emit()

    def _reset_file_selection(self):
        self.file_selection_status = "not_requested"
        self.file_selection_model = ""
        self.file_selection_provider = ""
        self.file_selection_input = None

    def select_files(self, task: str, hints="", *, cloud_allowed=False):
        if self.state not in {"editing", "reviewing"} or not task.strip():
            return
        self.task, self.hints = task.strip(), hints.strip()
        if not cloud_allowed:
            self.error.emit("Allow AI file selection, or choose files manually.")
            return
        self.files = ()
        self.file_selection_model = getattr(self.selector, "model", "")
        self.file_selection_provider = getattr(self.selector, "provider", "")
        self.file_selection_input = (self.task, self.hints)
        self._state("selecting")
        root, cancel, task, hints = self.project, self.cancel_event, self.task, self.hints

        def select():
            catalog = inventory(root)
            return catalog, self.selector.select(task, catalog, hints, cancel)

        def done(result):
            catalog, selection = result
            self.catalog = catalog
            self.files = selection.files
            self.warnings.extend(selection.warnings)
            if selection.question:
                self.file_selection_status = "needs_clarification"
                self.warnings.append(selection.question)
                self._state("editing")
                self.error.emit(selection.question)
            else:
                self.file_selection_status = "selected" if selection.files else "empty"
                if not selection.files and not selection.warnings:
                    self.warnings.append("AI file selection returned no relevant file excerpts. Retry with an exact file path or choose files manually.")
                self._state("reviewing")

        self._work(select, done)

    def manual_files(self, paths: list[str], task: str, hints=""):
        if self.state not in {"editing", "reviewing"} or not task.strip():
            return
        if len(paths) > MAX_FILES:
            self.error.emit("Choose at most eight files.")
            return
        self.task, self.hints = task.strip(), hints.strip()
        self.files = ()
        self.file_selection_model = ""
        self.file_selection_provider = ""
        self.file_selection_input = (self.task, self.hints)
        self._state("selecting")
        root = self.project

        def read():
            catalog = inventory(root)
            reader = ProjectReader(catalog)
            files = tuple(bounded_excerpt(reader.read(path)) for path in dict.fromkeys(paths))
            return catalog, files

        def done(result):
            self.catalog, self.files = result
            self.file_selection_status = "manual" if self.files else "empty"
            if self.catalog.truncated:
                self.warnings.append("Project inventory was bounded; some files were omitted.")
            self._state("reviewing")

        self._work(read, done)

    def approve(self, task: str, included_files=None, *, include_screenshot=True,
                message_platform="", message_recipient=""):
        if self.state not in {"editing", "reviewing"} or not task.strip():
            return None
        allowed = set(range(len(self.files))) if included_files is None else set(included_files)
        files = tuple(f for i, f in enumerate(self.files) if i in allowed)
        warnings = list(self.warnings)
        status = self.file_selection_status
        if not files:
            if self.files:
                status = "excluded"
                warnings.append("All selected file excerpts were excluded from this packet.")
            elif status == "not_requested":
                warnings.append("File selection was not requested; this packet contains no saved project-file context.")
        self.packet = ContextPacket(task=task.strip(), transcript=self.transcript,
            project_root=str(self.project), window=self.snapshot.window,
            files=files,
            screenshot_png=self.snapshot.png if include_screenshot else None,
            warnings=tuple(warnings), session_id=self.session_id, captured_at=self.captured_at,
            file_selection_status=status, file_selection_model=self.file_selection_model,
            file_selection_provider=self.file_selection_provider,
            message_platform=message_platform, message_recipient=message_recipient.strip()[:200])
        self._state("ready")
        self.packet_ready.emit(self.packet)
        return self.packet

    def cancel(self):
        self.generation += 1
        self.cancel_event.set()
        self.stop.set()
        self.snapshot = WindowCapture()
        self.files, self.catalog, self.packet = (), None, None
        self._reset_file_selection()
        self.transcript = self.task = self.hints = ""
        self.warnings = []
        self.recording_started_at = None
        self._state("idle")

    def close(self):
        self.cancel()
        deadline = time.monotonic() + 3
        for thread in self._threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
