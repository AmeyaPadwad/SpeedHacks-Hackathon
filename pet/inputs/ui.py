"""Review overlay for the coding-input mode, with explicit packet approval/export."""
from pathlib import Path
from types import SimpleNamespace
import time

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget)

from .macos import Hotkeys
from .service import CollectionService
from pet.sprites import SpriteSheet, asset_path


class InputPanel(QWidget):
    def __init__(self, service: CollectionService):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.Tool)
        self.service = service
        self.companion_mode = False
        self.setObjectName("inputPanel")
        self.setWindowTitle("Pette · Review your context")
        self.resize(600, min(760, self.screen().availableGeometry().height() - 48))
        self.setMinimumWidth(520)
        self.setStyleSheet("""
            QWidget#inputPanel { background: #FCF5E8; }
            QWidget#reviewBody { background: #FCF5E8; }
            QLabel, QCheckBox { color: #243B32; font-size: 13px; background: transparent; }
            QLabel#brand { font-size: 25px; font-weight: bold; }
            QLabel#subtitle, QLabel#projectLabel { color: #64756C; }
            QLabel#status { background: #E3F5EB; border-radius: 10px; padding: 10px; color: #286148; }
            QLabel#section { font-size: 14px; font-weight: bold; }
            QPlainTextEdit, QListWidget { background: #FFFDFA; color: #243B32;
                border: 1px solid #D5DFD5; border-radius: 9px; padding: 8px; font-size: 13px; }
            QPlainTextEdit:focus, QListWidget:focus { border: 1px solid #3E9E78; }
            QListWidget::item { padding: 5px; border-radius: 5px; }
            QListWidget::item:selected { background: #DDF4E7; color: #1C4937; }
            QPushButton { background: #FFFDFA; color: #243B32; border: 1px solid #CBD8CD;
                border-radius: 8px; padding: 8px 12px; font-size: 12px; }
            QPushButton:hover { background: #E3F5EB; border-color: #5AAB87; }
            QPushButton:disabled { color: #9AA69D; background: #EFEDE5; border-color: #E0E2D8; }
            QPushButton#primary { background: #7CF2C4; border-color: #53C89B; font-weight: bold; }
            QPushButton#primary:hover { background: #A2F6D6; }
            QPushButton#primary:disabled { background: #DFE8DE; border-color: #DFE8DE; }
            QCheckBox { spacing: 7px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px;
                border: 1px solid #A8BCAF; background: #FFFDFA; }
            QCheckBox::indicator:checked { image: url(CHECKMARK_URL); border-color: #53C89B; }
            QTabWidget::pane { border: 0; background: transparent; }
            QTabBar::tab { color: #64756C; padding: 9px 16px; background: #EEECE2;
                border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px; }
            QTabBar::tab:selected { background: #DDF4E7; color: #245A42; }
        """.replace("CHECKMARK_URL", asset_path("context-check.svg").as_posix()))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget(objectName="reviewBody")
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(10)
        header = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(SpriteSheet().frames[0].scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header.addWidget(icon)
        names = QVBoxLayout()
        names.setSpacing(2)
        names.addWidget(QLabel("Pette", objectName="brand"))
        names.addWidget(QLabel("A little context. A lot less switching.", objectName="subtitle"))
        header.addLayout(names)
        header.addStretch()
        layout.addLayout(header)
        self.status = QLabel("Ready. Switch to your IDE and press Ctrl+Shift+Space.", objectName="status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(250)
        self.recording_timer.timeout.connect(self._recording_status)
        self.project = QLabel(objectName="projectLabel")
        self.project.setWordWrap(True)
        project_row = QHBoxLayout()
        project_row.addWidget(self.project, 1)
        self.pick_project = QPushButton("Choose project")
        self.pick_project.clicked.connect(self._project)
        project_row.addWidget(self.pick_project)
        layout.addLayout(project_row)
        row = QHBoxLayout()
        self.record = QPushButton("Start recording")
        self.typed = QPushButton("Type task")
        self.stop = QPushButton("Stop")
        row.addWidget(self.record)
        row.addWidget(self.typed)
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.capture_controls = (self.record, self.typed, self.stop)
        self.record.clicked.connect(lambda: service.start())
        self.typed.clicked.connect(lambda: service.start(typed=True))
        self.stop.clicked.connect(service.stop_recording)
        layout.addWidget(QLabel("What we're working on", objectName="section"))
        self.task = QPlainTextEdit()
        self.task.setFixedHeight(82)
        self.task.setPlaceholderText("Tell Pette what you'd like to build, fix, or explain…")
        layout.addWidget(self.task)
        self.hints = QPlainTextEdit()
        self.hints.setPlaceholderText("Optional file/folder hints, e.g. src/auth.py or tests/")
        self.hints.setFixedHeight(48)
        layout.addWidget(self.hints)
        self.clarify = QPushButton("Tell Pette more")
        self.clarify.clicked.connect(self._clarify)
        layout.addWidget(self.clarify)
        provider_name = getattr(service.selector, "provider_name", "AI")
        self.cloud = QCheckBox(f"Let {provider_name} find the relevant files")
        self.cloud.toggled.connect(self._cloud_toggled)
        layout.addWidget(self.cloud)
        pricing_note = "Groq's Free plan has usage limits." if getattr(service.selector, "provider", "") == "groq" else "Hosted AI calls use your provider account."
        self.cloud_note = QLabel(f"Sends project paths and short file excerpts to {provider_name}. {pricing_note} Audio and the screenshot stay local during file selection.", objectName="subtitle")
        self.cloud_note.setWordWrap(True)
        layout.addWidget(self.cloud_note)
        self.tabs = QTabWidget()
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        file_layout.setContentsMargins(0, 12, 0, 0)
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        image_layout.setContentsMargins(0, 12, 0, 0)
        self.tabs.addTab(file_tab, "Files · 0")
        self.tabs.addTab(image_tab, "Window snapshot")
        layout.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        self.select = QPushButton("Find relevant files")
        self.manual = QPushButton("Choose files manually")
        row.addWidget(self.select)
        row.addWidget(self.manual)
        file_layout.addLayout(row)
        self.select.clicked.connect(lambda: service.select_files(self.task.toPlainText(), self.hints.toPlainText(),
                                                                 cloud_allowed=self.cloud.isChecked()))
        self.manual.clicked.connect(self._manual)
        self.files = QListWidget()
        self.files.setMaximumHeight(90)
        self.files.currentRowChanged.connect(self._preview_file)
        file_layout.addWidget(self.files)
        self.excerpt = QPlainTextEdit()
        self.excerpt.setReadOnly(True)
        self.excerpt.setMinimumHeight(120)
        self.excerpt.setPlaceholderText("Select a file to review the exact excerpt included in your context.")
        file_layout.addWidget(self.excerpt, 1)
        self.include_image = QCheckBox("Include foreground-window screenshot")
        self.include_image.setChecked(True)
        image_layout.addWidget(self.include_image)
        self.image = QLabel("No screenshot yet")
        self.image.setWordWrap(True)
        self.image.setTextFormat(Qt.PlainText)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(120)
        image_layout.addWidget(self.image, 1)
        self.preview_image = QPushButton("View screenshot at full size")
        self.preview_image.clicked.connect(self._preview_image)
        image_layout.addWidget(self.preview_image)
        self.screen_settings = QPushButton("Open Screen Recording settings")
        self.screen_settings.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")))
        image_layout.addWidget(self.screen_settings)
        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        self.warnings.setTextFormat(Qt.PlainText)
        self.warnings.setStyleSheet("color: #99602E; font-size: 12px;")
        layout.addWidget(self.warnings)
        row = QHBoxLayout()
        self.use = QPushButton("Use context", objectName="primary")
        self.export = QPushButton("Export JSON")
        self.cancel_button = QPushButton("Cancel / New")
        self.quit_button = QPushButton("Quit")
        for button in (self.use, self.export, self.cancel_button, self.quit_button):
            row.addWidget(button)
        row.setContentsMargins(20, 12, 20, 16)
        outer.addLayout(row)
        self.use.clicked.connect(self._approve)
        self.export.clicked.connect(self._export)
        self.cancel_button.clicked.connect(service.cancel)
        self.quit_button.clicked.connect(QApplication.instance().quit)
        service.changed.connect(self.refresh)
        service.error.connect(self._error)
        self.task.textChanged.connect(self._enable_use)
        self._files_identity = None
        self._image_identity = None
        self.refresh()

    def _enable_use(self):
        self.use.setEnabled(self.service.state in {"editing", "reviewing"} and bool(self.task.toPlainText().strip()))

    def refresh(self):
        s = self.service
        labels = {"idle": "Ready. Switch to your IDE and press Ctrl+Shift+Space.",
                  "capturing": "Capturing the foreground window…",
                  "opening_microphone": "Opening microphone… waiting for audio. Allow microphone access if macOS asks.",
                  "recording": "Recording…",
                  "stopping": "Stopping microphone… (up to 5 seconds)", "transcribing": "Microphone stopped. Transcribing locally with Whisper…",
                  "editing": "Review your task, then select relevant files or continue with the screenshot.",
                  "selecting": "Gathering relevant saved project files…", "reviewing": "Review the selected excerpts and screenshot before using context.",
                  "ready": "Context approved. The packet is ready for the Darwin integration; you can export JSON."}
        self.status.setText(labels[s.state])
        if s.state == "recording":
            self.status.setStyleSheet("background-color: #fff0f0; color: #a31818; font-size: 16px; font-weight: bold; padding: 12px; border-radius: 6px;")
            self._recording_status()
            self.recording_timer.start()
        else:
            self.recording_timer.stop()
            self.status.setStyleSheet("")
        self.project.setText(f"Project · {s.project.name}")
        self.project.setToolTip(str(s.project))
        for control in self.capture_controls:
            control.setVisible(not self.companion_mode)
        editable = s.state in {"editing", "reviewing"}
        self.record.setEnabled(s.state == "idle")
        self.typed.setEnabled(s.state == "idle")
        self.stop.setEnabled(s.state in {"opening_microphone", "recording"})
        self.pick_project.setEnabled(s.state in {"idle", "editing", "reviewing"})
        for widget in (self.task, self.hints, self.select, self.manual, self.clarify, self.cloud,
                       self.files, self.include_image):
            widget.setEnabled(editable)
        self.export.setEnabled(s.state == "ready")
        if s.state == "idle":
            self.task.clear()
            self.hints.clear()
        elif s.state == "editing":
            if s.task and not self.task.toPlainText():
                self.task.setPlainText(s.task)
            if s.hints != self.hints.toPlainText():
                self.hints.setPlainText(s.hints)
        self._enable_use()
        if self._files_identity is not s.files:
            self._files_identity = s.files
            self.files.clear()
            self.excerpt.clear()
            for file in s.files:
                item = QListWidgetItem(f"{file.path} · lines {file.start_line}–{file.end_line}")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.files.addItem(item)
            if s.files:
                self.files.setCurrentRow(0)
            self.tabs.setTabText(0, f"Files · {len(s.files)}")
        if self._image_identity is not s.snapshot.png:
            self._image_identity = s.snapshot.png
            self.image.clear()
            if s.snapshot.png:
                pixmap = QPixmap()
                pixmap.loadFromData(s.snapshot.png)
                self.image.setPixmap(pixmap.scaled(540, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.image.setText(s.snapshot.warning or "No foreground-window screenshot available")
        self.include_image.setEnabled(editable and bool(s.snapshot.png))
        if not s.snapshot.png:
            self.image.setText(s.snapshot.warning or "No foreground-window screenshot available")
        self.preview_image.setEnabled(bool(s.snapshot.png))
        self.screen_settings.setVisible(not s.snapshot.png and "Screen Recording" in s.snapshot.warning)
        self.warnings.setText("\n".join(dict.fromkeys(s.warnings))[-2000:])
        if s.transcription_seconds is not None and s.state in {"editing", "reviewing", "ready"}:
            self.status.setText(self.status.text() + f" (Transcription: {s.transcription_seconds:.1f}s)")
        if not self.companion_mode and s.state not in {"idle", "capturing"}:
            self.show()
            # Avoid repeatedly stealing focus when the user edits or toggles fields.
            if s.state == "recording":
                self.raise_()

    def _recording_status(self):
        if self.service.state != "recording":
            return
        started = self.service.recording_started_at
        elapsed = min(60, int(time.monotonic() - started)) if started is not None else 0
        self.status.setText(
            f"● RECORDING — {elapsed // 60:02d}:{elapsed % 60:02d} / 01:00\n"
            "Speak now. Press Ctrl+Shift+Space again or click Stop to transcribe."
        )

    def _error(self, text):
        self.warnings.setText(text)

    def _project(self):
        value = QFileDialog.getExistingDirectory(self, "Connect project", str(self.service.project))
        if value:
            self.service.set_project(value)

    def _manual(self):
        if not self.task.toPlainText().strip():
            self._error("Describe the task first.")
            return
        values, _ = QFileDialog.getOpenFileNames(self, "Select up to eight project files", str(self.service.project))
        try:
            paths = [Path(value).relative_to(self.service.project).as_posix() for value in values]
        except ValueError:
            self._error("Choose files inside the connected project.")
            return
        if paths:
            self.service.manual_files(paths, self.task.toPlainText(), self.hints.toPlainText())

    def _clarify(self):
        self.service.task = self.task.toPlainText()
        self.service.hints = self.hints.toPlainText()
        self.service.clarify_by_voice()

    def _cloud_toggled(self, enabled):
        if enabled and self.service.state in {"editing", "reviewing"} and self.task.toPlainText().strip():
            self.service.select_files(self.task.toPlainText(), self.hints.toPlainText(), cloud_allowed=True)

    def _preview_file(self, index):
        if 0 <= index < len(self.service.files):
            self.excerpt.setPlainText(self.service.files[index].context_text())

    def _preview_image(self):
        if not self.service.snapshot.png:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Foreground window — review")
        dialog.resize(900, 650)
        layout = QVBoxLayout(dialog)
        area = QScrollArea()
        image = QLabel()
        pixmap = QPixmap()
        pixmap.loadFromData(self.service.snapshot.png)
        image.setPixmap(pixmap)
        area.setWidget(image)
        layout.addWidget(area)
        close = QPushButton("Close preview")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def _approve(self):
        task, hints = self.task.toPlainText(), self.hints.toPlainText()
        if (self.cloud.isChecked() and task.strip()
                and self.service.file_selection_input != (task.strip(), hints.strip())):
            self.service.select_files(task, hints, cloud_allowed=True)
            return  # Review the results, then click Use context again.
        included = [i for i in range(self.files.count()) if self.files.item(i).checkState() == Qt.Checked]
        self.service.approve(task, included, include_screenshot=self.include_image.isChecked())

    def _export(self):
        packet = self.service.packet
        if packet is None:
            return
        value, _ = QFileDialog.getSaveFileName(self, "Export reviewed context", "flowstate-context.json", "JSON (*.json)")
        if value:
            try:
                Path(value).write_text(packet.to_json(), encoding="utf-8")
                self.status.setText("Reviewed context exported. This file contains the selected code and screenshot.")
            except OSError:
                self._error("Could not write the export. Choose another location.")

    def closeEvent(self, event):
        self.service.cancel()
        self.hide()
        event.ignore()


def build_input_gathering(app, project: Path, *, on_packet_ready=None):
    from .companion import InputCompanion

    service = CollectionService(project, on_packet_ready=on_packet_ready)
    panel = InputPanel(service)
    companion = InputCompanion(service, panel)
    hotkeys = Hotkeys()
    ok = hotkeys.start(service.hotkey_pressed.emit, service.cancel_pressed.emit)
    if not ok:
        companion.overlay.say(f"{hotkeys.error}\nClick me to record, or use Type a task in my menu.", "Hotkey unavailable", seconds=None)
    app.aboutToQuit.connect(hotkeys.close)
    app.aboutToQuit.connect(service.close)
    app.aboutToQuit.connect(companion.close)
    return SimpleNamespace(app=app, service=service, panel=panel, companion=companion,
                           overlay=companion.overlay, hotkeys=hotkeys)


def run_input_gathering(app, project: Path, *, on_packet_ready=None):
    # Keep wrappers alive through the event loop.
    ctx = build_input_gathering(app, project, on_packet_ready=on_packet_ready)
    return ctx.app.exec()
