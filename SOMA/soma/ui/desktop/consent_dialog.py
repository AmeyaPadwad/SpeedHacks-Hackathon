"""Consent-first dialog: one independent, plain-language checkbox per scope.

Everything starts unchecked on a fresh install. Nothing is captured until the
user ticks "Screen capture" and saves. Remote inference is not offered in
v0.1. Folder indexing only ever covers folders the user picks here, and a
whole home directory or drive root is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from soma.consent import ConsentError, ConsentManager, Scope

SCOPE_TEXT: dict[Scope, tuple[str, str]] = {
    Scope.SCREEN_CAPTURE: (
        "Screen capture",
        "Take screenshots of the active window about once a second to read the text on it. "
        "Screenshots stay in memory and are never saved. Password managers, private browsing, "
        "banking and sign-in windows are always skipped."),
    Scope.APP_MONITORING: (
        "App monitoring",
        "Note which app is in front and its window title (secrets masked) so suggestions fit "
        "what you are doing. No keystrokes are ever recorded."),
    Scope.FOLDER_INDEXING: (
        "Folder indexing",
        "Index only the folders you choose below so SOMA can point you to related notes or "
        "files. Files are read locally and never changed, moved or uploaded."),
    Scope.FEEDBACK_STORAGE: (
        "Feedback storage",
        "Remember when you accept, reject or edit a suggestion, on this computer only. "
        "You can view, export and delete it from the Memory tab."),
    Scope.MODEL_TRAINING: (
        "Model training",
        "Allow SOMA to use your stored feedback to improve its local suggestion model. "
        "Updates are checked against a baseline and rolled back if they get worse."),
    Scope.REMOTE_INFERENCE: (
        "Remote inference",
        "Send screen content to a remote model. Not available in v0.1: everything runs locally."),
}

REMOTE_NOTE = "not available in v0.1"
HOME_REFUSAL = ("SOMA won't index a whole home folder or drive root. "
                "Please choose a smaller folder, such as a single project or notes folder.")

FolderPicker = Callable[[QWidget], str]


def default_folder_picker(parent: QWidget) -> str:
    return QFileDialog.getExistingDirectory(parent, "Choose a folder for SOMA to index")


class ConsentDialog(QDialog):
    """Edits a :class:`ConsentManager`. Grants apply only when the user saves."""

    def __init__(self, consent: ConsentManager, pick_folder: FolderPicker | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SOMA permissions")
        self.setMinimumWidth(560)
        self.consent = consent
        self._pick = pick_folder or default_folder_picker
        self.chosen_folders: list[str] = []
        self.checkboxes: dict[Scope, QCheckBox] = {}

        lay = QVBoxLayout(self)
        intro = QLabel("SOMA is off until you choose what it may do. Each switch is independent "
                       "and you can change it any time. SOMA only suggests; it never clicks, "
                       "types, or runs anything for you.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        for scope in Scope:
            title, text = SCOPE_TEXT[scope]
            box = QCheckBox(title)
            box.setChecked(scope in self._granted())
            expl = QLabel(text)
            expl.setObjectName("muted")
            expl.setWordWrap(True)
            expl.setContentsMargins(24, 0, 0, 6)
            if scope is Scope.REMOTE_INFERENCE:
                box.setChecked(False)
                box.setEnabled(False)
                box.setText(f"{title} ({REMOTE_NOTE})")
            self.checkboxes[scope] = box
            lay.addWidget(box)
            lay.addWidget(expl)
            if scope is Scope.FOLDER_INDEXING:
                lay.addWidget(self._folder_section())

        self.buttons = QDialogButtonBox()
        self.save_btn = self.buttons.addButton("Save choices", QDialogButtonBox.ButtonRole.AcceptRole)
        self.save_btn.setObjectName("primary")
        self.buttons.addButton("Keep SOMA off", QDialogButtonBox.ButtonRole.RejectRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self.checkboxes[Scope.FOLDER_INDEXING].toggled.connect(self._sync_folder_button)
        self._sync_folder_button()

    def _granted(self) -> set[Scope]:
        """Current grants, including screen capture while paused (allowed() hides it)."""
        state = getattr(self.consent, "_state", None)
        if state is not None and hasattr(state, "granted"):
            return set(state.granted)
        return {s for s in Scope if self.consent.allowed(s)}

    def _folder_section(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(24, 0, 0, 6)
        row = QHBoxLayout()
        self.folder_btn = QPushButton("Choose folder…")
        self.folder_btn.clicked.connect(self.choose_folder)
        row.addWidget(self.folder_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.folder_list = QListWidget()
        self.folder_list.setMaximumHeight(70)
        for f in self.consent.allowed_folders:
            self.folder_list.addItem(f)
        v.addWidget(self.folder_list)
        self.folder_msg = QLabel("")
        self.folder_msg.setWordWrap(True)
        self.folder_msg.setObjectName("muted")
        v.addWidget(self.folder_msg)
        return w

    def _sync_folder_button(self) -> None:
        self.folder_btn.setEnabled(self.checkboxes[Scope.FOLDER_INDEXING].isChecked())

    # -- actions
    def choose_folder(self) -> str | None:
        """Pick a folder; refuse home/drive roots with a message. Returns the added path."""
        path = self._pick(self)
        if not path:
            return None
        try:
            self.consent.add_folder(path)
        except ConsentError:
            self.folder_msg.setText(HOME_REFUSAL)
            return None
        resolved = str(Path(path).expanduser().resolve())
        if resolved not in self.chosen_folders:
            self.chosen_folders.append(resolved)
        if not self.folder_list.findItems(resolved, self._match_flag()):
            self.folder_list.addItem(resolved)
        self.folder_msg.setText(f"Added. SOMA will index {Path(resolved).name} after you save.")
        return resolved

    @staticmethod
    def _match_flag():
        from PySide6.QtCore import Qt
        return Qt.MatchFlag.MatchExactly

    def selected(self) -> set[Scope]:
        return {s for s, b in self.checkboxes.items() if b.isChecked() and b.isEnabled()}

    def apply(self) -> None:
        """Write the ticked scopes into the consent manager (unticked are revoked)."""
        chosen = self.selected()
        for scope in Scope:
            if scope in chosen:
                self.consent.grant(scope)
            else:
                self.consent.revoke(scope)
        if Scope.FOLDER_INDEXING not in chosen:
            self.chosen_folders.clear()

    def accept(self) -> None:
        self.apply()
        super().accept()

    @property
    def capture_granted(self) -> bool:
        return Scope.SCREEN_CAPTURE in self.selected()
