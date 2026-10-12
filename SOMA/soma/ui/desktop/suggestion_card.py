"""Suggestion card: text, reason, evidence, confidence and Accept / Reject / Edit.

Accepting a suggestion only records feedback. SOMA never clicks, types, opens
or runs anything on the user's behalf.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from soma.types import FeedbackOutcome, Suggestion

NEVER_EXECUTES = "SOMA only suggests; it never clicks or types for you."


class SuggestionCard(QFrame):
    """Shows one :class:`Suggestion` and emits ``feedback(id, outcome, edited_text)``."""

    feedback = Signal(str, object, object)   # suggestion_id, FeedbackOutcome, str | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._suggestion: Suggestion | None = None
        self._answered = False
        self._stale = False

        self.title = QLabel("No suggestion yet")
        self.title.setObjectName("heading")
        self.title.setWordWrap(True)
        self.reason = QLabel("")
        self.reason.setObjectName("muted")
        self.reason.setWordWrap(True)
        self.evidence = QLabel("")
        self.evidence.setWordWrap(True)
        self.evidence.setTextFormat(Qt.TextFormat.PlainText)
        self.confidence = QLabel("")
        self.confidence.setObjectName("muted")
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.note = QLabel(NEVER_EXECUTES)
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)

        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setObjectName("primary")
        self.reject_btn = QPushButton("Reject")
        self.edit_btn = QPushButton("Edit")
        self.editor = QLineEdit()
        self.editor.setPlaceholderText("Rewrite the suggestion, then press Enter")
        self.editor.hide()
        self.save_edit_btn = QPushButton("Save edit")
        self.save_edit_btn.hide()

        buttons = QHBoxLayout()
        for b in (self.accept_btn, self.reject_btn, self.edit_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.editor, 1)
        edit_row.addWidget(self.save_edit_btn)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        for w in (self.title, self.reason, self.evidence, self.confidence):
            lay.addWidget(w)
        lay.addLayout(buttons)
        lay.addLayout(edit_row)
        lay.addWidget(self.status)
        lay.addWidget(self.note)

        self.accept_btn.clicked.connect(lambda: self._send(FeedbackOutcome.ACCEPTED))
        self.reject_btn.clicked.connect(lambda: self._send(FeedbackOutcome.REJECTED))
        self.edit_btn.clicked.connect(self._open_editor)
        self.save_edit_btn.clicked.connect(self._save_edit)
        self.editor.returnPressed.connect(self._save_edit)
        self._set_buttons(False)

    # -- state
    @property
    def suggestion(self) -> Suggestion | None:
        return self._suggestion

    @property
    def active(self) -> bool:
        """True while a non-silent, unanswered, non-stale suggestion is shown."""
        return (self._suggestion is not None and not self._suggestion.is_silent
                and not self._answered and not self._stale)

    @property
    def stale(self) -> bool:
        return self._stale

    def buttons_enabled(self) -> bool:
        return any(b.isEnabled() for b in (self.accept_btn, self.reject_btn, self.edit_btn))

    def _set_buttons(self, on: bool) -> None:
        for b in (self.accept_btn, self.reject_btn, self.edit_btn):
            b.setEnabled(on)
        if not on:
            self.editor.hide()
            self.save_edit_btn.hide()

    def _set_stale_prop(self, stale: bool) -> None:
        self.setProperty("stale", "true" if stale else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    # -- rendering
    def show_suggestion(self, s: Suggestion) -> None:
        self._suggestion = s
        self._answered = False
        self._stale = False
        self._set_stale_prop(False)
        self.status.setText("")
        self.editor.clear()
        if s.is_silent:
            self.title.setText(f"No suggestion ({s.reason or 'staying quiet'})")
            self.reason.setText("")
            self.evidence.setText("")
            self.confidence.setText("")
            self._set_buttons(False)
            return
        c = s.candidate
        self.title.setText(c.text)
        self.reason.setText(f"Why: {s.reason}" if s.reason else "")
        ev = [e for e in c.evidence if e][:4]
        self.evidence.setText("Evidence:\n" + "\n".join(f"- {e}" for e in ev) if ev else "")
        self.confidence.setText(f"Confidence {s.confidence:.0%} · {c.action.value}"
                                f" · screen v{s.screen_version}")
        self._set_buttons(True)

    def mark_stale(self) -> None:
        """The screen changed: grey out and disable feedback for this suggestion."""
        if self._suggestion is None or self._suggestion.is_silent or self._answered:
            return
        self._stale = True
        self._set_stale_prop(True)
        self._set_buttons(False)
        self.status.setText("Screen changed; this suggestion no longer applies.")

    def clear(self) -> None:
        self._suggestion = None
        self._answered = False
        self._stale = False
        self._set_stale_prop(False)
        self.title.setText("No suggestion yet")
        for w in (self.reason, self.evidence, self.confidence, self.status):
            w.setText("")
        self._set_buttons(False)

    # -- feedback
    def _open_editor(self) -> None:
        if not self.active:
            return
        self.editor.setText(self._suggestion.candidate.text)
        self.editor.show()
        self.save_edit_btn.show()
        self.editor.setFocus()

    def _save_edit(self) -> None:
        text = self.editor.text().strip()
        if not text:
            return
        self._send(FeedbackOutcome.EDITED, text)

    def _send(self, outcome: FeedbackOutcome, edited: str | None = None) -> None:
        if not self.active:
            return
        self._answered = True
        self._set_buttons(False)
        self.status.setText({FeedbackOutcome.ACCEPTED: "Accepted. Nothing was executed.",
                             FeedbackOutcome.REJECTED: "Rejected. Thanks, SOMA will learn from this.",
                             FeedbackOutcome.EDITED: "Edit saved as feedback."}.get(outcome, ""))
        self.feedback.emit(self._suggestion.id, outcome, edited)
