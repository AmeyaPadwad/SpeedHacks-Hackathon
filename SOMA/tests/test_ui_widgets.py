"""Score chart and suggestion card widgets (offscreen)."""

import pytest
from PySide6.QtWidgets import QApplication

from soma.types import ActionType, Candidate, FeedbackOutcome, Suggestion
from soma.ui.desktop.score_chart import ScoreChart
from soma.ui.desktop.suggestion_card import NEVER_EXECUTES, SuggestionCard


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def suggestion(version=3):
    cand = Candidate(action=ActionType.EXPLAIN_ERROR, text="Explain the TypeError on line 12",
                     evidence=["TypeError: unsupported operand", "app.py:12"])
    return Suggestion(candidate=cand, confidence=0.81, screen_version=version, reason="error visible",
                      distribution={"NO_SUGGESTION": 0.19, "EXPLAIN_ERROR": 0.81})


def silent(version=3):
    return Suggestion(candidate=None, confidence=0.9, screen_version=version, reason="below threshold",
                      distribution={"NO_SUGGESTION": 0.9, "SUGGEST_NEXT_STEP": 0.1})


def test_chart_orders_no_suggestion_first_and_highlights(qapp):
    chart = ScoreChart()
    chart.set_distribution({"EXPLAIN_ERROR": 0.7, "NO_SUGGESTION": 0.2, "SUGGEST_NEXT_STEP": 0.1})
    assert chart.items[0][0] == "NO_SUGGESTION"
    assert chart.chosen == "EXPLAIN_ERROR"
    assert chart.total() == pytest.approx(1.0)
    chart.resize(500, 120)
    chart.grab()  # paints without error
    chart.set_stale(True)
    chart.grab()
    chart.clear()
    assert chart.items == [] and chart.chosen is None
    chart.grab()


def test_card_shows_suggestion_and_note(qapp):
    card = SuggestionCard()
    card.show_suggestion(suggestion())
    assert "TypeError" in card.title.text()
    assert "error visible" in card.reason.text()
    assert "app.py:12" in card.evidence.text()
    assert "81%" in card.confidence.text()
    assert card.note.text() == NEVER_EXECUTES
    assert card.buttons_enabled() and card.active


def test_silent_disables_buttons(qapp):
    card = SuggestionCard()
    card.show_suggestion(silent())
    assert card.title.text().startswith("No suggestion (below threshold")
    assert not card.buttons_enabled()


@pytest.mark.parametrize("button,outcome", [("accept_btn", FeedbackOutcome.ACCEPTED),
                                            ("reject_btn", FeedbackOutcome.REJECTED)])
def test_accept_reject_emit_once(qapp, button, outcome):
    card = SuggestionCard()
    s = suggestion()
    card.show_suggestion(s)
    got = []
    card.feedback.connect(lambda *a: got.append(a))
    getattr(card, button).click()
    getattr(card, button).click()  # single-use
    assert got == [(s.id, outcome, None)]
    assert not card.buttons_enabled()


def test_edit_inline(qapp):
    card = SuggestionCard()
    s = suggestion()
    card.show_suggestion(s)
    got = []
    card.feedback.connect(lambda *a: got.append(a))
    card.edit_btn.click()
    assert not card.editor.isHidden()
    assert card.editor.text() == s.candidate.text
    card.editor.setText("Look at the stack trace first")
    card.save_edit_btn.click()
    assert got == [(s.id, FeedbackOutcome.EDITED, "Look at the stack trace first")]


def test_stale_greys_and_blocks_feedback(qapp):
    card = SuggestionCard()
    card.show_suggestion(suggestion())
    got = []
    card.feedback.connect(lambda *a: got.append(a))
    card.mark_stale()
    assert card.stale and not card.buttons_enabled()
    assert card.property("stale") == "true"
    card.accept_btn.click()
    assert got == []
