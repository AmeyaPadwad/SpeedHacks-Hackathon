"""Main window with a fake runtime: pause, result rendering, feedback, staleness."""

from dataclasses import dataclass, field

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from soma.consent import ConsentManager, Scope
from soma.types import (ActionType, Candidate, FeedbackOutcome, OCRElement, ScreenObservation,
                        Suggestion, WindowInfo)
from soma.ui.desktop.bridge import RuntimeBridge
from soma.ui.desktop.main_window import MainWindow, chosen_label
from tests.test_ui_bridge import FakeRuntime, FakeScheduler


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@dataclass
class FakeResult:
    frame_version: int
    window: WindowInfo
    observation: ScreenObservation
    candidates: list
    suggestion: Suggestion
    retrieved: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    skipped_reason: str | None = None


def make_result(version=1, silent=False, title="main.py - Code"):
    win = WindowInfo("Code.exe", title)
    els = [OCRElement("TypeError: bad operand", (0.1, 0.1, 0.5, 0.15), 0.93, ui_role="error"),
           OCRElement("Run", (0.6, 0.1, 0.7, 0.15), 0.88, ui_role="button")]
    obs = ScreenObservation(window=win, elements=els, version=version)
    cands = [Candidate(ActionType.EXPLAIN_ERROR, "Explain the TypeError", target=els[0],
                       evidence=["TypeError: bad operand"]),
             Candidate(ActionType.SUGGEST_NEXT_STEP, "Re-run the tests")]
    dist = {"NO_SUGGESTION": 0.2, "EXPLAIN_ERROR": 0.7, "SUGGEST_NEXT_STEP": 0.1}
    if silent:
        dist = {"NO_SUGGESTION": 0.8, "EXPLAIN_ERROR": 0.15, "SUGGEST_NEXT_STEP": 0.05}
        sug = Suggestion(None, 0.8, version, reason="below threshold", distribution=dist)
    else:
        sug = Suggestion(cands[0], 0.7, version, reason="error on screen", distribution=dist)
    return FakeResult(version, win, obs, cands, sug,
                      timings_ms={"capture": 5.0, "ocr": 20.0, "policy": 3.0, "total": 30.0})


@pytest.fixture
def window(qapp):
    consent = ConsentManager()
    consent.grant(Scope.SCREEN_CAPTURE)
    rt = FakeRuntime()
    bridge = RuntimeBridge(rt, 1.0, scheduler_factory=FakeScheduler)
    win = MainWindow(bridge, consent)
    bridge.start()
    yield win, bridge, rt, consent
    win.close()


def deliver(bridge, res):
    bridge.on_worker_result(res)
    QCoreApplication.processEvents()


def test_initial_off_state(qapp):
    bridge = RuntimeBridge(FakeRuntime(), 1.0, scheduler_factory=FakeScheduler)
    win = MainWindow(bridge, ConsentManager())
    assert win.state == "Off" and not win.pause_btn.isEnabled()


def test_pause_button_flips_state_and_calls_runtime(window):
    win, bridge, rt, _ = window
    assert win.state == "Capturing" and win.pause_btn.text() == "Pause"
    win.pause_btn.click()
    assert win.state == "Paused" and win.pause_btn.text() == "Resume"
    assert ("pause",) in rt.calls
    win.pause_btn.click()
    assert win.state == "Capturing" and ("resume",) in rt.calls


def test_result_fills_ocr_chart_card_and_timings(window):
    win, bridge, _, _ = window
    deliver(bridge, make_result())
    assert win.ocr_table.rowCount() == 2
    assert win.ocr_table.item(0, 0).text() == "TypeError: bad operand"
    assert win.ocr_table.item(0, 1).text() == "error"
    assert win.ocr_table.item(0, 2).text() == "0.93"
    labels = [k for k, _ in win.chart.items]
    assert "NO_SUGGESTION" in labels and win.chart.total() == pytest.approx(1.0)
    assert win.chart.chosen == "EXPLAIN_ERROR"
    assert win.card.title.text() == "Explain the TypeError" and win.card.buttons_enabled()
    assert "Code.exe" in win.app_label.text()
    assert "P50" in win.timing_label.text() and "ocr 20" in win.timing_label.text()
    assert win.debug.has_image  # layout-only view without a frame tap


def test_title_hidden_without_app_monitoring_and_redacted_with(window):
    win, bridge, _, consent = window
    deliver(bridge, make_result())
    assert "hidden" in win.title_label.text()
    consent.grant(Scope.APP_MONITORING)
    deliver(bridge, make_result(2, title="token sk-abcdefghijklmnopqrstuvwxyz123456"))
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in win.title_label.text()


@pytest.mark.parametrize("button,outcome", [("accept_btn", FeedbackOutcome.ACCEPTED),
                                            ("reject_btn", FeedbackOutcome.REJECTED)])
def test_accept_reject_call_feedback(window, button, outcome):
    win, bridge, rt, _ = window
    res = make_result()
    deliver(bridge, res)
    getattr(win.card, button).click()
    assert ("feedback", res.suggestion.id, outcome, None) in rt.calls


def test_edit_calls_feedback_with_text(window):
    win, bridge, rt, _ = window
    res = make_result()
    deliver(bridge, res)
    win.card.edit_btn.click()
    win.card.editor.setText("Check line 12 first")
    win.card.editor.returnPressed.emit()
    assert ("feedback", res.suggestion.id, FeedbackOutcome.EDITED, "Check line 12 first") in rt.calls


def test_silent_result_has_no_enabled_buttons(window):
    win, bridge, _, _ = window
    deliver(bridge, make_result(silent=True))
    assert win.card.title.text().startswith("No suggestion (below threshold")
    assert not win.card.buttons_enabled()
    assert win.chart.chosen == "NO_SUGGESTION"


def test_version_change_greys_out_open_suggestion(window):
    win, bridge, rt, _ = window
    deliver(bridge, make_result(1))
    deliver(bridge, make_result(2, silent=True))
    assert win.card.stale and not win.card.buttons_enabled()
    win.card.accept_btn.click()
    assert not any(c[0] == "feedback" for c in rt.calls)
    deliver(bridge, make_result(3, silent=True))   # next quiet frame replaces the stale card
    assert not win.card.stale and win.card.title.text().startswith("No suggestion")


def test_same_version_silent_keeps_suggestion(window):
    win, bridge, _, _ = window
    deliver(bridge, make_result(1))
    deliver(bridge, make_result(1, silent=True))
    assert win.card.active


def test_skipped_result_only_updates_status(window):
    win, bridge, _, _ = window
    deliver(bridge, make_result(1))
    skipped = make_result(1)
    skipped.skipped_reason = "unchanged"
    deliver(bridge, skipped)
    assert "unchanged" in win.skip_label.text() and win.card.active


def test_results_after_pause_not_rendered_and_debug_cleared(window):
    win, bridge, _, _ = window
    deliver(bridge, make_result(1))
    assert win.debug.has_image
    bridge.on_worker_result(make_result(2, title="later"))   # in flight
    win.pause_btn.click()
    QCoreApplication.processEvents()
    assert win.card.suggestion.screen_version == 1
    assert not win.debug.has_image
    assert win.card.stale


def test_chosen_label_for_repeated_actions():
    res = make_result()
    res.candidates.append(Candidate(ActionType.EXPLAIN_ERROR, "Explain another"))
    res.suggestion = Suggestion(res.candidates[2], 0.6, 1)
    assert chosen_label(res) == "EXPLAIN_ERROR#2"


def test_feedback_error_does_not_crash(window):
    win, bridge, rt, _ = window

    def boom(*a, **k):
        raise RuntimeError("store closed")
    rt.feedback = boom
    deliver(bridge, make_result())
    win.card.accept_btn.click()
    assert win.feedback_errors == ["RuntimeError"]


def test_ingest_folders_reports_counts(window, tmp_path):
    win, _, _, _ = window

    class Rep:
        changed = 3
        errors = []

    class Ing:
        def ingest_folder(self, f):
            return Rep()
    win.ingest_folders([str(tmp_path)], Ing(), spawn=lambda fn: fn())
    QCoreApplication.processEvents()
    assert win.ingest_label.text() == "Indexed 3 file(s)"
