"""Grounding: targets must still exist on the latest frame; stale suggestions are invalid."""

from __future__ import annotations

import pytest

from soma.policy.grounding import (
    find_target,
    ground_box,
    ground_point,
    iou,
    is_stale,
    normalize_text,
    text_matches,
    verify_target,
)
from soma.types import ActionType, Candidate, OCRElement, ScreenObservation, Suggestion, WindowInfo

ERR = "ZeroDivisionError: division by zero"
BOX = (0.10, 0.50, 0.60, 0.54)


def obs(*elements: OCRElement, version: int = 2, window_id: str = "") -> ScreenObservation:
    return ScreenObservation(window=WindowInfo("Code.exe", "t", window_id=window_id),
                             elements=list(elements), version=version)


def cand(text: str = ERR, box=BOX, window_id: str = "") -> Candidate:
    return Candidate(ActionType.EXPLAIN_ERROR, f'Explain the error "{text}"',
                     target=OCRElement(text, box, window_id=window_id))


def test_iou_and_normalize():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iou((0, 0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
    assert normalize_text("  ZeroDivisionError:\tdivision  BY zero. ") == "zerodivisionerror: division by zero"
    assert text_matches(ERR, "ZeroDivisionErr0r: division by zero")  # OCR jitter
    assert not text_matches(ERR, "KeyError: 'x'")
    assert not text_matches("", "")


def test_same_target_verifies():
    assert verify_target(cand(), obs(OCRElement(ERR, BOX)))


def test_slightly_resized_target_verifies():
    resized = (0.11, 0.505, 0.58, 0.545)
    assert iou(BOX, resized) >= 0.5
    assert verify_target(cand(), obs(OCRElement(ERR, resized)))


def test_small_shift_verifies_even_with_low_iou():
    shifted = (0.10, 0.53, 0.60, 0.57)  # moved down by 3% of the screen
    assert iou(BOX, shifted) < 0.5
    assert verify_target(cand(), obs(OCRElement(ERR, shifted)))


def test_far_moved_target_fails():
    moved = (0.10, 0.10, 0.60, 0.14)
    assert not verify_target(cand(), obs(OCRElement(ERR, moved)))


def test_missing_target_fails():
    assert not verify_target(cand(), obs(OCRElement("All tests passed", BOX)))
    assert not verify_target(cand(), obs())


def test_changed_text_at_same_place_fails():
    assert not verify_target(cand(), obs(OCRElement("KeyError: 'user'", BOX)))


def test_other_window_fails():
    assert not verify_target(cand(window_id="w1"), obs(OCRElement(ERR, BOX), window_id="w2"))
    assert verify_target(cand(window_id="w1"), obs(OCRElement(ERR, BOX), window_id="w1"))


def test_untargeted_candidate_always_verifies():
    c = Candidate(ActionType.ASK_CLARIFYING_QUESTION, "Which error first?")
    assert verify_target(c, obs())


def test_find_target_picks_best_of_duplicates():
    near = OCRElement(ERR, (0.10, 0.51, 0.60, 0.55))
    exact = OCRElement(ERR, BOX)
    assert find_target(cand().target, obs(near, exact)) is exact


def test_ground_point_maps_and_clamps():
    assert ground_point((0.0, 0.0, 1.0, 1.0), (1920, 1080)) == (960, 540)
    assert ground_point(OCRElement("x", (0.0, 0.0, 0.0, 0.0)), (100, 50)) == (0, 0)
    assert ground_point((1.2, 1.2, 1.4, 1.4), (100, 50)) == (99, 49)
    # same normalized target after a window resize lands proportionally
    assert ground_point(BOX, (800, 600)) == (round(0.35 * 799), round(0.52 * 599))
    assert ground_box(BOX, (1000, 100)) == (100, 50, 600, 54)
    with pytest.raises(ValueError):
        ground_point(BOX, (0, 10))


def test_is_stale_versions():
    s = Suggestion(candidate=cand(), confidence=0.9, screen_version=5)
    assert not is_stale(s, 5)
    assert is_stale(s, 6)                              # conservative default
    assert not is_stale(s, 6, changed=False)           # caller says nothing relevant changed
    assert is_stale(s, 6, changed=True)
    assert is_stale(s, 7, changed=lambda old, new: new - old > 1)
    assert not is_stale(s, 6, changed=lambda old, new: new - old > 1)
    assert is_stale(s, 4)                              # future suggestion: inconsistent


def test_is_stale_checks_target_on_latest_obs():
    s = Suggestion(candidate=cand(), confidence=0.9, screen_version=5)
    assert not is_stale(s, 6, changed=False, latest_obs=obs(OCRElement(ERR, BOX)))
    assert is_stale(s, 6, changed=False, latest_obs=obs(OCRElement("ok", BOX)))
    silent = Suggestion(candidate=None, confidence=0.9, screen_version=5)
    assert not is_stale(silent, 5, latest_obs=obs())
