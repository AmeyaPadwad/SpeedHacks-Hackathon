from types import SimpleNamespace as P

from pet.vision import (EYES_CLOSED, EYES_OPEN, FACE_AWAY, FACE_BACK, PINCH, Observation, VisionPolicy,
                        blink_scores, pinch_ratio)


def run(policy, frames):
    out = []
    for o in frames:
        out += [(o.t, e) for e in policy.update(o)]
    return out


def ticks(start, end, step=0.1, **kw):
    t = start
    while t < end - 1e-9:
        yield Observation(round(t, 3), **kw)
        t += step


def test_face_away_after_3s_back_after_1s():
    p = VisionPolicy()
    ev = run(p, [*ticks(0, 2.9, face=False), Observation(2.95, face=True), *ticks(3, 6.2, face=False)])
    assert ev == [(6.0, FACE_AWAY)]   # the brief return reset the 3 s timer
    ev = run(p, ticks(7, 8.2, face=True))
    assert ev == [(8.0, FACE_BACK)]


def test_eye_closure_hold_and_hysteresis():
    p = VisionPolicy()
    closed = dict(face=True, blink_left=0.8, blink_right=0.7)
    ev = run(p, ticks(0, 0.5, **closed))
    assert ev == [(0.3, EYES_CLOSED)]
    assert run(p, ticks(0.5, 0.8, face=True, blink_left=0.5, blink_right=0.5)) == []  # between thresholds
    assert run(p, [Observation(0.9, True, 0.3, 0.2)]) == [(0.9, EYES_OPEN)]


def test_one_eye_is_not_closure():
    assert run(VisionPolicy(), ticks(0, 1, face=True, blink_left=0.9, blink_right=0.1)) == []


def test_pinch_hold_rearm_cooldown():
    p = VisionPolicy()
    ev = run(p, ticks(0, 1.5, face=True, pinch_ratio=0.1))
    assert ev == [(0.5, PINCH)]                                  # fires once, then needs rearm
    run(p, [Observation(1.6, True, pinch_ratio=0.5)])            # rearm
    assert run(p, ticks(1.7, 2.4, face=True, pinch_ratio=0.1)) == []  # held long enough but in cooldown
    run(p, [Observation(2.5, True, pinch_ratio=0.5)])
    assert run(p, ticks(2.6, 3.2, face=True, pinch_ratio=0.1)) == [(3.1, PINCH)]


def test_missing_hand_resets_pinch_hold():
    p = VisionPolicy()
    frames = [*ticks(0, 0.4, face=True, pinch_ratio=0.1), Observation(0.4, True, pinch_ratio=None),
              *ticks(0.5, 0.9, face=True, pinch_ratio=0.1)]
    assert run(p, frames) == []


def test_pinch_ratio_and_blink_helpers():
    lm = [P(x=0, y=0) for _ in range(21)]
    lm[9] = P(x=0, y=1)
    lm[4], lm[8] = P(x=0.1, y=0.5), P(x=0.2, y=0.5)
    assert abs(pinch_ratio(lm) - 0.1) < 1e-9
    assert pinch_ratio(lm[:5]) is None
    lm[9] = P(x=0, y=0)
    assert pinch_ratio(lm) is None  # degenerate hand scale
    shapes = [P(category_name="eyeBlinkLeft", score=0.7), P(category_name="eyeBlinkRight", score=0.2)]
    assert blink_scores(shapes) == (0.7, 0.2)
