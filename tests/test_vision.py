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


class LM:
    def __init__(self, x, y):
        self.x, self.y = x, y


def hand(extended=(True, True, True, True), pinch=False):
    """21 landmarks: wrist at the bottom, fingers pointing up; folded fingers curl back toward the wrist."""
    pts = [LM(0.5, 0.9)] + [LM(0.42, 0.8), LM(0.38, 0.7), LM(0.36, 0.62), LM(0.35, 0.55)]   # wrist + thumb
    for i, up in enumerate(extended):
        x = 0.45 + i * 0.05
        pts += [LM(x, 0.65), LM(x, 0.55), LM(x, 0.47), LM(x, 0.40)] if up else \
               [LM(x, 0.65), LM(x, 0.58), LM(x, 0.66), LM(x, 0.74)]
    if pinch:
        pts[4] = LM(pts[8].x + 0.01, pts[8].y)
    return pts


def test_gesture_names():
    from pet.vision import gesture_name
    # pinch is the only gesture; every other hand shape reports nothing
    assert gesture_name(hand()) == ""
    assert gesture_name(hand((False,) * 4)) == ""
    assert gesture_name(hand((True, False, False, False))) == ""
    assert gesture_name(hand(pinch=True)) == "pinch"
    assert gesture_name(None) == ""


def test_preview_is_small_mirrored_and_owns_its_pixels(qapp):
    import numpy as np
    from pet.vision import preview
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[:, :320] = (255, 0, 0)                           # left half red in the camera image
    out = preview(rgb, hand())
    img = out["image"]
    del rgb
    assert (img.width(), img.height()) == (168, 168)
    assert img.pixelColor(150, 84).red() == 255 and img.pixelColor(10, 84).red() == 0   # mirrored
    assert len(out["hand"]) == 21 and all(0 <= x <= 1.2 and 0 <= y <= 1 for x, y in out["hand"])
    assert out["gesture"] == ""
