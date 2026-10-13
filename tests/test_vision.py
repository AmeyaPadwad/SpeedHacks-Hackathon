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


# --- new gestures: pointing left / right, pinch and drag ------------------------------------------

def pointing_hand(dx: float, curled=True):
    """Index finger stretched sideways by `dx` (image coords: + = image right = YOUR left)."""
    pts = [LM(0.5, 0.6)] + [LM(0.42, 0.55), LM(0.40, 0.52), LM(0.42, 0.50), LM(0.45, 0.50)]  # wrist + thumb
    pts += [LM(0.45, 0.45), LM(0.45 + dx * 0.35, 0.455), LM(0.45 + dx * 0.7, 0.458), LM(0.45 + dx, 0.46)]
    for mcp_x in (0.5, 0.55, 0.58):                     # middle, ring, pinky
        tip = (mcp_x, 0.52) if curled else (mcp_x, 0.2)
        pts += [LM(mcp_x, 0.43), LM(mcp_x, 0.40), LM(*tip), LM(*tip)]
    return pts


def test_pointing_geometry():
    from pet.vision import gesture_name, pointing
    assert pointing(pointing_hand(+0.2)) == "left"       # fingertip to the image's right = your left
    assert pointing(pointing_hand(-0.2)) == "right"
    assert pointing(pointing_hand(+0.04)) is None        # index not stretched out
    assert pointing(pointing_hand(+0.2, curled=False)) is None   # open hand is not pointing
    assert pointing(hand()) is None and pointing(None) is None   # fingers up
    assert gesture_name(pointing_hand(-0.2)) == "point right"


def test_point_fires_once_after_hold_and_rearms_on_relax():
    from pet.vision import POINT_LEFT, POINT_RIGHT
    p = VisionPolicy()
    assert run(p, ticks(0, 1.0, face=True, point="left")) == [(0.3, POINT_LEFT)]   # 0.25 s hold, once
    run(p, [Observation(1.0, True)])                                               # relax
    assert run(p, ticks(1.1, 1.6, face=True, point="left")) == [(1.4, POINT_LEFT)]
    assert run(p, ticks(1.6, 2.0, face=True, point="right")) == []                 # cooldown 0.6 s
    assert run(p, ticks(2.0, 2.4, face=True, point="right")) == [(2.0, POINT_RIGHT)]  # held since 1.6


def test_pinch_drag_scrolls_and_never_pinches():
    from pet.vision import DRAG_END, DRAG_START
    p = VisionPolicy()
    frames = [Observation(round(0.1 * i, 2), True, pinch_ratio=0.1, pinch_y=0.5 - 0.02 * i) for i in range(10)]
    ev = run(p, frames)
    assert [e for _, e in ev] == [DRAG_START] and PINCH not in [e for _, e in ev]
    assert p.dragging and p.take_scroll() < -0.1 and p.take_scroll() == 0.0      # hand moved up
    ev = run(p, [Observation(1.1, True, pinch_ratio=0.6, pinch_y=0.3)])
    assert ev == [(1.1, DRAG_END)] and not p.dragging


def test_still_pinch_is_a_pinch_and_small_jitter_is_not_a_drag():
    p = VisionPolicy()
    frames = [Observation(round(0.1 * i, 2), True, pinch_ratio=0.1, pinch_y=0.5 + (0.01 if i % 2 else 0))
              for i in range(8)]
    assert run(p, frames) == [(0.5, PINCH)] and not p.dragging and p.take_scroll() == 0.0


def test_losing_the_hand_ends_a_drag_and_no_pointing_while_dragging():
    from pet.vision import DRAG_END, DRAG_START
    p = VisionPolicy()
    ev = run(p, [Observation(0.0, True, pinch_ratio=0.1, pinch_y=0.5),
                 Observation(0.1, True, pinch_ratio=0.1, pinch_y=0.6, point="left")])
    assert ev == [(0.1, DRAG_START)] and p.dragging
    assert run(p, ticks(0.2, 0.6, face=True, pinch_ratio=0.1, pinch_y=0.7, point="left")) == []
    assert run(p, [Observation(0.7, True)]) == [(0.7, DRAG_END)]
