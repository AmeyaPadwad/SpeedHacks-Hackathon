// Port of tests/test_vision.py (policy, geometry helpers, baseline meter).
import type { VisionEvent } from "../contracts";
import {
  BaselineMeter, gestureName, observation, pinchRatio, pointing, VisionPolicy,
  type Landmark, type Observation,
} from "./policy";

type Rest = Parameters<typeof observation>[2];

function run(policy: VisionPolicy, frames: Iterable<Observation>): [number, VisionEvent][] {
  const out: [number, VisionEvent][] = [];
  for (const o of frames) for (const e of policy.update(o)) out.push([o.t, e]);
  return out;
}

function* ticks(start: number, end: number, face: boolean, rest: Rest = {}, step = 0.1): Generator<Observation> {
  let t = start;
  while (t < end - 1e-9) {
    yield observation(Math.round(t * 1000) / 1000, face, rest);
    t += step;
  }
}

const lm = (x: number, y: number): Landmark => ({ x, y });

test("face away after 3 s, back after 1 s", () => {
  const p = new VisionPolicy();
  const ev = run(p, [...ticks(0, 2.9, false), observation(2.95, true), ...ticks(3, 6.2, false)]);
  expect(ev).toEqual([[6.0, "face_away"]]);   // the brief return reset the 3 s timer
  expect(run(p, ticks(7, 8.2, true))).toEqual([[8.0, "face_back"]]);
});

test("eye closure hold and hysteresis", () => {
  const p = new VisionPolicy();
  expect(run(p, ticks(0, 0.5, true, { blinkLeft: 0.8, blinkRight: 0.7 }))).toEqual([[0.3, "eyes_closed"]]);
  expect(run(p, ticks(0.5, 0.8, true, { blinkLeft: 0.5, blinkRight: 0.5 }))).toEqual([]);   // between thresholds
  expect(run(p, [observation(0.9, true, { blinkLeft: 0.3, blinkRight: 0.2 })])).toEqual([[0.9, "eyes_open"]]);
});

test("one eye is not closure", () => {
  expect(run(new VisionPolicy(), ticks(0, 1, true, { blinkLeft: 0.9, blinkRight: 0.1 }))).toEqual([]);
});

test("calibrated blink threshold keeps reopen 0.2 below, at least 0.2", () => {
  expect(new VisionPolicy(0.7).blinkOff).toBeCloseTo(0.5);
  expect(new VisionPolicy(0.3).blinkOff).toBe(0.2);
  expect(new VisionPolicy().blinkOn).toBe(0.6);
});

test("losing the face clears eye closure", () => {
  const p = new VisionPolicy();
  run(p, ticks(0, 0.5, true, { blinkLeft: 0.9, blinkRight: 0.9 }));
  expect(p.eyesClosed).toBe(true);
  expect(run(p, ticks(0.5, 3.6, false))).toEqual([[3.5, "face_away"]]);
  expect(p.eyesClosed).toBe(false);
});

test("pinch hold, rearm, cooldown", () => {
  const p = new VisionPolicy();
  expect(run(p, ticks(0, 1.5, true, { pinchRatio: 0.1 }))).toEqual([[0.5, "pinch"]]);   // once, then rearm
  run(p, [observation(1.6, true, { pinchRatio: 0.5 })]);                                  // rearm
  expect(run(p, ticks(1.7, 2.4, true, { pinchRatio: 0.1 }))).toEqual([]);                // in cooldown
  run(p, [observation(2.5, true, { pinchRatio: 0.5 })]);
  expect(run(p, ticks(2.6, 3.2, true, { pinchRatio: 0.1 }))).toEqual([[3.1, "pinch"]]);
});

test("missing hand resets pinch hold", () => {
  const p = new VisionPolicy();
  const frames = [...ticks(0, 0.4, true, { pinchRatio: 0.1 }), observation(0.4, true),
    ...ticks(0.5, 0.9, true, { pinchRatio: 0.1 })];
  expect(run(p, frames)).toEqual([]);
});

test("pinch ratio helper", () => {
  const pts = Array.from({ length: 21 }, () => lm(0, 0));
  pts[9] = lm(0, 1);
  pts[4] = lm(0.1, 0.5);
  pts[8] = lm(0.2, 0.5);
  expect(Math.abs(pinchRatio(pts)! - 0.1)).toBeLessThan(1e-9);
  expect(pinchRatio(pts.slice(0, 5))).toBeNull();
  pts[9] = lm(0, 0);
  expect(pinchRatio(pts)).toBeNull();   // degenerate hand scale
});

/** 21 landmarks: wrist at the bottom, fingers up; folded fingers curl back toward the wrist. */
function hand(extended: boolean[] = [true, true, true, true], pinch = false): Landmark[] {
  const pts = [lm(0.5, 0.9), lm(0.42, 0.8), lm(0.38, 0.7), lm(0.36, 0.62), lm(0.35, 0.55)];   // wrist + thumb
  extended.forEach((up, i) => {
    const x = 0.45 + i * 0.05;
    pts.push(...(up ? [lm(x, 0.65), lm(x, 0.55), lm(x, 0.47), lm(x, 0.4)]
      : [lm(x, 0.65), lm(x, 0.58), lm(x, 0.66), lm(x, 0.74)]));
  });
  if (pinch) pts[4] = lm(pts[8].x + 0.01, pts[8].y);
  return pts;
}

test("gesture names", () => {
  expect(gestureName(hand())).toBe("");
  expect(gestureName(hand([false, false, false, false]))).toBe("");
  expect(gestureName(hand([true, false, false, false]))).toBe("");
  expect(gestureName(hand(undefined, true))).toBe("pinch");
  expect(gestureName(null)).toBe("");
});

/** Index finger stretched sideways by `dx` (image coords: + = image right = YOUR left). */
function pointingHand(dx: number, curled = true): Landmark[] {
  const pts = [lm(0.5, 0.6), lm(0.42, 0.55), lm(0.4, 0.52), lm(0.42, 0.5), lm(0.45, 0.5)];
  pts.push(lm(0.45, 0.45), lm(0.45 + dx * 0.35, 0.455), lm(0.45 + dx * 0.7, 0.458), lm(0.45 + dx, 0.46));
  for (const mcpX of [0.5, 0.55, 0.58]) {
    const tip = curled ? lm(mcpX, 0.52) : lm(mcpX, 0.2);
    pts.push(lm(mcpX, 0.43), lm(mcpX, 0.4), tip, tip);
  }
  return pts;
}

test("pointing geometry", () => {
  expect(pointing(pointingHand(+0.2))).toBe("left");          // fingertip to the image's right = your left
  expect(pointing(pointingHand(-0.2))).toBe("right");
  expect(pointing(pointingHand(+0.04))).toBeNull();           // index not stretched out
  expect(pointing(pointingHand(+0.2, false))).toBeNull();     // open hand is not pointing
  expect(pointing(hand())).toBeNull();
  expect(pointing(null)).toBeNull();
  expect(gestureName(pointingHand(-0.2))).toBe("point right");
});

test("point fires once after hold and rearms on relax", () => {
  const p = new VisionPolicy();
  expect(run(p, ticks(0, 1.0, true, { point: "left" }))).toEqual([[0.3, "point_left"]]);   // 0.25 s hold, once
  run(p, [observation(1.0, true)]);                                                         // relax
  expect(run(p, ticks(1.1, 1.6, true, { point: "left" }))).toEqual([[1.4, "point_left"]]);
  expect(run(p, ticks(1.6, 2.0, true, { point: "right" }))).toEqual([]);                    // cooldown 0.6 s
  expect(run(p, ticks(2.0, 2.4, true, { point: "right" }))).toEqual([[2.0, "point_right"]]); // held since 1.6
});

test("pinch-drag scrolls and never pinches", () => {
  const p = new VisionPolicy();
  const frames = Array.from({ length: 10 }, (_, i) =>
    observation(Math.round(0.1 * i * 100) / 100, true, { pinchRatio: 0.1, pinchY: 0.5 - 0.02 * i }));
  const ev = run(p, frames);
  expect(ev.map(([, e]) => e)).toEqual(["drag_start"]);
  expect(p.dragging).toBe(true);
  expect(p.takeScroll()).toBeLessThan(-0.1);   // hand moved up
  expect(p.takeScroll()).toBe(0);
  expect(run(p, [observation(1.1, true, { pinchRatio: 0.6, pinchY: 0.3 })])).toEqual([[1.1, "drag_end"]]);
  expect(p.dragging).toBe(false);
});

test("still pinch is a pinch and small jitter is not a drag", () => {
  const p = new VisionPolicy();
  const frames = Array.from({ length: 8 }, (_, i) =>
    observation(Math.round(0.1 * i * 100) / 100, true, { pinchRatio: 0.1, pinchY: 0.5 + (i % 2 ? 0.01 : 0) }));
  expect(run(p, frames)).toEqual([[0.5, "pinch"]]);
  expect(p.dragging).toBe(false);
  expect(p.takeScroll()).toBe(0);
});

test("losing the hand ends a drag and no pointing while dragging", () => {
  const p = new VisionPolicy();
  const ev = run(p, [observation(0, true, { pinchRatio: 0.1, pinchY: 0.5 }),
    observation(0.1, true, { pinchRatio: 0.1, pinchY: 0.6, point: "left" })]);
  expect(ev).toEqual([[0.1, "drag_start"]]);
  expect(p.dragging).toBe(true);
  expect(run(p, ticks(0.2, 0.6, true, { pinchRatio: 0.1, pinchY: 0.7, point: "left" }))).toEqual([]);
  expect(run(p, [observation(0.7, true)])).toEqual([[0.7, "drag_end"]]);
});

test("baseline meter summarises open-eye scores every 30 s", () => {
  const m = new BaselineMeter();
  let out = null;
  for (let i = 0; i <= 30; i++) {
    const r = m.add(observation(i, true, { blinkLeft: i / 100, blinkRight: 0 }), false);
    if (r) out = r;
  }
  expect(out).toEqual({ mean: 0.15, p95: 0.28, samples: 31 });
  // Too few samples (eyes closed or no face): nothing.
  const m2 = new BaselineMeter();
  for (let i = 0; i <= 30; i++) expect(m2.add(observation(i, false), false)).toBeNull();
});
