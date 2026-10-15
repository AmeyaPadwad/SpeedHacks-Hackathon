// Watcher against a fake target, fake fact log and fake camera (port of the watcher parts of
// tests/test_e2e.py and tests/test_browser.py).
import type { FactLog, Outcome, PreviewFrame, VisionEvent, WatcherTarget } from "../contracts";
import type { BaselineSummary } from "./policy";
import { Watcher, type CameraLike } from "./watcher";

class FakeTarget implements WatcherTarget {
  paused = false;
  calls: unknown[][] = [];
  pinch() { this.calls.push(["pinch"]); }
  nudge() { this.calls.push(["nudge"]); }
  setPresence(away: boolean, eyesClosed: boolean) { this.calls.push(["presence", away, eyesClosed]); }
  setFrame(frame: PreviewFrame) { this.calls.push(["frame", frame.gesture]); }
  step(direction: -1 | 1) { this.calls.push(["step", direction]); }
  scroll(dy: number) { this.calls.push(["scroll", dy]); }
  of(kind: string) { return this.calls.filter((c) => c[0] === kind); }
}

class FakeLog implements FactLog {
  events: [string, number | null | undefined][] = [];
  offers: [string, Outcome][] = [];
  baselines: number[][] = [];
  logEvent(kind: string, duration?: number | null) { this.events.push([kind, duration]); }
  logOffer(kind: string, outcome: Outcome) { this.offers.push([kind, outcome]); }
  logBlinkBaseline(mean: number, p95: number, samples: number) { this.baselines.push([mean, p95, samples]); }
}

class FakeCamera implements CameraLike {
  onEvent: (ev: VisionEvent) => void = () => {};
  onFrame: (frame: PreviewFrame) => void = () => {};
  onScroll: (dy: number) => void = () => {};
  onBaseline: (summary: BaselineSummary) => void = () => {};
  onError: (message: string) => void = () => {};
  onStatus: (message: string) => void = () => {};
  running = false;
  constructor(readonly blinkOn: number | null) {}
  async start() { this.running = true; }
  stop() { this.running = false; }
}

function setup() {
  const target = new FakeTarget();
  const log = new FakeLog();
  const clock = { t: 100 };
  const cams: FakeCamera[] = [];
  const indicator: boolean[] = [];
  const statuses: string[] = [];
  const w = new Watcher(target, {
    factLog: log,
    clock: () => clock.t,
    blinkOn: () => 0.7,
    createCamera: (b) => { const c = new FakeCamera(b); cams.push(c); return c; },
    onCameraChange: (on) => indicator.push(on),
    onStatus: (m) => statuses.push(m),
  });
  return { w, target, log, clock, cams, indicator, statuses };
}

const frame = (gesture = ""): PreviewFrame => ({ bitmap: null, hands: [], gesture });

test("away and back logs the away time and nudges", () => {
  const { w, target, log, clock } = setup();
  w.onVisionEvent("face_away");
  expect(target.of("presence").at(-1)).toEqual(["presence", true, false]);
  clock.t = 112.34;
  w.onVisionEvent("face_back");
  expect(target.of("nudge")).toHaveLength(1);
  expect(target.of("presence").at(-1)).toEqual(["presence", false, false]);
  expect(log.events).toEqual([["face_away", null], ["face_back", null], ["away_ended", 12.3]]);
});

test("eyes closed and open log the closed time", () => {
  const { w, target, log, clock } = setup();
  w.onVisionEvent("eyes_closed");
  expect(target.of("presence").at(-1)).toEqual(["presence", false, true]);
  clock.t = 101.234;
  w.onVisionEvent("eyes_open");
  expect(target.of("presence").at(-1)).toEqual(["presence", false, false]);
  expect(log.events.at(-1)).toEqual(["eyes_closed_ended", 1.23]);
});

test("gestures: pinch, point left/right step, drags logged, scroll forwarded", () => {
  const { w, target, log } = setup();
  w.onVisionEvent("point_right");
  w.onVisionEvent("point_left");
  w.onVisionEvent("drag_start");
  w.onScroll(-0.1);
  w.onVisionEvent("drag_end");
  w.onVisionEvent("pinch");
  expect(target.calls).toEqual([["step", 1], ["step", -1], ["scroll", -0.1], ["pinch"]]);   // no presence changes
  expect(log.events.map(([k]) => k)).toEqual(["point_right", "point_left", "drag_start", "drag_end", "pinch"]);
});

test("paused: no reactions, no logging, no frames", () => {
  const { w, target, log } = setup();
  w.setCamera(true);
  target.calls = [];
  target.paused = true;
  for (const ev of ["pinch", "point_right", "face_away", "eyes_closed", "drag_start"] as VisionEvent[]) {
    w.onVisionEvent(ev);
  }
  w.onScroll(-1);
  let closed = false;
  w.onFrame({ bitmap: { close: () => { closed = true; } } as unknown as ImageBitmap, hands: [], gesture: "" });
  expect(target.calls).toEqual([]);
  expect(log.events).toEqual([]);
  expect(closed).toBe(true);   // a dropped preview is freed, never kept
});

test("frames reach the orb only while the camera is on", () => {
  const { w, target } = setup();
  w.onFrame(frame("pinch"));
  expect(target.of("frame")).toEqual([]);
  w.setCamera(true);
  w.onFrame(frame("pinch"));
  expect(target.of("frame")).toEqual([["frame", "pinch"]]);
});

test("setCamera wires the worker with the calibrated threshold and releases it", () => {
  const { w, target, log, cams, indicator } = setup();
  w.setCamera(true);
  w.setCamera(true);                                   // already on: no second camera
  expect(cams).toHaveLength(1);
  const cam = cams[0];
  expect(cam.blinkOn).toBe(0.7);
  expect(cam.running).toBe(true);
  expect(w.cameraOn).toBe(true);
  cam.onEvent("face_away");
  cam.onScroll(0.2);
  cam.onFrame(frame());
  cam.onBaseline({ mean: 0.1, p95: 0.3, samples: 40 });
  expect(target.of("scroll")).toEqual([["scroll", 0.2]]);
  expect(log.baselines).toEqual([[0.1, 0.3, 40]]);
  expect(w.present).toBe(false);
  w.setCamera(false);
  expect(cam.running).toBe(false);
  expect(w.cameraOn).toBe(false);
  expect(w.present).toBe(true);
  expect(target.of("presence").at(-1)).toEqual(["presence", false, false]);
  expect(indicator).toEqual([true, true, false]);
  cam.onEvent("pinch");                                // a stale worker is ignored
  expect(target.of("pinch")).toEqual([]);
});

test("a camera that fails resets presence and turns the indicator off", () => {
  const { w, target, cams, indicator, statuses } = setup();
  w.setCamera(true);
  w.onVisionEvent("eyes_closed");
  cams[0].running = false;
  cams[0].onError("Camera disconnected.");
  expect(statuses).toEqual(["Camera disconnected."]);
  expect(w.cameraOn).toBe(false);
  expect(indicator.at(-1)).toBe(false);
  expect(target.of("presence").at(-1)).toEqual(["presence", false, false]);
});

test("works without a fact log", () => {
  const target = new FakeTarget();
  const w = new Watcher(target, { createCamera: (b) => new FakeCamera(b) });
  w.onVisionEvent("face_away");
  w.onVisionEvent("face_back");
  w.onBaseline({ mean: 0, p95: 0, samples: 20 });
  expect(target.of("nudge")).toHaveLength(1);
});
