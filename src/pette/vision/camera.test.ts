// CameraWorker with a fake stream, video, detectors and scheduler: never the real camera.
import type { PreviewFrame, VisionEvent } from "../contracts";
import { CameraWorker, type Detectors } from "./camera";
import type { PixelKeypoint } from "./landmarks";

class FakeTrack extends EventTarget {
  stopped = false;
  stop() { this.stopped = true; }
}

function fakeStream() {
  const track = new FakeTrack();
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
  return { stream, track };
}

/** A pinch at pixel height `y` in a 640x480 frame. */
function pinchHand(y = 250): PixelKeypoint[] {
  const k = Array.from({ length: 21 }, () => ({ x: 320, y: 300 }));
  k[0] = { x: 320, y: 440 };
  k[4] = { x: 300, y };
  k[8] = { x: 302, y };
  return k;
}

function setup(opts: { hand?: () => PixelKeypoint[] | null; face?: boolean; gumError?: Error } = {}) {
  const { stream, track } = fakeStream();
  const calls: MediaStreamConstraints[] = [];
  const disposed: string[] = [];
  const state = { hand: opts.hand ?? (() => null), face: opts.face ?? true, t: 0, hidden: false };
  const detectors: Detectors = {
    hands: {
      estimateHands: async () => { const k = state.hand(); return k ? [{ keypoints: k }] : []; },
      dispose: () => disposed.push("hands"),
    },
    faces: {
      estimateFaces: async () => (state.face ? [{ keypoints: [] }] : []),
      dispose: () => disposed.push("faces"),
    },
  };
  const video = { videoWidth: 640, videoHeight: 480, srcObject: null as unknown, play: async () => {} };
  let pending: (() => void) | null = null;
  const cam = new CameraWorker({
    getUserMedia: async (c) => { calls.push(c); if (opts.gumError) throw opts.gumError; return stream; },
    loadDetectors: async () => detectors,
    createVideo: () => video as unknown as HTMLVideoElement,
    makeBitmap: async () => null,
    schedule: (_v, cb) => { pending = cb; return () => { pending = null; }; },
    clock: () => state.t,
    isHidden: () => state.hidden,
  });
  const events: VisionEvent[] = [];
  const frames: PreviewFrame[] = [];
  const scrolls: number[] = [];
  const errors: string[] = [];
  const statuses: string[] = [];
  cam.onEvent = (e) => events.push(e);
  cam.onFrame = (f) => frames.push(f);
  cam.onScroll = (d) => scrolls.push(d);
  cam.onError = (m) => errors.push(m);
  cam.onStatus = (m) => statuses.push(m);
  /** Fires the scheduled frame callback at time t and lets the async frame finish. */
  const tick = async (t: number) => {
    state.t = t;
    const cb = pending;
    pending = null;
    cb?.();
    await new Promise((r) => setTimeout(r, 0));
  };
  return { cam, calls, track, video, disposed, state, events, frames, scrolls, errors, statuses, tick,
    get scheduled() { return pending !== null; } };
}

test("start asks for video only and stop releases the camera", async () => {
  const s = setup();
  await s.cam.start();
  expect(s.calls[0].audio).toBe(false);
  expect(s.calls[0].video).toBeTruthy();
  expect(s.cam.running).toBe(true);
  expect(s.video.srcObject).not.toBeNull();
  expect(s.statuses).toEqual(["Camera on (processed locally)"]);
  s.cam.stop();
  expect(s.track.stopped).toBe(true);
  expect(s.video.srcObject).toBeNull();
  expect(s.cam.running).toBe(false);
  expect(s.disposed.sort()).toEqual(["faces", "hands"]);
  expect(s.scheduled).toBe(false);
  expect(s.statuses.at(-1)).toBe("Camera off");
});

test("a denied camera is reported, not thrown", async () => {
  const err = Object.assign(new Error("no"), { name: "NotAllowedError" });
  const s = setup({ gumError: err });
  await s.cam.start();
  expect(s.errors).toEqual(["Camera unavailable: permission denied."]);
  expect(s.cam.running).toBe(false);
});

test("frames become events, previews and a pinch; capped at 15 fps", async () => {
  const s = setup({ hand: () => pinchHand() });
  await s.cam.start();
  for (let i = 0; i <= 8; i++) await s.tick(i * 0.1);
  expect(s.events).toEqual(["pinch"]);
  expect(s.frames).toHaveLength(9);
  expect(s.frames[0].gesture).toBe("pinch");
  expect(s.frames[0].hands[0]).toHaveLength(21);
  expect(s.frames[0].bitmap).toBeNull();
  // A callback 10 ms later is skipped (15 fps cap).
  await s.tick(0.81);
  expect(s.frames).toHaveLength(9);
  s.cam.stop();
});

test("pinch-and-drag emits scroll amounts", async () => {
  let y = 250;
  const s = setup({ hand: () => pinchHand(y) });
  await s.cam.start();
  for (let i = 0; i < 6; i++) { y = 250 + 20 * i; await s.tick(i * 0.1); }
  expect(s.events).toEqual(["drag_start"]);
  expect(s.scrolls.length).toBeGreaterThan(0);
  expect(s.scrolls.every((d) => d > 0)).toBe(true);   // hand moved down
  s.cam.stop();
});

test("no face for 3 s is face_away; hidden tab skips frames", async () => {
  const s = setup();
  s.state.face = false;
  await s.cam.start();
  s.state.hidden = true;
  await s.tick(0);
  expect(s.frames).toHaveLength(0);
  s.state.hidden = false;
  for (let i = 0; i <= 31; i++) await s.tick(0.1 * i);
  expect(s.events).toEqual(["face_away"]);
  s.cam.stop();
});

test("a track that ends reports a disconnect and stops", async () => {
  const s = setup();
  await s.cam.start();
  s.track.dispatchEvent(new Event("ended"));
  expect(s.errors).toEqual(["Camera disconnected."]);
  expect(s.cam.running).toBe(false);
});

test("stop during the permission prompt releases the stream when it arrives", async () => {
  const s = setup();
  const p = s.cam.start();
  s.cam.stop();
  await p;
  expect(s.track.stopped).toBe(true);
  expect(s.cam.running).toBe(false);
});
