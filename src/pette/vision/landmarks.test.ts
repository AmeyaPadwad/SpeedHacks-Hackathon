import {
  blinkScores, centerSquare, earToBlink, EYE_KEYPOINTS, eyeAspectRatio, handLandmarks, previewPoints,
  type PixelKeypoint,
} from "./landmarks";
import { gestureName } from "./policy";

/** A synthetic 478-point face whose eyes are `width` px wide and open by `gap` px. */
function face(gapLeft: number, gapRight: number, width = 60): PixelKeypoint[] {
  const pts: PixelKeypoint[] = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
  const eye = (which: "left" | "right", cx: number, gap: number) => {
    const { corners, lids } = EYE_KEYPOINTS[which];
    pts[corners[0]] = { x: cx - width / 2, y: 200 };
    pts[corners[1]] = { x: cx + width / 2, y: 200 };
    lids.forEach(([up, down], i) => {
      const x = cx - 10 + 20 * i;
      pts[up] = { x, y: 200 - gap / 2 };
      pts[down] = { x, y: 200 + gap / 2 };
    });
  };
  eye("right", 250, gapRight);
  eye("left", 390, gapLeft);
  return pts;
}

test("eye aspect ratio is lid gap over eye width", () => {
  expect(eyeAspectRatio(face(18, 6), "left")).toBeCloseTo(0.3);
  expect(eyeAspectRatio(face(18, 6), "right")).toBeCloseTo(0.1);
  expect(eyeAspectRatio(face(18, 6).slice(0, 300), "left")).toBeNull();   // missing keypoints
  expect(eyeAspectRatio(face(18, 6, 0), "left")).toBeNull();               // no width
});

test("EAR maps to a blendshape-like blink score (closed ~ 1)", () => {
  expect(earToBlink(0.3)).toBe(0);
  expect(earToBlink(0.28)).toBe(0);
  expect(earToBlink(0.1)).toBe(1);
  expect(earToBlink(0.02)).toBe(1);
  expect(earToBlink(0.19)).toBeCloseTo(0.5);
  // Policy thresholds: closed above 0.6, open again below 0.4.
  expect(earToBlink(0.16)).toBeGreaterThan(0.6);
  expect(earToBlink(0.22)).toBeLessThan(0.4);
});

test("blink scores per eye, scale invariant", () => {
  const [l, r] = blinkScores(face(18, 3));
  expect(l).toBe(0);
  expect(r).toBe(1);
  const big = face(18, 3).map((k) => ({ x: k.x * 3, y: k.y * 3 }));
  expect(blinkScores(big)).toEqual([l, r]);
  expect(blinkScores(null)).toEqual([0, 0]);
  expect(blinkScores([])).toEqual([0, 0]);
});

test("hand keypoints are normalised by the video size", () => {
  const kps = Array.from({ length: 21 }, (_, i) => ({ x: 32 * i, y: 24 * i, name: "k" }));
  const lm = handLandmarks(kps, 640, 480)!;
  expect(lm).toHaveLength(21);
  expect(lm[10]).toEqual({ x: 0.5, y: 0.5 });
  expect(handLandmarks(kps.slice(0, 20), 640, 480)).toBeNull();
  expect(handLandmarks(kps, 0, 480)).toBeNull();
  expect(handLandmarks(null, 640, 480)).toBeNull();
  // A pinch in pixels reads as a pinch after normalising.
  const pinch = Array.from({ length: 21 }, () => ({ x: 320, y: 300 }));
  pinch[0] = { x: 320, y: 440 };
  pinch[9] = { x: 320, y: 300 };
  pinch[4] = { x: 300, y: 250 };
  pinch[8] = { x: 302, y: 250 };
  expect(gestureName(handLandmarks(pinch, 640, 480))).toBe("pinch");
});

test("preview points are mirrored into the centred square", () => {
  expect(centerSquare(640, 480)).toEqual({ x0: 80, y0: 0, side: 480 });
  const pts = previewPoints([{ x: 0.5, y: 0.5 }, { x: 0.125, y: 0 }, { x: 0.875, y: 1 }], 640, 480);
  expect(pts[0]).toEqual({ x: 0.5, y: 0.5 });
  expect(pts[1]).toEqual({ x: 1, y: 0 });       // image-left edge of the square -> preview right
  expect(pts[2]).toEqual({ x: 0, y: 1 });
  expect(previewPoints(null, 640, 480)).toEqual([]);
});
