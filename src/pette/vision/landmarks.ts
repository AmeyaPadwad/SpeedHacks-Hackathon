/**
 * Adapters from the tfjs models' outputs to what `VisionPolicy` needs. Kept free of tfjs imports: the
 * keypoint shape below is structurally the models' `Keypoint` (pixel coordinates in the input frame).
 */
import type { PreviewPoint } from "../contracts";
import type { Landmark } from "./policy";

/** A model keypoint in input-frame pixels (hand-pose-detection / face-landmarks-detection `Keypoint`). */
export interface PixelKeypoint { x: number; y: number }

/**
 * Hand-pose-detection's 21 keypoints (pixels) -> landmarks normalised by the video size (0..1, image
 * coordinates, not mirrored), the same list format MediaPipe's HandLandmarker gave the Python port.
 * Null when there is no complete hand or the frame size is unknown.
 */
export function handLandmarks(keypoints: readonly PixelKeypoint[] | null | undefined, width: number,
  height: number): Landmark[] | null {
  if (!keypoints || keypoints.length < 21 || width <= 0 || height <= 0) return null;
  return keypoints.slice(0, 21).map((k) => ({ x: k.x / width, y: k.y / height }));
}

/**
 * MediaPipe FaceMesh eyelid keypoints: two upper/lower lid pairs and the two corners per eye.
 * "left" / "right" are the person's own eyes, as in the blendshapes `eyeBlinkLeft` / `eyeBlinkRight`.
 */
export const EYE_KEYPOINTS = {
  left: { corners: [362, 263], lids: [[385, 380], [387, 373]] },
  right: { corners: [33, 133], lids: [[160, 144], [158, 153]] },
} as const;

/** Eye aspect ratio at which the eye counts as fully open (blink score 0). */
export const EAR_OPEN = 0.28;
/** Eye aspect ratio at which the eye counts as fully closed (blink score 1). */
export const EAR_CLOSED = 0.1;

const dist = (a: PixelKeypoint, b: PixelKeypoint): number => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Eye aspect ratio (Soukupová & Čech): mean lid gap over corner-to-corner width. Scale invariant, so it
 * works on pixel keypoints directly (normalising x and y by different sizes would distort it).
 * Null when the keypoints are missing or the eye has no width.
 */
export function eyeAspectRatio(keypoints: readonly PixelKeypoint[], eye: "left" | "right"): number | null {
  const { corners, lids } = EYE_KEYPOINTS[eye];
  const need = Math.max(...corners, ...lids.flat());
  if (keypoints.length <= need) return null;
  const width = dist(keypoints[corners[0]], keypoints[corners[1]]);
  if (width < 1e-6) return null;
  const gap = lids.reduce((sum, [up, down]) => sum + dist(keypoints[up], keypoints[down]), 0) / lids.length;
  return gap / width;
}

/**
 * Maps an eye aspect ratio to a 0..1 "blink" score comparable to MediaPipe's eyeBlink blendshape
 * (which the tfjs face model doesn't output): linear from EAR_OPEN (open, 0) down to EAR_CLOSED
 * (closed, 1), clamped. With the policy's thresholds, "closed" (0.6) is EAR <= 0.172 and "open again"
 * (0.4) is EAR >= 0.208; a relaxed open eye sits around 0.25..0.32, a shut one around 0.05..0.12.
 */
export function earToBlink(ear: number): number {
  return Math.min(1, Math.max(0, (EAR_OPEN - ear) / (EAR_OPEN - EAR_CLOSED)));
}

/** [blinkLeft, blinkRight] from FaceMesh keypoints (468 or, refined, 478); [0, 0] when unusable. */
export function blinkScores(keypoints: readonly PixelKeypoint[] | null | undefined): [number, number] {
  if (!keypoints) return [0, 0];
  const l = eyeAspectRatio(keypoints, "left");
  const r = eyeAspectRatio(keypoints, "right");
  return [l === null ? 0 : earToBlink(l), r === null ? 0 : earToBlink(r)];
}

/** The centred square of a `width` x `height` frame: where the orb's crop comes from. */
export function centerSquare(width: number, height: number): { x0: number; y0: number; side: number } {
  const side = Math.min(width, height);
  return { x0: Math.floor((width - side) / 2), y0: Math.floor((height - side) / 2), side };
}

/**
 * Hand landmarks (normalised image coordinates) -> points in the mirrored square preview (0..1, x grows
 * to the user's right), matching the crop drawn on the orb. Points outside the square fall outside 0..1.
 */
export function previewPoints(lm: readonly Landmark[] | null | undefined, width: number,
  height: number): PreviewPoint[] {
  if (!lm || width <= 0 || height <= 0) return [];
  const { x0, y0, side } = centerSquare(width, height);
  return lm.map((p) => ({ x: 1 - (p.x * width - x0) / side, y: (p.y * height - y0) / side }));
}
