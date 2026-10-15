/**
 * Webcam debounce rules (port of pet/vision.py's `VisionPolicy`): face presence, eye closure and hand
 * gestures turned into debounced events. Pure logic with time passed in, so it is tested without a camera.
 *
 * Gestures:
 *   pinch (thumb to index) held still ~0.5 s   -> "pinch" ("Do it" on a card)
 *   pinch, then move the hand                  -> "drag_start", scroll amounts, "drag_end"
 *   index finger pointing to your left / right -> "point_left" / "point_right"
 * Left and right are from the user's point of view, the same as in the mirrored preview on the orb.
 * Eye closure is a heuristic, not gaze, attention or emotion measurement.
 */
import type { VisionEvent } from "../contracts";

/** A hand landmark in camera-image coordinates (0..1, x to the image's right, y down). */
export interface Landmark { x: number; y: number }

/** "left" / "right" from the user's point of view. */
export type PointSide = "left" | "right";

/** One processed frame. Hand fields are null when no usable hand was found. */
export interface Observation {
  /** Monotonic seconds. */
  t: number;
  face: boolean;
  blinkLeft: number;
  blinkRight: number;
  pinchRatio: number | null;
  /** Vertical position of the pinch (0 top .. 1 bottom of the camera image). */
  pinchY: number | null;
  /** Set while the index finger points sideways. */
  point: PointSide | null;
}

/** Builds an `Observation` with Python's defaults (no blink, no hand). */
export function observation(t: number, face: boolean, rest: Partial<Omit<Observation, "t" | "face">> = {}): Observation {
  return { t, face, blinkLeft: 0, blinkRight: 0, pinchRatio: null, pinchY: null, point: null, ...rest };
}

export const AWAY_AFTER_S = 3.0;
export const BACK_AFTER_S = 1.0;
export const BLINK_ON = 0.6;
export const BLINK_OFF = 0.4;
export const BLINK_HOLD_S = 0.3;
export const PINCH_ON = 0.25;
export const PINCH_REARM = 0.4;
export const PINCH_HOLD_S = 0.5;
export const PINCH_COOLDOWN_S = 2.0;
/** Pinch moves this far (fraction of image height) -> it's a drag, not a pinch. */
export const DRAG_START_DY = 0.04;
/** EMA on the pinch position; less jitter in the scroll. */
export const DRAG_SMOOTH = 0.5;
export const POINT_HOLD_S = 0.25;
export const POINT_COOLDOWN_S = 0.6;

export class VisionPolicy {
  readonly blinkOn: number;
  readonly blinkOff: number;
  present = true;
  eyesClosed = false;
  dragging = false;
  private faceChangeSince: number | null = null;
  private blinkSince: number | null = null;
  private pinchSince: number | null = null;
  private pinchArmed = true;
  private lastPinch = -Infinity;
  /** Where this pinch began (smoothed). */
  private pinchStartY: number | null = null;
  /** Smoothed current pinch position. */
  private pinchYs: number | null = null;
  /** Pending scroll (image heights; + = hand moved down). */
  private pending = 0;
  private point: PointSide | null = null;
  private pointSince: number | null = null;
  private pointArmed = true;
  private lastPoint = -Infinity;

  /** `blinkOn` is calibrated per user by the world model; reopen stays 0.2 below (at least 0.2). */
  constructor(blinkOn?: number | null) {
    if (blinkOn != null) {
      this.blinkOn = blinkOn;
      this.blinkOff = Math.max(0.2, blinkOn - 0.2);
    } else {
      this.blinkOn = BLINK_ON;
      this.blinkOff = BLINK_OFF;
    }
  }

  /** Scroll accumulated since the last call, in image heights (+ = the hand moved down). */
  takeScroll(): number {
    const dy = this.pending;
    this.pending = 0;
    return dy;
  }

  update(o: Observation): VisionEvent[] {
    const events: VisionEvent[] = [];

    // Face presence with hysteresis in time.
    if (o.face !== this.present) {
      if (this.faceChangeSince === null) this.faceChangeSince = o.t;
      const need = o.face ? BACK_AFTER_S : AWAY_AFTER_S;
      if (o.t - this.faceChangeSince >= need) {
        this.present = o.face;
        this.faceChangeSince = null;
        events.push(o.face ? "face_back" : "face_away");
        if (!o.face && this.eyesClosed) {
          this.eyesClosed = false;
          this.blinkSince = null;
        }
      }
    } else {
      this.faceChangeSince = null;
    }

    // Eye closure: both scores high for a hold time; reopen below a lower threshold.
    if (o.face) {
      if (!this.eyesClosed) {
        if (Math.min(o.blinkLeft, o.blinkRight) > this.blinkOn) {
          if (this.blinkSince === null) this.blinkSince = o.t;
          if (o.t - this.blinkSince >= BLINK_HOLD_S) {
            this.eyesClosed = true;
            events.push("eyes_closed");
          }
        } else {
          this.blinkSince = null;
        }
      } else if (Math.max(o.blinkLeft, o.blinkRight) < this.blinkOff) {
        this.eyesClosed = false;
        this.blinkSince = null;
        events.push("eyes_open");
      }
    }

    events.push(...this.pinchAndDrag(o), ...this.pointing(o));
    return events;
  }

  private endDrag(): VisionEvent[] {
    const was = this.dragging;
    this.dragging = false;
    this.pinchStartY = this.pinchYs = null;
    return was ? ["drag_end"] : [];
  }

  /** Pinch held still -> "pinch" (needs rearm + cooldown). Pinch that moves -> a drag that scrolls. */
  private pinchAndDrag(o: Observation): VisionEvent[] {
    const r = o.pinchRatio;
    if (r === null || r > PINCH_REARM) {   // hand gone or fingers apart: pinch / drag over
      if (r !== null) this.pinchArmed = true;
      this.pinchSince = null;
      return this.endDrag();
    }
    if (r >= PINCH_ON) return [];           // between thresholds: keep whatever is going on
    const events: VisionEvent[] = [];
    if (o.pinchY !== null) {
      const prev = this.pinchYs;
      const y = prev === null ? o.pinchY : prev + DRAG_SMOOTH * (o.pinchY - prev);
      this.pinchYs = y;
      if (this.pinchStartY === null) this.pinchStartY = y;
      if (this.dragging) {
        this.pending += y - (prev ?? y);
      } else if (Math.abs(y - this.pinchStartY) >= DRAG_START_DY) {
        this.dragging = true;
        this.pinchSince = null;              // a drag is never also a pinch
        this.pending += y - this.pinchStartY;
        events.push("drag_start");
      }
    }
    if (this.dragging || !this.pinchArmed) return events;
    if (this.pinchSince === null) this.pinchSince = o.t;
    if (o.t - this.pinchSince >= PINCH_HOLD_S && o.t - this.lastPinch >= PINCH_COOLDOWN_S) {
      events.push("pinch");
      this.lastPinch = o.t;
      this.pinchArmed = false;
      this.pinchSince = null;
    }
    return events;
  }

  /** Point held briefly -> one "point_left" / "point_right"; relax (or switch side) to fire again. */
  private pointing(o: Observation): VisionEvent[] {
    const d = this.dragging ? null : o.point;
    if (d === null) {
      this.point = null;
      this.pointSince = null;
      this.pointArmed = true;
      return [];
    }
    if (d !== this.point || this.pointSince === null) {
      this.point = d;
      this.pointSince = o.t;
      this.pointArmed = true;
    }
    if (this.pointArmed && o.t - this.pointSince >= POINT_HOLD_S && o.t - this.lastPoint >= POINT_COOLDOWN_S) {
      this.pointArmed = false;
      this.lastPoint = o.t;
      return [d === "left" ? "point_left" : "point_right"];
    }
    return [];
  }
}

const dist = (a: Landmark, b: Landmark): number => Math.hypot(a.x - b.x, a.y - b.y);

/** Thumb tip (4) to index tip (8) distance, normalised by wrist (0) to middle knuckle (9). */
export function pinchRatio(lm: readonly Landmark[] | null | undefined): number | null {
  if (!lm || lm.length < 10) return null;
  const scale = dist(lm[0], lm[9]);
  if (scale < 1e-3) return null;
  return dist(lm[4], lm[8]) / scale;
}

/** Vertical position of the pinch: the midpoint of the thumb and index tips (0 top .. 1 bottom). */
export function pinchY(lm: readonly Landmark[] | null | undefined): number | null {
  if (!lm || lm.length < 9) return null;
  return (lm[4].y + lm[8].y) / 2;
}

/**
 * "left" / "right" (your point of view) when only the index finger is out and it points sideways.
 *
 * The camera sees you face to face, so pointing to your left moves the fingertip to the image's right
 * (+x). Other fingers must be curled (tips close to the wrist) so an open hand never counts.
 */
export function pointing(lm: readonly Landmark[] | null | undefined): PointSide | null {
  if (!lm || lm.length < 21) return null;
  const wrist = lm[0];
  const scale = dist(lm[0], lm[9]);
  if (scale < 1e-3) return null;
  const vx = lm[8].x - lm[5].x;
  const vy = lm[8].y - lm[5].y;
  if (Math.hypot(vx, vy) < 0.6 * scale || Math.abs(vx) < 1.6 * Math.abs(vy)) return null; // not stretched sideways
  for (const [tip, mcp] of [[12, 9], [16, 13], [20, 17]] as const) {   // middle, ring, pinky curled
    if (dist(lm[tip], wrist) > dist(lm[mcp], wrist) * 1.25) return null;
  }
  return vx > 0 ? "left" : "right";
}

/** Orb label: "pinch", "point left", "point right", or "" for anything else. */
export function gestureName(lm: readonly Landmark[] | null | undefined): string {
  if (!lm || lm.length < 21) return "";
  const r = pinchRatio(lm);
  if (r !== null && r < PINCH_ON) return "pinch";
  const p = pointing(lm);
  return p ? `point ${p}` : "";
}

/** Open-eye blink-score summary (numbers only) for threshold calibration. */
export interface BaselineSummary { mean: number; p95: number; samples: number }

const round3 = (x: number): number => Math.round(x * 1000) / 1000;

/** Summarises open-eye blink scores over a window (numbers only) for threshold calibration. */
export class BaselineMeter {
  static readonly WINDOW_S = 30;
  private samples: number[] = [];
  private start: number | null = null;

  add(o: Observation, eyesClosed: boolean): BaselineSummary | null {
    if (this.start === null) this.start = o.t;
    if (o.face && !eyesClosed) this.samples.push(Math.max(o.blinkLeft, o.blinkRight));
    if (o.t - this.start < BaselineMeter.WINDOW_S) return null;
    const s = this.samples.sort((a, b) => a - b);
    this.samples = [];
    this.start = o.t;
    if (s.length < 20) return null;
    const mean = s.reduce((a, b) => a + b, 0) / s.length;
    // Python: s[int(len * 0.95) - 1]
    return { mean: round3(mean), p95: round3(s[Math.floor(s.length * 0.95) - 1]), samples: s.length };
  }
}
