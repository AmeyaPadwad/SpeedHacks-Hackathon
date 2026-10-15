/**
 * Webcam presence and gestures (port of pet/watcher.py): owns the optional camera worker and turns its
 * events into reactions on a `WatcherTarget`.
 *
 * Face away -> Pette dozes; back -> an open card is brought to attention (`nudge`); eyes closed -> she
 * shuts hers; pinch -> "Do it" on a card; point left / right -> `step(-1 | +1)`; pinch-and-drag -> `scroll`.
 * Preview frames go straight to the orb. Nothing is uploaded or stored apart from event names and timings.
 * Paused means no reactions and no logging.
 */
import type { FactLog, PreviewFrame, VisionEvent, WatcherTarget } from "../contracts";
import { CameraWorker } from "./camera";
import type { BaselineSummary } from "./policy";

/** The part of `CameraWorker` the watcher drives (tests pass a fake). */
export interface CameraLike {
  onEvent: (ev: VisionEvent) => void;
  onFrame: (frame: PreviewFrame) => void;
  onScroll: (dy: number) => void;
  onBaseline: (summary: BaselineSummary) => void;
  onError: (message: string) => void;
  onStatus: (message: string) => void;
  readonly running: boolean;
  start(): Promise<void>;
  stop(): void;
}

export interface WatcherOptions {
  factLog?: FactLog | null;
  /** Monotonic seconds. */
  clock?: () => number;
  /** Per-user eye-closure threshold (the world model's `blinkThreshold()`), read each time the camera starts. */
  blinkOn?: () => number | null;
  createCamera?: (blinkOn: number | null) => CameraLike;
  /** The camera indicator: true while the camera is on. */
  onCameraChange?: (on: boolean) => void;
  /** Human-readable camera status and errors for the UI. */
  onStatus?: (message: string) => void;
}

export class Watcher {
  present = true;
  eyesClosed = false;
  private camera: CameraLike | null = null;
  private awaySince: number | null = null;
  private closedSince: number | null = null;
  private readonly factLog: FactLog | null;
  private readonly clock: () => number;
  private readonly opts: WatcherOptions;

  constructor(private readonly target: WatcherTarget, options: WatcherOptions = {}) {
    this.opts = options;
    this.factLog = options.factLog ?? null;
    this.clock = options.clock ?? (() => performance.now() / 1000);
  }

  get cameraOn(): boolean { return this.camera !== null; }

  // --- lifecycle ----------------------------------------------------------
  stop(): void { this.setCamera(false); }

  setCamera(on: boolean): void {
    if (on && this.camera === null) {
      const blinkOn = this.opts.blinkOn?.() ?? null;
      const cam = this.opts.createCamera?.(blinkOn) ?? new CameraWorker({ blinkOn });
      this.camera = cam;
      cam.onEvent = (ev) => { if (this.camera === cam) this.onVisionEvent(ev); };
      cam.onFrame = (f) => { if (this.camera === cam) this.onFrame(f); else f.bitmap?.close(); };
      cam.onScroll = (dy) => { if (this.camera === cam) this.onScroll(dy); };
      cam.onBaseline = (s) => { if (this.camera === cam) this.onBaseline(s); };
      cam.onStatus = (m) => this.opts.onStatus?.(m);
      cam.onError = (m) => {
        this.opts.onStatus?.(m);
        if (this.camera === cam && !cam.running) this.cameraFinished();
      };
      void cam.start();
    } else if (!on && this.camera !== null) {
      const cam = this.camera;
      this.camera = null;
      cam.stop();
      this.cameraFinished();
    }
    this.opts.onCameraChange?.(this.camera !== null);
  }

  private cameraFinished(): void {
    if (this.camera !== null && !this.camera.running) {
      this.camera = null;
      this.opts.onCameraChange?.(false);
    }
    this.present = true;
    this.eyesClosed = false;
    this.awaySince = this.closedSince = null;
    this.target.setPresence(false, false);
  }

  // --- inputs -------------------------------------------------------------
  /** A preview for the orb; dropped (and freed) while paused or with the camera off. */
  onFrame(frame: PreviewFrame): void {
    if (this.camera !== null && !this.target.paused) this.target.setFrame(frame);
    else frame.bitmap?.close();
  }

  /** Pinch-and-drag movement (image heights, + = hand down). */
  onScroll(dy: number): void {
    if (!this.target.paused) this.target.scroll(dy);
  }

  onVisionEvent(ev: VisionEvent): void {
    if (this.target.paused) return;   // paused means no reactions and no logging
    if (ev === "point_left" || ev === "point_right" || ev === "drag_start" || ev === "drag_end") {
      this.log(ev);
      if (ev === "point_left") this.target.step(-1);         // your left = previous
      else if (ev === "point_right") this.target.step(1);
      return;
    }
    const now = this.clock();
    this.log(ev);
    switch (ev) {
      case "face_away":
        this.present = false;
        this.awaySince = now;
        break;
      case "face_back":
        this.present = true;
        if (this.awaySince !== null) {
          this.log("away_ended", round(now - this.awaySince, 1));
          this.awaySince = null;
        }
        this.target.nudge();               // a card that waited while you were away comes forward
        break;
      case "eyes_closed":
        this.eyesClosed = true;
        this.closedSince = now;
        break;
      case "eyes_open":
        this.eyesClosed = false;
        if (this.closedSince !== null) {
          this.log("eyes_closed_ended", round(now - this.closedSince, 2));
          this.closedSince = null;
        }
        break;
      case "pinch":
        this.target.pinch();               // a card is up: Do it; otherwise ignored
        return;
    }
    this.target.setPresence(!this.present, this.eyesClosed);
  }

  onBaseline(summary: BaselineSummary): void {
    this.factLog?.logBlinkBaseline(summary.mean, summary.p95, summary.samples);
  }

  private log(kind: string, duration: number | null = null): void {
    this.factLog?.logEvent(kind, duration);
  }
}

function round(x: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}
