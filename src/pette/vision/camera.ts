/**
 * Webcam worker (port of pet/vision.py's `CameraWorker`): owns the camera stream and the tfjs hand and
 * face models, runs them at up to 15 fps and turns each frame into debounced `VisionEvent`s, scroll
 * amounts, ~30 s blink baselines and a small preview for the orb.
 *
 * Privacy: frames stay in memory. The preview (a mirrored 168px crop) is only handed to `onFrame` for the
 * orb to draw; nothing is stored, logged or sent. Video only: no audio is ever requested.
 */
import type { PreviewFrame, VisionEvent } from "../contracts";
import { blinkScores, centerSquare, handLandmarks, previewPoints, type PixelKeypoint } from "./landmarks";
import {
  BaselineMeter, gestureName, observation, pinchRatio, pinchY, pointing, VisionPolicy, type BaselineSummary,
} from "./policy";

export const MAX_FPS = 15;
export const PREVIEW_SIZE = 168;

/** The part of hand-pose-detection's `HandDetector` the worker uses. */
export interface HandEstimator {
  estimateHands(input: HTMLVideoElement): Promise<{ keypoints: PixelKeypoint[] }[]>;
  dispose(): void;
}

/** The part of face-landmarks-detection's `FaceLandmarksDetector` the worker uses. */
export interface FaceEstimator {
  estimateFaces(input: HTMLVideoElement): Promise<{ keypoints: PixelKeypoint[] }[]>;
  dispose(): void;
}

export interface Detectors { hands: HandEstimator; faces: FaceEstimator }

/** Cancels a scheduled callback. */
export type Cancel = () => void;

export interface CameraOptions {
  /** Per-user eye-closure threshold from the world model (null: the policy's default). */
  blinkOn?: number | null;
  /** Defaults to `navigator.mediaDevices.getUserMedia`. */
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  /** Defaults to `loadTfjsDetectors` (webgl backend, tfjs runtime). Called once per worker. */
  loadDetectors?: () => Promise<Detectors>;
  /** Defaults to a detached, muted `<video playsinline>`. */
  createVideo?: () => HTMLVideoElement;
  /** Mirrored square preview of the current frame; defaults to a canvas crop + `createImageBitmap`. */
  makeBitmap?: (video: HTMLVideoElement) => Promise<ImageBitmap | null>;
  /** Schedules the next frame callback; defaults to `requestVideoFrameCallback`, else `requestAnimationFrame`. */
  schedule?: (video: HTMLVideoElement, cb: () => void) => Cancel;
  /** Monotonic seconds. */
  clock?: () => number;
  /** True while the tab is hidden: frames are skipped. */
  isHidden?: () => boolean;
  maxFps?: number;
}

/** Loads both detectors with the tfjs runtime on the webgl backend (lazy imports keep them out of the main chunk). */
export async function loadTfjsDetectors(): Promise<Detectors> {
  const tf = await import("@tensorflow/tfjs");
  try { await tf.setBackend("webgl"); } catch { /* falls back to tfjs' default backend */ }
  await tf.ready();
  const handPose = await import("@tensorflow-models/hand-pose-detection");
  const faceMesh = await import("@tensorflow-models/face-landmarks-detection");
  const hands = await handPose.createDetector(handPose.SupportedModels.MediaPipeHands,
    { runtime: "tfjs", modelType: "lite", maxHands: 1 });
  const faces = await faceMesh.createDetector(faceMesh.SupportedModels.MediaPipeFaceMesh,
    { runtime: "tfjs", refineLandmarks: true, maxFaces: 1 });
  return { hands, faces };
}

function defaultVideo(): HTMLVideoElement {
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.autoplay = true;
  return v;
}

function defaultSchedule(video: HTMLVideoElement, cb: () => void): Cancel {
  if (typeof video.requestVideoFrameCallback === "function") {
    const id = video.requestVideoFrameCallback(() => cb());
    return () => video.cancelVideoFrameCallback(id);
  }
  const id = requestAnimationFrame(() => cb());
  return () => cancelAnimationFrame(id);
}

/** Mirrored centre-square crop, PREVIEW_SIZE px, drawn on one reused canvas. Null without a 2D context. */
function bitmapMaker(): (video: HTMLVideoElement) => Promise<ImageBitmap | null> {
  let canvas: HTMLCanvasElement | null = null;
  return async (video) => {
    if (typeof createImageBitmap !== "function") return null;
    canvas ??= document.createElement("canvas");
    canvas.width = canvas.height = PREVIEW_SIZE;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const { x0, y0, side } = centerSquare(video.videoWidth, video.videoHeight);
    ctx.setTransform(-1, 0, 0, 1, PREVIEW_SIZE, 0);   // mirror: the orb shows you like a mirror
    ctx.drawImage(video, x0, y0, side, side, 0, 0, PREVIEW_SIZE, PREVIEW_SIZE);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return createImageBitmap(canvas);
  };
}

function cameraError(e: unknown): string {
  const name = (e as { name?: string } | null)?.name;
  if (name === "NotAllowedError" || name === "SecurityError") return "Camera unavailable: permission denied.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "Camera unavailable: no camera found.";
  if (name === "NotReadableError") return "Camera unavailable: in use by another app.";
  return "Camera unavailable.";
}

/** Owns the camera and the models. Emits event names, scroll amounts, blink summaries and previews only. */
export class CameraWorker {
  onEvent: (ev: VisionEvent) => void = () => {};
  /** Preview for the orb; display only. */
  onFrame: (frame: PreviewFrame) => void = () => {};
  /** Pinch-drag: image heights moved since the last frame (+ = hand down). */
  onScroll: (dy: number) => void = () => {};
  /** Open-eye blink summary every ~30 s, for the world model. */
  onBaseline: (summary: BaselineSummary) => void = () => {};
  onError: (message: string) => void = () => {};
  /** Human-readable camera status ("Camera on (processed locally)", "Camera off"). */
  onStatus: (message: string) => void = () => {};

  private readonly opts: Required<Omit<CameraOptions, "blinkOn">> & { blinkOn: number | null };
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private detectors: Detectors | null = null;
  private policy: VisionPolicy;
  private meter = new BaselineMeter();
  private cancel: Cancel | null = null;
  private generation = 0;
  private busy = false;
  private lastFrame = -Infinity;

  constructor(options: CameraOptions = {}) {
    this.opts = {
      blinkOn: options.blinkOn ?? null,
      getUserMedia: options.getUserMedia ?? ((c) => navigator.mediaDevices.getUserMedia(c)),
      loadDetectors: options.loadDetectors ?? loadTfjsDetectors,
      createVideo: options.createVideo ?? defaultVideo,
      makeBitmap: options.makeBitmap ?? bitmapMaker(),
      schedule: options.schedule ?? defaultSchedule,
      clock: options.clock ?? (() => performance.now() / 1000),
      isHidden: options.isHidden ?? (() => typeof document !== "undefined" && document.hidden),
      maxFps: options.maxFps ?? MAX_FPS,
    };
    this.policy = new VisionPolicy(this.opts.blinkOn);
  }

  get running(): boolean { return this.stream !== null; }

  /** Asks for the camera (video only), loads the models and starts the frame loop. Errors go to `onError`. */
  async start(): Promise<void> {
    if (this.stream) return;
    const gen = ++this.generation;
    let stream: MediaStream;
    try {
      stream = await this.opts.getUserMedia({
        audio: false,
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: this.opts.maxFps }, facingMode: "user" },
      });
    } catch (e) {
      if (gen === this.generation) this.onError(cameraError(e));
      return;
    }
    if (gen !== this.generation) { stopTracks(stream); return; }   // stopped while asking
    this.stream = stream;
    try {
      this.detectors ??= await this.opts.loadDetectors();
      if (gen !== this.generation) return;
      const video = this.opts.createVideo();
      video.srcObject = stream;
      this.video = video;
      for (const track of stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (gen === this.generation) { this.onError("Camera disconnected."); this.stop(); }
        });
      }
      await video.play();
    } catch {
      if (gen === this.generation) { this.onError("Camera unavailable: could not start the models."); this.stop(); }
      return;
    }
    if (gen !== this.generation) return;
    this.policy = new VisionPolicy(this.opts.blinkOn);
    this.meter = new BaselineMeter();
    this.lastFrame = -Infinity;
    this.onStatus("Camera on (processed locally)");
    this.loop(gen);
  }

  /** Releases the camera at once (tracks stopped) and frees the models. */
  stop(): void {
    const wasOn = this.stream !== null;
    this.generation++;
    this.cancel?.();
    this.cancel = null;
    if (this.stream) stopTracks(this.stream);
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    this.video = null;
    if (!this.busy) this.disposeDetectors();   // otherwise after the in-flight frame
    if (wasOn) this.onStatus("Camera off");
  }

  private disposeDetectors(): void {
    if (!this.detectors) return;
    this.detectors.hands.dispose();
    this.detectors.faces.dispose();
    this.detectors = null;
  }

  private loop(gen: number): void {
    const video = this.video;
    if (!video || gen !== this.generation) return;
    this.cancel = this.opts.schedule(video, () => {
      if (gen !== this.generation) return;
      const t = this.opts.clock();
      if (!this.busy && !this.opts.isHidden() && t - this.lastFrame >= 1 / this.opts.maxFps) {
        this.lastFrame = t;
        void this.processFrame(t);
      }
      this.loop(gen);
    });
  }

  /** Runs the models on the current video frame and emits what it means. The loop calls this. */
  async processFrame(t: number = this.opts.clock()): Promise<void> {
    const video = this.video;
    const det = this.detectors;
    if (!video || !det || this.busy) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    const gen = this.generation;
    this.busy = true;
    try {
      const faces = await det.faces.estimateFaces(video);
      const hands = await det.hands.estimateHands(video);
      if (gen !== this.generation) return;
      const face = faces[0]?.keypoints ?? null;
      const [blinkLeft, blinkRight] = face ? blinkScores(face) : [0, 0];
      const lm = handLandmarks(hands[0]?.keypoints, w, h);
      const obs = observation(t, face !== null, {
        blinkLeft, blinkRight, pinchRatio: pinchRatio(lm), pinchY: pinchY(lm), point: pointing(lm),
      });
      for (const ev of this.policy.update(obs)) this.onEvent(ev);
      const dy = this.policy.takeScroll();
      if (dy !== 0) this.onScroll(dy);
      const summary = this.meter.add(obs, this.policy.eyesClosed);
      if (summary) this.onBaseline(summary);
      const bitmap = await this.opts.makeBitmap(video);
      if (gen !== this.generation) { bitmap?.close(); return; }
      this.onFrame({ bitmap, hands: lm ? [previewPoints(lm, w, h)] : [], gesture: gestureName(lm) });
    } catch {
      if (gen === this.generation) { this.onError("Camera stopped: the models failed on a frame."); this.stop(); }
    } finally {
      this.busy = false;
      if (gen !== this.generation && !this.stream) this.disposeDetectors();
    }
  }
}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) track.stop();
}
