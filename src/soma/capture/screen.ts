/**
 * Consent-gated, rate-limited, change-detecting screen capture (port of soma/capture/screen.py).
 *
 * Consent is the browser's own: `grant()` calls `getDisplayMedia`, which shows the picker, and the user can
 * stop sharing at any time from the browser UI (`onEnded` fires and the stream is released). Pette's own tab
 * is excluded with `selfBrowserSurface: "exclude"`. Frames live only in memory; nothing is written anywhere.
 * Change detection keeps a 64x64 grey thumbnail of the last meaningful frame, never the frame itself.
 */
import type { CaptureConfig } from "../config";
import { changeRatioThumb, thumb } from "../perception/preprocess";

/** Skip reasons reported through `ScreenCapturer.lastSkipReason`. */
export const SKIP_NO_CONSENT = "no-consent";
export const SKIP_RATE_LIMITED = "rate-limited";
export const SKIP_ERROR = "grab-error";

/** One captured frame. `changed` is false when the screen barely moved since the last meaningful frame. */
export interface CapturedFrame {
  image: ImageData;
  /** "monitor" | "window" | "browser" | "" (`MediaTrackSettings.displaySurface`). */
  surface: string;
  /** The track label; Chrome labels a shared window with its title. Untrusted text. */
  trackLabel: string;
  /** Version of the last meaningful frame (increments only when `changed`). */
  version: number;
  changed: boolean;
}

export interface ScreenCapturerDeps {
  mediaDevices?: Pick<MediaDevices, "getDisplayMedia">;
  /** Milliseconds, injectable for tests. */
  now?: () => number;
  /** Grab one RGBA frame from the stream. Defaults to ImageCapture.grabFrame or a <video> + canvas. */
  grabFrame?: (stream: MediaStream) => Promise<ImageData>;
}

/** Frames whose longest side exceeds this are downscaled when grabbed (OCR downsamples further anyway). */
export const MAX_GRAB_SIDE = 2560;

/** Draw a video frame / bitmap into RGBA pixels, downscaling very large screens. */
function toImageData(source: CanvasImageSource, w: number, h: number): ImageData {
  const scale = Math.min(1, MAX_GRAB_SIDE / Math.max(w, h));
  const cw = Math.max(1, Math.round(w * scale));
  const ch = Math.max(1, Math.round(h * scale));
  const canvas: OffscreenCanvas | HTMLCanvasElement =
    typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(cw, ch) : Object.assign(document.createElement("canvas"), { width: cw, height: ch });
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!ctx) throw new Error("no 2d canvas");
  ctx.drawImage(source, 0, 0, cw, ch);
  return ctx.getImageData(0, 0, cw, ch);
}

interface ImageCaptureLike { grabFrame(): Promise<ImageBitmap> }
type ImageCaptureCtor = new (track: MediaStreamTrack) => ImageCaptureLike;

/** Default frame grabber: ImageCapture.grabFrame where available, else a muted <video> drawn to a canvas. */
export function makeDefaultGrabber(): (stream: MediaStream) => Promise<ImageData> {
  let video: HTMLVideoElement | null = null;
  let videoStream: MediaStream | null = null;
  return async (stream) => {
    const track = stream.getVideoTracks()[0];
    if (!track) throw new Error("no video track");
    const IC = (globalThis as unknown as { ImageCapture?: ImageCaptureCtor }).ImageCapture;
    if (IC) {
      const bmp = await new IC(track).grabFrame();
      try {
        return toImageData(bmp, bmp.width, bmp.height);
      } finally {
        bmp.close();
      }
    }
    if (!video || videoStream !== stream) {
      video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      videoStream = stream;
      await video.play();
    }
    if (!video.videoWidth) await new Promise((r) => setTimeout(r, 100));
    return toImageData(video, video.videoWidth, video.videoHeight);
  };
}

export class ScreenCapturer {
  /** Called when the user stops sharing from the browser UI (the stream is already released). */
  onEnded?: () => void;
  lastSkipReason = "";
  lastChangeRatio = 0;

  private readonly cfg: CaptureConfig;
  private readonly media?: Pick<MediaDevices, "getDisplayMedia">;
  private readonly now: () => number;
  private readonly grab: (stream: MediaStream) => Promise<ImageData>;
  private stream: MediaStream | null = null;
  private versionN = 0;
  private lastGrabT: number | null = null;
  private lastThumb: { values: Float32Array; size: number } | null = null;
  private lastSize: [number, number] | null = null;
  private lastLabel: string | null = null;

  constructor(cfg: CaptureConfig, deps: ScreenCapturerDeps = {}) {
    this.cfg = cfg;
    this.media = deps.mediaDevices ?? globalThis.navigator?.mediaDevices;
    this.now = deps.now ?? (() => performance.now());
    this.grab = deps.grabFrame ?? makeDefaultGrabber();
  }

  /** True while a live stream is held. */
  get active(): boolean {
    const track = this.stream?.getVideoTracks()[0];
    return !!track && track.readyState !== "ended";
  }

  /** Version of the last meaningful frame (0 before the first one). */
  get version(): number {
    return this.versionN;
  }

  /** Ask the user to share a screen, window or tab. Resolves false when they decline or it is unsupported. */
  async grant(): Promise<boolean> {
    if (this.active) return true;
    if (!this.media?.getDisplayMedia) return false;
    const options = {
      video: { frameRate: { ideal: 2, max: 5 } },
      audio: false,
      selfBrowserSurface: "exclude",
      surfaceSwitching: "include",
      systemAudio: "exclude",
      monitorTypeSurfaces: "include",
      preferCurrentTab: false,
    } as DisplayMediaStreamOptions;
    let stream: MediaStream;
    try {
      stream = await this.media.getDisplayMedia(options);
    } catch {
      return false;
    }
    this.stream = stream;
    this.reset();
    for (const t of stream.getTracks()) {
      t.addEventListener("ended", () => {
        if (this.stream !== stream) return;
        this.revoke();
        this.onEnded?.();
      });
    }
    return true;
  }

  /** Stop capturing and release every track. Forgets the change-detection reference. */
  revoke(): void {
    const s = this.stream;
    this.stream = null;
    s?.getTracks().forEach((t) => t.stop());
    this.reset();
  }

  /** Forget the change-detection reference and rate limit (e.g. after resume). */
  reset(): void {
    this.lastGrabT = null;
    this.lastThumb = null;
    this.lastSize = null;
    this.lastLabel = null;
  }

  /**
   * Grab a frame, or null with `lastSkipReason` set (no stream, rate-limited, grab error). `force` bypasses
   * the rate limit and change detection, never the consent gate.
   */
  async capture(force = false): Promise<CapturedFrame | null> {
    const stream = this.stream;
    if (!stream || !this.active) {
      this.lastSkipReason = SKIP_NO_CONSENT;
      return null;
    }
    const t = this.now();
    if (!force && this.lastGrabT !== null && t - this.lastGrabT < this.cfg.minIntervalS * 1000) {
      this.lastSkipReason = SKIP_RATE_LIMITED;
      return null;
    }
    this.lastGrabT = t;
    const track = stream.getVideoTracks()[0];
    const surface = String((track.getSettings() as MediaTrackSettings & { displaySurface?: string }).displaySurface ?? "");
    const trackLabel = track.label ?? "";
    let image: ImageData;
    try {
      image = await this.grab(stream);
    } catch (err) {
      this.lastSkipReason = `${SKIP_ERROR}:${(err as Error)?.name ?? "Error"}`;
      return null;
    }
    if (this.stream !== stream) {
      // revoked while grabbing: drop the frame
      this.lastSkipReason = SKIP_NO_CONSENT;
      return null;
    }
    if (!image || image.width <= 0 || image.height <= 0 || image.data.length < image.width * image.height * 4) {
      this.lastSkipReason = `${SKIP_ERROR}:bad-shape`;
      return null;
    }
    const th = thumb(image, 64);
    const sameSource = trackLabel === this.lastLabel && this.lastSize !== null
      && this.lastSize[0] === image.width && this.lastSize[1] === image.height;
    const ratio = sameSource ? changeRatioThumb(this.lastThumb, th) : 1;
    this.lastChangeRatio = ratio;
    const changed = force || !sameSource || ratio >= this.cfg.changeThreshold;
    if (changed) {
      this.versionN += 1;
      this.lastThumb = th;
      this.lastSize = [image.width, image.height];
      this.lastLabel = trackLabel;
    }
    this.lastSkipReason = "";
    return { image, surface, trackLabel, version: this.versionN, changed };
  }
}
