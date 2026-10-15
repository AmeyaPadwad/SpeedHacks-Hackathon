/**
 * OCR engines producing normalized, redacted `OCRElement` lists (port of soma/perception/ocr.py).
 *
 * - `TesseractOcr` runs tesseract.js v5 in its own web worker (lazily created, English), one element per
 *   recognized line.
 * - `FakeOcr` returns fixed elements (tests, demos); `NullOcr` returns nothing.
 * - `FallbackOcr` swallows backend errors so a broken OCR never crashes the step loop.
 *
 * Every engine passes text through `redactText` and drops empty lines before returning.
 */
import { redactText } from "../capture/redaction";
import type { OCRElement } from "../types";

export interface OcrEngine {
  /** Recognize text lines in RGBA pixels. Boxes are normalized xyxy in [0, 1]. */
  recognize(image: ImageData, windowId: string): Promise<OCRElement[]>;
  dispose(): Promise<void>;
}

const clip01 = (v: number): number => Math.min(1, Math.max(0, v));

/** Redact text and drop empty elements. */
export function finalizeElements(elements: readonly OCRElement[]): OCRElement[] {
  const out: OCRElement[] = [];
  for (const e of elements) {
    const text = redactText(e.text).trim();
    if (text) out.push({ ...e, text });
  }
  return out;
}

/** OCR disabled: always `[]`. */
export class NullOcr implements OcrEngine {
  async recognize(): Promise<OCRElement[]> {
    return [];
  }
  async dispose(): Promise<void> {}
}

/** Returns copies of fixed elements (re-stamped with the window id and time), redacted like real OCR. */
export class FakeOcr implements OcrEngine {
  calls = 0;
  constructor(public elements: Array<Partial<OCRElement> & { text: string }> = []) {}
  async recognize(_image: ImageData, windowId: string): Promise<OCRElement[]> {
    this.calls += 1;
    const ts = Date.now();
    return finalizeElements(this.elements.map((e) => ({
      bbox: [0, 0, 1, 1] as [number, number, number, number], confidence: 1, source: "ocr" as const, uiRole: null,
      ...e, text: e.text, windowId, timestampMs: ts,
    })));
  }
  async dispose(): Promise<void> {}
}

/** Uses `primary`, returning `[]` (with `lastError` set) when it throws. */
export class FallbackOcr implements OcrEngine {
  lastError = "";
  constructor(public primary: OcrEngine) {}
  async recognize(image: ImageData, windowId: string): Promise<OCRElement[]> {
    try {
      this.lastError = "";
      return await this.primary.recognize(image, windowId);
    } catch (err) {
      this.lastError = `${(err as Error)?.name ?? "Error"}: ${(err as Error)?.message ?? String(err)}`;
      return [];
    }
  }
  dispose(): Promise<void> {
    return this.primary.dispose();
  }
}

// ----------------------------------------------------------------- tesseract.js

interface TessBbox { x0: number; y0: number; x1: number; y1: number }
interface TessLine { text: string; confidence: number; bbox: TessBbox }
interface TessBlock { paragraphs?: { lines?: TessLine[] }[]; lines?: TessLine[] }
/** The slice of a tesseract.js v5 worker used here (injectable for tests). */
export interface TessWorkerLike {
  recognize(image: unknown, options?: object, output?: object): Promise<{ data: { blocks?: TessBlock[] | null; lines?: TessLine[] } }>;
  terminate(): Promise<unknown>;
}

/** What gets handed to tesseract: a drawable source and its pixel size (after any downscale). */
export interface PreparedImage { source: unknown; width: number; height: number }

export interface TesseractOcrOptions {
  /** Drop lines below this confidence (0..1). */
  minConfidence?: number;
  /** Downscale images whose longest side exceeds this (speed). */
  maxSide?: number;
  createWorker?: () => Promise<TessWorkerLike>;
  prepare?: (image: ImageData, maxSide: number) => PreparedImage;
}

/** Draw `image` onto a canvas (downscaled to `maxSide`), which tesseract.js reads in every browser. */
export function prepareOnCanvas(image: ImageData, maxSide: number): PreparedImage {
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const full = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(image.width, image.height)
    : Object.assign(document.createElement("canvas"), { width: image.width, height: image.height });
  const fctx = full.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!fctx) throw new Error("no 2d canvas");
  fctx.putImageData(image, 0, 0);
  if (scale >= 1) return { source: full, width: image.width, height: image.height };
  const w = Math.max(1, Math.round(image.width * scale));
  const h = Math.max(1, Math.round(image.height * scale));
  const small = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement("canvas"), { width: w, height: h });
  const sctx = small.getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!sctx) throw new Error("no 2d canvas");
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(full, 0, 0, w, h);
  return { source: small, width: w, height: h };
}

async function defaultWorker(): Promise<TessWorkerLike> {
  const { createWorker } = await import("tesseract.js");
  return (await createWorker("eng")) as unknown as TessWorkerLike;
}

/** tesseract.js adapter: lazily created worker, calls serialized, line-level elements. */
export class TesseractOcr implements OcrEngine {
  readonly minConfidence: number;
  readonly maxSide: number;
  private readonly create: () => Promise<TessWorkerLike>;
  private readonly prepare: (image: ImageData, maxSide: number) => PreparedImage;
  private worker: Promise<TessWorkerLike> | null = null;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(opts: TesseractOcrOptions = {}) {
    this.minConfidence = opts.minConfidence ?? 0.5;
    this.maxSide = opts.maxSide ?? 2048;
    this.create = opts.createWorker ?? defaultWorker;
    this.prepare = opts.prepare ?? prepareOnCanvas;
  }

  /** Start loading the worker and language data ahead of the first frame. */
  warmUp(): Promise<void> {
    return this.load().then(() => undefined);
  }

  private load(): Promise<TessWorkerLike> {
    if (!this.worker) {
      this.worker = this.create();
      this.worker.catch(() => { this.worker = null; });
    }
    return this.worker;
  }

  recognize(image: ImageData, windowId: string): Promise<OCRElement[]> {
    const run = this.queue.then(() => this.run(image, windowId));
    this.queue = run.catch(() => undefined);
    return run;
  }

  private async run(image: ImageData, windowId: string): Promise<OCRElement[]> {
    if (!image || image.width <= 0 || image.height <= 0) return [];
    const prepared = this.prepare(image, this.maxSide);
    const worker = await this.load();
    const ts = Date.now();
    const { data } = await worker.recognize(prepared.source, {}, { blocks: true, text: false });
    const lines: TessLine[] = data.blocks
      ? data.blocks.flatMap((b) => b.paragraphs?.flatMap((p) => p.lines ?? []) ?? b.lines ?? [])
      : data.lines ?? [];
    const { width: sw, height: sh } = prepared;
    const out: OCRElement[] = [];
    for (const l of lines) {
      const conf = (l.confidence ?? 0) / 100;
      if (conf < this.minConfidence) continue;
      const b = l.bbox;
      out.push({
        text: String(l.text ?? "").replace(/\s+/g, " "),
        bbox: [clip01(b.x0 / sw), clip01(b.y0 / sh), clip01(b.x1 / sw), clip01(b.y1 / sh)],
        confidence: clip01(conf), source: "ocr", uiRole: null, windowId, timestampMs: ts,
      });
    }
    return finalizeElements(out);
  }

  async dispose(): Promise<void> {
    const w = this.worker;
    this.worker = null;
    if (w) await (await w.catch(() => null))?.terminate();
  }
}
