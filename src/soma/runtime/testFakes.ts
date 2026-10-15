/** Fakes for the runtime / service suites: never touch the real screen, OCR or webcam. */
import type { CapturedFrame } from "../capture/screen";
import { DEFAULT_CONFIG, type ModelConfig, type SomaConfig } from "../config";
import type { ScreenInputs, StepInputs } from "../models/batch";
import type { StepOutput } from "../models/somaModel";
import type { OcrEngine } from "../perception/ocr";
import type { FeedbackOutcome, OCRElement, RetrievedMemory } from "../types";
import type { CapturerLike, EmbedderLike, ModelLike, RankerLike, RetrieverLike } from "./deps";

export const smallModel = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  ...DEFAULT_CONFIG.model,
  dModel: 32, nHeads: 4, imageSize: 32, patchSize: 16, visionLayers: 1, vocabSize: 512,
  maxOcrTokens: 32, textLayers: 1, fusionLayers: 1, historyLen: 4, nApps: 16, maxMemories: 4,
  ...over,
});

/** The pette profile with a small model; `policy` overrides merge in. */
export function testConfig(policy: Partial<SomaConfig["policy"]> = {}): SomaConfig {
  return {
    ...DEFAULT_CONFIG,
    model: smallModel(),
    capture: { ...DEFAULT_CONFIG.capture, minIntervalS: 0 },
    memory: { ...DEFAULT_CONFIG.memory },
    policy: { ...DEFAULT_CONFIG.policy, ...policy },
  };
}

export function fakeImage(w = 16, h = 10, fill = 255): ImageData {
  return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4).fill(fill), colorSpace: "srgb" } as ImageData;
}

/** A shared screen whose content (track label, OCR lines) the test changes between steps. */
export class FakeCapturer implements CapturerLike {
  active = true;
  version = 0;
  lastSkipReason = "";
  grabs = 0;
  changed = true;
  resets = 0;
  constructor(public trackLabel = "", public surface = "window") {}
  async capture(force = false): Promise<CapturedFrame | null> {
    if (!this.active) {
      this.lastSkipReason = "no-consent";
      return null;
    }
    this.grabs += 1;
    const changed = force || this.changed;
    if (changed) this.version += 1;
    this.lastSkipReason = "";
    return { image: fakeImage(), surface: this.surface, trackLabel: this.trackLabel, version: this.version, changed };
  }
  reset(): void {
    this.resets += 1;
  }
}

/** OCR returning fixed lines stacked down the screen. */
export class LinesOcr implements OcrEngine {
  calls = 0;
  constructor(public lines: string[] = []) {}
  async recognize(_image: ImageData, windowId: string): Promise<OCRElement[]> {
    this.calls += 1;
    return this.lines.map((text, i) => ({
      text, bbox: [0.05, 0.2 + 0.04 * i, 0.7, 0.23 + 0.04 * i] as [number, number, number, number],
      confidence: 1, source: "ocr" as const, uiRole: null, windowId, timestampMs: 0,
    }));
  }
  async dispose(): Promise<void> {}
}

/**
 * Deterministic stand-in for SomaModel. By default every unmasked candidate scores `candidateLogit` (minus a
 * small rank penalty so the first wins) against NO_SUGGESTION at 0; `scoreFn` overrides.
 */
export class FakeModel implements ModelLike {
  encodes = 0;
  scores = 0;
  lastInputs: Omit<StepInputs, keyof ScreenInputs> | null = null;
  constructor(public dim = 32, public candidateLogit = 3,
    public scoreFn?: (inp: Omit<StepInputs, keyof ScreenInputs>) => number[]) {}
  encodeScreen(inp: ScreenInputs): Float32Array {
    this.encodes += 1;
    const v = new Float32Array(this.dim);
    inp.ocrIds.forEach((id, i) => (v[i % this.dim] += id / 1000));
    return v;
  }
  score(screenState: Float32Array, inp: Omit<StepInputs, keyof ScreenInputs>): StepOutput {
    this.scores += 1;
    this.lastInputs = inp;
    const A = inp.candidateEmbeddings.length;
    const raw = this.scoreFn?.(inp)
      ?? [0, ...Array.from({ length: A }, (_, i) => this.candidateLogit - 0.01 * i)];
    const logits = Float32Array.from(raw.map((v, i) => (i > 0 && inp.candidateMask?.[i - 1] ? -1e9 : v)));
    return { screenState, temporalState: Float32Array.from(screenState), logits, memoryGate: 0.5,
      memoryAttention: new Float32Array(1 + inp.memoryEmbeddings.length) };
  }
}

/** Unit-length bag of characters (stands in for the memory TextEmbedder). */
export class CharEmbedder implements EmbedderLike {
  constructor(public dim = 32) {}
  embed(text: string): Float32Array {
    const v = new Float32Array(this.dim);
    for (const ch of text) v[ch.charCodeAt(0) % this.dim] += 1;
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  }
}

export class FakeRetriever implements RetrieverLike {
  queries: string[] = [];
  constructor(public memories: RetrievedMemory[] = []) {}
  async retrieve(queryText: string, k = 8): Promise<RetrievedMemory[]> {
    this.queries.push(queryText);
    return this.memories.slice(0, k);
  }
}

export class FakeRanker implements RankerLike {
  updates: { outcome: FeedbackOutcome; context: Float32Array; candidate: Float32Array }[] = [];
  constructor(public biasValue = 0) {}
  bias(_context: Float32Array, candidateEmbeddings: Float32Array[]): Float32Array {
    return new Float32Array(candidateEmbeddings.length).fill(this.biasValue);
  }
  async update(context: Float32Array, candidate: Float32Array, outcome: FeedbackOutcome): Promise<void> {
    this.updates.push({ outcome, context, candidate });
  }
}

export class FakeClock {
  t = 1000;
  now = (): number => this.t;
}
