/**
 * The narrow, structural views the runtime has of its collaborators. The real classes
 * (`ScreenCapturer`, `TesseractOcr`, `SomaModel`, `MemoryRetriever`, `TextEmbedder`, `OnlinePreferenceRanker`) satisfy
 * them; tests pass small fakes. Keeping the runtime on these interfaces means it never imports tfjs or
 * tesseract itself, so they load lazily (see `soma/index.ts`).
 */
import type { CapturedFrame } from "../capture/screen";
import type { ScreenInputs, StepInputs } from "../models/batch";
import type { StepOutput } from "../models/somaModel";
import type { OcrEngine } from "../perception/ocr";
import type { Candidate, FeedbackOutcome, RetrievedMemory } from "../types";

export type { OcrEngine };

/** `ScreenCapturer` as the runtime uses it. */
export interface CapturerLike {
  capture(force?: boolean): Promise<CapturedFrame | null>;
  readonly active: boolean;
  /** Version of the last meaningful frame. */
  readonly version: number;
  /** Why the last `capture` returned null ("no-consent", "rate-limited", "grab-error:..."). */
  readonly lastSkipReason: string;
  /** Forget change detection / rate limit (after resume). */
  reset(): void;
}

/** `SomaModel` as the runtime uses it: the screen is encoded once, then scored. */
export interface ModelLike {
  encodeScreen(inp: ScreenInputs): Float32Array;
  score(screenState: Float32Array, inp: Omit<StepInputs, keyof ScreenInputs>, history: Float32Array[]): StepOutput;
}

/** `TextEmbedder` (memory) as the runtime and the candidate encoder use it. */
export interface EmbedderLike {
  embed(text: string): Float32Array;
}

/** `MemoryRetriever`: permission-filtered top-k for a (redacted) query text. */
export interface RetrieverLike {
  retrieve(queryText: string, k?: number): Promise<RetrievedMemory[]>;
}

/**
 * `OnlinePreferenceRanker` (learning/preferenceRanker.ts): per-app Beta-Bernoulli arms over action types,
 * candidate keys and abstention, trained from explicit feedback. Never touches the model's weights.
 */
export interface RankerLike {
  /** Additive logit bias `[A + 1]` (index 0 = NO_SUGGESTION), padded to `nSlots`. */
  bias(candidates: (Candidate | null)[], app: string, nSlots?: number): Float32Array;
  /** Count one explicit outcome; false (no-op) for IGNORED. */
  updateCandidate(candidate: Candidate, app: string, outcome: FeedbackOutcome): boolean;
}
