/**
 * Explicit feedback -> labeled examples (port of `feedback.py`), plus the book of shown suggestions whose
 * feature snapshot waits for the user's answer.
 *
 * Rules kept from Python:
 *  - IGNORED outcomes are kept as *ambiguous*: never a rejection, never a negative label, never trained on.
 *  - Only consented, unambiguous examples are usable for training.
 *  - Snapshots never keep screen pixels: {@link FeedbackBook.remember} drops any `pixels` field. Only
 *    vectors (and optionally OCR ids/boxes) are kept, in memory, and the book is bounded.
 */
import type { StepInputs } from "../models/batch";
import { ActionType, FeedbackOutcome, nowMs } from "../types";

export const NO_SUGGESTION_INDEX = 0;

/**
 * Relative training weights per outcome. An edit is a softer positive ("right idea, wrong details"); a
 * rejection teaches "silence was better here" with a softer weight because the user may have wanted another
 * candidate instead.
 */
export const OUTCOME_WEIGHTS: Record<FeedbackOutcome, number> = {
  [FeedbackOutcome.ACCEPTED]: 1.0,
  [FeedbackOutcome.EDITED]: 0.5,
  [FeedbackOutcome.REJECTED]: 0.5,
  [FeedbackOutcome.IGNORED]: 0.0,
};

/** Feature snapshot of one shown suggestion: what learning needs, no pixels and no text. */
export interface FeedbackSnapshot {
  /** App the suggestion was shown in (`WindowInfo.appName`). */
  app: string;
  action: ActionType | string;
  /** `candidateKey(candidate)`. */
  candidateKey: string;
  /** Logits index of the shown candidate (1..A). */
  shownIndex: number;
  /** Model context for the step (e.g. `StepOutput.temporalState`). */
  context: Float32Array;
  /** The shown candidate's embedding. */
  candidateEmbedding: Float32Array;
  /** Optional model inputs for later re-scoring; pixels are never kept. */
  inputs?: Omit<StepInputs, "pixels">;
}

/** A training example derived from one feedback event. `target` is the logits index to learn (0 = silence). */
export interface LabeledExample {
  snapshot: FeedbackSnapshot;
  outcome: FeedbackOutcome;
  consented: boolean;
  ambiguous: boolean;
  target: number;
  weight: number;
  shownIndex: number;
  /** Shown candidate's logits index for a rejection (for a pairwise loss); null otherwise. */
  rejectedIndex: number | null;
  app: string;
  task: string;
  source: "feedback" | "anchor" | "synthetic";
  feedbackId: string;
  createdMs: number;
  /** Consented, unambiguous and not IGNORED. */
  usableForTraining: boolean;
}

function copySnapshot(s: FeedbackSnapshot): FeedbackSnapshot {
  const out: FeedbackSnapshot = {
    app: s.app, action: s.action, candidateKey: s.candidateKey, shownIndex: s.shownIndex,
    context: Float32Array.from(s.context), candidateEmbedding: Float32Array.from(s.candidateEmbedding),
  };
  if (s.inputs) {
    const i = s.inputs;
    out.inputs = {
      ocrIds: Int32Array.from(i.ocrIds), ocrBoxes: Float32Array.from(i.ocrBoxes), ocrPadding: Uint8Array.from(i.ocrPadding),
      appId: i.appId,
      memoryEmbeddings: i.memoryEmbeddings.map((m) => Float32Array.from(m)),
      candidateEmbeddings: i.candidateEmbeddings.map((c) => Float32Array.from(c)),
      candidateTypes: [...i.candidateTypes],
      ...(i.candidateMask ? { candidateMask: Uint8Array.from(i.candidateMask) } : {}),
    };
  }
  return out;
}

/**
 * Convert one outcome on the shown candidate into a labeled example:
 *  - accepted -> target = shown, weight 1.0
 *  - edited   -> target = shown, weight 0.5
 *  - rejected -> target = NO_SUGGESTION, weight 0.5, `rejectedIndex` = shown
 *  - ignored  -> ambiguous, weight 0, never usable for training
 */
export function exampleFromFeedback(snapshot: FeedbackSnapshot, outcome: FeedbackOutcome, opts: {
  consented: boolean; task?: string; feedbackId?: string;
}): LabeledExample {
  const shown = snapshot.shownIndex;
  const ambiguous = outcome === FeedbackOutcome.IGNORED;
  const positive = outcome === FeedbackOutcome.ACCEPTED || outcome === FeedbackOutcome.EDITED;
  const rejected = outcome === FeedbackOutcome.REJECTED && shown !== NO_SUGGESTION_INDEX ? shown : null;
  return {
    snapshot: copySnapshot(snapshot), outcome, consented: opts.consented, ambiguous,
    target: positive ? shown : NO_SUGGESTION_INDEX,
    weight: OUTCOME_WEIGHTS[outcome] ?? 0,
    shownIndex: shown, rejectedIndex: rejected,
    app: snapshot.app, task: opts.task ?? String(snapshot.action),
    source: "feedback", feedbackId: opts.feedbackId ?? "", createdMs: nowMs(),
    usableForTraining: opts.consented && !ambiguous,
  };
}

/** Pending shown suggestions kept for feedback (`pipeline.MAX_PENDING`). */
export const MAX_PENDING = 64;

/**
 * Bounded book of feature snapshots for shown suggestions, keyed by suggestion id. `take` is single-use, so
 * one answer trains once. The oldest entries are dropped beyond `maxPending`. Pixels are never kept.
 */
export class FeedbackBook {
  private entries = new Map<string, FeedbackSnapshot>();

  constructor(readonly maxPending = MAX_PENDING) {}

  get size(): number {
    return this.entries.size;
  }

  /** Keep a copy of `snapshot` for `suggestionId` (any `pixels` field is dropped). */
  remember(suggestionId: string, snapshot: FeedbackSnapshot): void {
    const copy = copySnapshot(snapshot);
    this.entries.delete(suggestionId);
    this.entries.set(suggestionId, copy);
    while (this.entries.size > this.maxPending) {
      const oldest = this.entries.keys().next().value as string;
      this.entries.delete(oldest);
    }
  }

  /** Look without consuming. */
  peek(suggestionId: string): FeedbackSnapshot | null {
    return this.entries.get(suggestionId) ?? null;
  }

  /** Remove and return the snapshot (null when unknown or already taken). */
  take(suggestionId: string): FeedbackSnapshot | null {
    const s = this.entries.get(suggestionId) ?? null;
    this.entries.delete(suggestionId);
    return s;
  }

  /** Forget everything (e.g. on pause or when consent is revoked). */
  clear(): void {
    this.entries.clear();
  }
}
