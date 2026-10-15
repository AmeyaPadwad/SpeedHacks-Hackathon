/**
 * Shared, framework-light data contracts used across SOMA's modules (port of SOMA/soma/types.py).
 *
 * Every module (capture, perception, memory, policy, learning, runtime) exchanges these types. Keep them
 * small and free of heavy imports (no tfjs, no DOM) so any module can depend on them. Tensor contracts
 * for the model live in `soma/models/batch.ts`.
 */

/** Short random identifier, optionally prefixed (`"sug_3f2a..."`). */
export function newId(prefix = ""): string {
  const bytes = new Uint8Array(8);
  crypto.getRandomValues(bytes);
  return prefix + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Wall-clock time in milliseconds. */
export const nowMs = (): number => Date.now();

// ---------------------------------------------------------------- capture

/**
 * What is being shared. A web page cannot see the foreground app, so `appName` is a canonical process-like
 * name inferred from the shared surface and its on-screen text (`"code"`, `"windowsterminal"`, `"chrome"`,
 * `""` when unknown); see `soma/capture/window.ts`. No pixels, no text content.
 */
export interface WindowInfo {
  appName: string;
  title: string;
  windowId: string;
  /** `getDisplayMedia` surface: "monitor" | "window" | "browser" | "". */
  surface: string;
}

/** One in-memory screenshot. Never persisted. `version` increases per capture. */
export interface ScreenFrame {
  image: ImageData;
  window: WindowInfo;
  version: number;
  timestampMs: number;
  redactedRegions: [number, number, number, number][];
}

// ---------------------------------------------------------------- perception

export type ElementSource = "ocr" | "accessibility" | "synthetic";

/** One recognized text element with normalized xyxy geometry. */
export interface OCRElement {
  text: string;
  bbox: [number, number, number, number];
  confidence: number;
  source: ElementSource;
  /** "button", "text", "error", ... when known. */
  uiRole: string | null;
  windowId: string;
  timestampMs: number;
}

/** Perceived screen state: metadata + OCR elements for one frame version. */
export interface ScreenObservation {
  window: WindowInfo;
  elements: OCRElement[];
  version: number;
  timestampMs: number;
}

export const observationText = (o: ScreenObservation): string => o.elements.map((e) => e.text).join("\n");

// ---------------------------------------------------------------- memory

export enum SourceType {
  IMAGE = "image",
  DOCUMENT = "document",
  PREFERENCE = "preference",
  FEEDBACK = "feedback",
  UI_EVENT = "ui-event",
}

/** A permissioned memory record. Raw files stay in the user's folder; `sourceUri` is `folderId/relative/path`. */
export interface MemoryItem {
  id: string;
  sourceType: SourceType;
  userApproved: boolean;
  textSummary: string;
  sourceUri: string | null;
  tags: string[];
  provenance: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  expiresAt: string | null;
}

/** A memory returned by retrieval, with its embedding and score. */
export interface RetrievedMemory {
  item: MemoryItem;
  embedding: Float32Array;
  score: number;
}

// ---------------------------------------------------------------- policy

/** Bounded action vocabulary. Index 0 of policy logits is NO_SUGGESTION. */
export enum ActionType {
  NO_SUGGESTION = "NO_SUGGESTION",
  EXPLAIN_ERROR = "EXPLAIN_ERROR",
  OPEN_RELEVANT_FILE = "OPEN_RELEVANT_FILE",
  COMPARE_WITH_PRIOR_WORK = "COMPARE_WITH_PRIOR_WORK",
  SUGGEST_NEXT_STEP = "SUGGEST_NEXT_STEP",
  ASK_CLARIFYING_QUESTION = "ASK_CLARIFYING_QUESTION",
}

export const ACTION_TYPES: ActionType[] = Object.values(ActionType);

/** A proposed (never executed) action from the candidate generator. */
export interface Candidate {
  action: ActionType;
  /** Human-readable suggestion. */
  text: string;
  target: OCRElement | null;
  memoryRef: string | null;
  /** Only authorized files (`folderId/relative/path`). */
  fileRef: string | null;
  evidence: string[];
  id: string;
  /** Context template id (`soma/policy/contextCandidates.ts`), "" otherwise. */
  template: string;
  /** Line in `fileRef`, when known (not part of the key). */
  line: number | null;
}

export function makeCandidate(c: Pick<Candidate, "action" | "text"> & Partial<Candidate>): Candidate {
  return {
    target: null, memoryRef: null, fileRef: null, evidence: [], template: "", line: null,
    ...c, id: c.id ?? newId("cand_"),
  };
}

/** Stable de-duplication key (action + target text + refs [+ template]). */
export function candidateKey(c: Candidate): string {
  const key = `${c.action}|${c.target?.text ?? ""}|${c.memoryRef ?? ""}|${c.fileRef ?? ""}`;
  return c.template ? `${key}|${c.template}` : key;
}

/** What the UI may show. `candidate === null` means stay silent. */
export interface Suggestion {
  candidate: Candidate | null;
  confidence: number;
  screenVersion: number;
  reason: string;
  /** label -> probability */
  distribution: Record<string, number>;
  id: string;
  createdMs: number;
}

export enum FeedbackOutcome {
  ACCEPTED = "accepted",
  REJECTED = "rejected",
  EDITED = "edited",
  IGNORED = "ignored",
}

export interface FeedbackEvent {
  suggestionId: string;
  outcome: FeedbackOutcome;
  consentedForTraining: boolean;
  id: string;
  observedAt: string;
}
