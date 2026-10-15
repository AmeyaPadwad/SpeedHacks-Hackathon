/**
 * SOMA runtime loop: one step per call (port of SOMA/soma/runtime/pipeline.py).
 *
 * `SomaRuntime.step()` runs:
 *
 *     pause / screen granted        -> capturer.capture()          (null / unchanged -> skipped, no OCR/model)
 *     exclusion by track label      -> OCR -> inferWindow -> exclusion again
 *     parseScreen -> redact         -> featurize -> model.encodeScreen   (pushed to history afterwards)
 *     retrieval                     -> redacted screen text -> permission-filtered top-k
 *     candidates                    -> generateCandidates (+ context templates) -> blocked mask -> encoder
 *     policy                        -> model.score (temporal -> memory -> logits) + online ranker bias
 *     gate                          -> SafetyGate.decide -> Suggestion (or silence)
 *
 * - **Consent.** Screen consent is the browser's: a granted, live `getDisplayMedia` stream. Folders are the
 *   ones the user picked (`setAllowedFolders`). Feedback storage is always on and local.
 * - **Nothing executes.** The runtime only returns suggestions. It imports nothing that runs commands,
 *   opens URLs or sends input, and its approval ledger has no executors.
 * - **Nothing persists.** Frames and pixels are dropped after each step. Snapshots kept for feedback hold
 *   model vectors only (no pixels, no screen text).
 */
import { redactObservation, redactText } from "../capture/redaction";
import { inferWindow, isExcluded } from "../capture/window";
import type { SomaConfig } from "../config";
import { FeedbackBook, type FeedbackSnapshot } from "../learning/feedback";
import { applyBias } from "../learning/preferenceRanker";
import { isAuthorizedRef } from "../memory/access";
import { ScreenHistory } from "../models/history";
import { appBucket, imageToPixels, observationToArrays } from "../perception/featurize";
import { parseScreen } from "../perception/screenParser";
import { CandidateEncoder } from "../policy/candidateEncoder";
import { generateCandidates } from "../policy/candidateGenerator";
import { addContextCandidates } from "../policy/contextCandidates";
import { ApprovalLedger } from "../policy/decision";
import { type Clock, monotonicSeconds, NO_SUGGESTION_LABEL, SafetyGate } from "../policy/safetyGate";
import {
  type Candidate,
  candidateKey,
  FeedbackOutcome,
  newId,
  nowMs,
  observationText,
  type RetrievedMemory,
  type ScreenObservation,
  type Suggestion,
  type WindowInfo,
} from "../types";
import type { CapturerLike, EmbedderLike, ModelLike, OcrEngine, RankerLike, RetrieverLike } from "./deps";
import { FileRefIndex } from "./files";

// Skip reasons (`StepResult.skippedReason`); capture adds its own ("rate-limited", "grab-error:<type>").
export const SKIP_PAUSED = "paused";
export const SKIP_NO_CONSENT = "no-consent";
export const SKIP_EXCLUDED = "excluded";
export const SKIP_UNCHANGED = "unchanged";
export const SKIP_STALE = "paused-during-step";
export const SKIP_NOT_READY = "not-ready";

/** Max characters of redacted screen text used for the retrieval query. */
export const QUERY_CHARS = 4000;
/** How many recent shown suggestions are remembered for feedback routing. */
export const MAX_PENDING = 64;

export type Timings = Record<"capture" | "ocr" | "parse" | "encode" | "retrieve" | "policy" | "gate" | "total", number>;

export interface StepResult {
  frameVersion: number;
  window: WindowInfo;
  observation: ScreenObservation;
  candidates: Candidate[];
  /** `suggestion.candidate === null` when NO_SUGGESTION or gated. */
  suggestion: Suggestion;
  retrieved: RetrievedMemory[];
  timingsMs: Timings;
  /** Set when no frame was processed. */
  skippedReason: string | null;
}

/** Routing state per shown suggestion (the learning snapshot lives in the `FeedbackBook`). */
interface Shown {
  suggestion: Suggestion;
  candidate: Candidate;
  app: string;
  resolved: boolean;
}

export interface RuntimeDeps {
  /** Null until the screen is granted. */
  capturer?: CapturerLike | null;
  /** Null until OCR / tfjs are loaded (steps skip with "not-ready"). */
  ocr?: OcrEngine | null;
  model?: ModelLike | null;
  /** Shared text embedder (memory index space); also embeds candidate text. */
  embedder: EmbedderLike;
  retriever?: RetrieverLike | null;
  ranker?: RankerLike | null;
  gate?: SafetyGate;
  candidateEncoder?: CandidateEncoder;
  /** Seconds, monotonic (gate cooldowns). */
  clock?: Clock;
  /** Every indexed file ref (`folderId/relative/path`), to authorize paths seen on screen. */
  fileRefs?: () => Promise<Iterable<string>>;
  /** Snapshots of shown suggestions (model vectors only) for learning. */
  feedbackBook?: FeedbackBook;
  /** Called after feedback updated the ranker (the service persists it). */
  onLearned?: () => void;
}

const ZERO_TIMINGS = (): Timings =>
  ({ capture: 0, ocr: 0, parse: 0, encode: 0, retrieve: 0, policy: 0, gate: 0, total: 0 });

const neutralWindow = (): WindowInfo => ({ appName: "", title: "", windowId: "", surface: "" });

function silentSuggestion(version: number, reason: string): Suggestion {
  return { candidate: null, confidence: 1, screenVersion: version, reason,
    distribution: { [NO_SUGGESTION_LABEL]: 1 }, id: newId("sug_"), createdMs: nowMs() };
}

/** Steps are serialised; `feedback` / `pause` / `resume` may be called at any time. */
export class SomaRuntime {
  capturer: CapturerLike | null;
  ocr: OcrEngine | null;
  model: ModelLike | null;
  retriever: RetrieverLike | null;
  ranker: RankerLike | null;
  readonly embedder: EmbedderLike;
  readonly gate: SafetyGate;
  readonly candidateEncoder: CandidateEncoder;
  /** Suggestions are never executed: this ledger has no executors. */
  readonly ledger = new ApprovalLedger();
  readonly history: ScreenHistory;
  /** Feature snapshots of shown suggestions: model vectors and ids, never pixels or screen text. */
  readonly feedbackBook: FeedbackBook;

  steps = 0;
  processed = 0;

  private pausedFlag = false;
  private gen = 0;
  private last: StepResult | null = null;
  private lastSkip: string | null = null;
  private lastVersion = 0;
  private readonly shown = new Map<string, Shown>();
  private chain: Promise<unknown> = Promise.resolve();
  private allowed = new Set<string>();
  private files = new FileRefIndex();
  private filesStale = true;
  private readonly fileRefs?: () => Promise<Iterable<string>>;
  /** Called after feedback trained the ranker (the service persists it). */
  onLearned?: () => void;

  constructor(readonly cfg: SomaConfig, deps: RuntimeDeps) {
    this.capturer = deps.capturer ?? null;
    this.ocr = deps.ocr ?? null;
    this.model = deps.model ?? null;
    this.retriever = deps.retriever ?? null;
    this.ranker = deps.ranker ?? null;
    this.embedder = deps.embedder;
    this.gate = deps.gate ?? new SafetyGate(cfg.policy, deps.clock ?? monotonicSeconds);
    this.candidateEncoder = deps.candidateEncoder
      ?? new CandidateEncoder(cfg.model.dModel, deps.embedder, cfg.model.nActionTypes, cfg.seed);
    this.history = new ScreenHistory(cfg.model.historyLen, { dim: cfg.model.dModel });
    this.feedbackBook = deps.feedbackBook ?? new FeedbackBook(MAX_PENDING);
    this.fileRefs = deps.fileRefs;
    this.onLearned = deps.onLearned;
  }

  // ------------------------------------------------------------ state

  get paused(): boolean {
    return this.pausedFlag;
  }

  /** Bumped on every pause; results computed under an older generation are stale. */
  get generation(): number {
    return this.gen;
  }

  /** The most recent result for which a frame was actually processed. */
  get lastResult(): StepResult | null {
    return this.last;
  }

  /** Reason the most recent step was skipped (null if it processed a frame). */
  get lastSkipReason(): string | null {
    return this.lastSkip;
  }

  /** Version of the latest meaningful frame. */
  get frameVersion(): number {
    return this.capturer?.version || this.lastVersion;
  }

  /** Stop instantly: capture is denied, history is cleared, in-flight results go stale. */
  pause(): void {
    this.pausedFlag = true;
    this.gen += 1;
    this.history.clear();
  }

  resume(): void {
    this.capturer?.reset();
    this.pausedFlag = false;
  }

  /** The folders the user picked (ids). Files outside them are never proposed or opened. */
  setAllowedFolders(folderIds: Iterable<string>): void {
    this.allowed = new Set(folderIds);
    this.filesStale = true;
  }

  /** Re-read the indexed file list before the next step (after indexing / forgetting). */
  invalidateFiles(): void {
    this.filesStale = true;
  }

  /** File ACL: a well-formed `folderId/relative/path` inside a folder the user picked. */
  authorizedFile(ref: string): boolean {
    return isAuthorizedRef(ref, this.allowed);
  }

  /** True while the last shown suggestion still matches `currentVersion` and has had no feedback. */
  currentSuggestionValid(currentVersion: number): boolean {
    const r = this.last;
    if (r === null || r.suggestion.candidate === null || this.pausedFlag) return false;
    const entry = this.shown.get(r.suggestion.id);
    if (entry?.resolved) return false;
    return r.suggestion.screenVersion === Math.trunc(currentVersion);
  }

  knownSuggestion(id: string): boolean {
    return this.shown.has(id);
  }

  /** Plain view of a remembered suggestion (no screen text beyond the suggestion). */
  suggestionInfo(id: string): { id: string; action: string; app: string; resolved: boolean } | null {
    const e = this.shown.get(id);
    return e ? { id, action: e.candidate.action, app: e.app, resolved: e.resolved } : null;
  }

  // ------------------------------------------------------------ loop

  /** One loop iteration: capture (if allowed and meaningful), then process. Calls are serialised. */
  step(force = false): Promise<StepResult> {
    const run = this.chain.then(() => this.runStep(force));
    this.chain = run.catch(() => undefined);
    return run;
  }

  private async runStep(force: boolean): Promise<StepResult> {
    const t0 = performance.now();
    this.steps += 1;
    if (this.pausedFlag) return this.skipped(SKIP_PAUSED, t0);
    const cap = this.capturer;
    if (cap === null || !cap.active) return this.skipped(SKIP_NO_CONSENT, t0);
    if (this.ocr === null || this.model === null) return this.skipped(SKIP_NOT_READY, t0);
    const gen = this.gen;
    const frame = await cap.capture(force);
    const captureMs = performance.now() - t0;
    if (frame === null) return this.skipped(cap.lastSkipReason || "no-frame", t0, null, { capture: captureMs });
    if (!frame.changed) return this.skipped(SKIP_UNCHANGED, t0, null, { capture: captureMs }, frame.version);
    if (gen !== this.gen || this.pausedFlag) return this.skipped(SKIP_STALE, t0, null, {}, frame.version);
    return this.process(frame.image, frame.surface, frame.trackLabel, frame.version, t0, captureMs, gen);
  }

  private async process(image: ImageData, surface: string, trackLabel: string, version: number,
    t0: number, captureMs: number, gen: number): Promise<StepResult> {
    const mc = this.cfg.model;
    const timings = ZERO_TIMINGS();
    timings.capture = captureMs;

    // exclusion before OCR when the track label already names the window
    const early = inferWindow(surface, trackLabel, []);
    const earlyWhy = isExcluded(early, this.cfg.capture);
    if (earlyWhy) return this.skipped(`${SKIP_EXCLUDED}:${earlyWhy}`, t0, early, timings, version);

    let t = performance.now();
    const elements = await this.ocr!.recognize(image, early.windowId);
    timings.ocr = performance.now() - t;
    if (gen !== this.gen || this.pausedFlag) return this.skipped(SKIP_STALE, t0, null, timings, version);

    t = performance.now();
    const rawWindow = inferWindow(surface, trackLabel, elements);
    const why = isExcluded(rawWindow, this.cfg.capture);
    if (why) return this.skipped(`${SKIP_EXCLUDED}:${why}`, t0, rawWindow, timings, version);
    const obs = redactObservation(parseScreen(rawWindow, elements, version, nowMs()));
    const window = obs.window;
    timings.parse = performance.now() - t;

    t = performance.now();
    const arrays = observationToArrays(obs, mc);
    const screen = {
      pixels: imageToPixels(image, mc.imageSize),
      ocrIds: arrays.ids, ocrBoxes: arrays.boxes, ocrPadding: arrays.mask,
    };
    const screenState = this.model!.encodeScreen(screen);
    this.history.observeApp(window.appName);
    timings.encode = performance.now() - t;

    t = performance.now();
    const retrieved = await this.retrieve(obs);
    await this.refreshFiles();
    timings.retrieve = performance.now() - t;
    if (gen !== this.gen || this.pausedFlag) return this.skipped(SKIP_STALE, t0, null, timings, version);

    t = performance.now();
    const A = mc.maxCandidates;
    let candidates = this.generate(obs, retrieved, A);
    if (this.cfg.policy.contextTemplates) candidates = addContextCandidates(obs, candidates, A);
    const memById = new Map(retrieved.map((r) => [r.item.id, r] as const));
    const blocked = SafetyGate.blockedMask(candidates, (p) => this.authorizedFile(p), [], memById);
    const enc = this.candidateEncoder.encode(candidates, memById, A, mc.dModel, blocked);
    const out = this.model!.score(screenState, {
      appId: appBucket(window.appName, mc.nApps),
      memoryEmbeddings: retrieved.slice(0, mc.maxMemories).map((r) => r.embedding),
      candidateEmbeddings: enc.embeddings,
      candidateTypes: enc.types,
      candidateMask: enc.mask,
    }, this.history.states());
    let logits = Float32Array.from(out.logits);
    if (this.ranker !== null && candidates.length) {
      logits = applyBias(logits, this.ranker.bias(candidates, window.appName, logits.length - 1));
    }
    timings.policy = performance.now() - t;

    t = performance.now();
    const suggestion = this.gate.decide(logits, candidates, obs, { blocked, record: false });
    timings.gate = performance.now() - t;

    // history gets the current state only after it was used as "previous" context
    this.history.push(screenState);

    if (gen !== this.gen || this.pausedFlag) {
      // paused while computing: drop the result, record nothing as shown
      return this.skipped(SKIP_STALE, t0, null, timings, version);
    }

    if (suggestion.candidate !== null) {
      const idx = candidates.indexOf(suggestion.candidate);
      this.remember(suggestion, suggestion.candidate, idx + 1, window.appName, out.temporalState, enc.embeddings[idx]);
    }
    timings.total = performance.now() - t0;
    const result: StepResult = { frameVersion: version, window, observation: obs, candidates, suggestion, retrieved,
      timingsMs: timings, skippedReason: null };
    this.last = result;
    this.lastSkip = null;
    this.lastVersion = version;
    this.processed += 1;
    return result;
  }

  private skipped(reason: string, t0: number, window: WindowInfo | null = null, timings: Partial<Timings> = {},
    version?: number): StepResult {
    // Only the app name of a skipped window is kept; titles of excluded windows may be sensitive.
    const win: WindowInfo = window ? { ...neutralWindow(), appName: window.appName } : neutralWindow();
    const v = version ?? this.frameVersion;
    const tm = { ...ZERO_TIMINGS(), ...timings, total: performance.now() - t0 };
    this.lastSkip = reason;
    return { frameVersion: v, window: win, observation: { window: win, elements: [], version: v, timestampMs: nowMs() },
      candidates: [], suggestion: silentSuggestion(v, `skipped: ${reason}`), retrieved: [], timingsMs: tm,
      skippedReason: reason };
  }

  // ------------------------------------------------------------ stages

  private async retrieve(obs: ScreenObservation): Promise<RetrievedMemory[]> {
    if (this.retriever === null) return [];
    const k = Math.min(this.cfg.memory.topK, this.cfg.model.maxMemories);
    const text = redactText(`${obs.window.appName} ${obs.window.title}\n${observationText(obs)}`).slice(0, QUERY_CHARS);
    if (!text.trim()) return [];
    try {
      const got = await this.retriever.retrieve(text, k);
      // only memories from folders the user still allows
      return got.filter((r) => !r.item.sourceUri || this.authorizedFile(r.item.sourceUri)).slice(0, k);
    } catch {
      return [];
    }
  }

  private async refreshFiles(): Promise<void> {
    if (!this.filesStale || !this.fileRefs) return;
    try {
      const refs = [...(await this.fileRefs())].filter((r) => this.authorizedFile(r));
      this.files = new FileRefIndex(refs);
      this.filesStale = false;
    } catch {
      this.files = new FileRefIndex();
    }
  }

  /** Generator over on-screen paths; authorized paths are rewritten to the indexed ref they resolve to. */
  private generate(obs: ScreenObservation, retrieved: RetrievedMemory[], A: number): Candidate[] {
    const resolved = new Map<string, string>();
    const onScreenAuthorized = (path: string): boolean => {
      if (this.authorizedFile(path)) {
        resolved.set(path, path);
        return true;
      }
      const ref = this.files.resolve(path);
      if (ref !== null && this.authorizedFile(ref)) {
        resolved.set(path, ref);
        return true;
      }
      return false;
    };
    const cands = generateCandidates(obs, retrieved, A, onScreenAuthorized);
    for (const c of cands) {
      if (c.fileRef !== null) c.fileRef = resolved.get(c.fileRef) ?? null;
    }
    return cands.filter((c) => !(c.fileRef === null && c.action === "OPEN_RELEVANT_FILE"));
  }

  // ------------------------------------------------------------ feedback

  private remember(suggestion: Suggestion, candidate: Candidate, shownIndex: number, app: string,
    context: Float32Array, candidateEmbedding: Float32Array): void {
    const key = candidateKey(candidate);
    this.shown.set(suggestion.id, { suggestion, candidate, app, resolved: false });
    this.feedbackBook.remember(suggestion.id, { app, action: candidate.action, candidateKey: key, shownIndex,
      context, candidateEmbedding });
    this.gate.recordShown(key);
    while (this.shown.size > MAX_PENDING) this.shown.delete(this.shown.keys().next().value!);
  }

  /**
   * Route explicit feedback on a shown suggestion. REJECTED dismisses the candidate in the gate (not repeated
   * within the duplicate window). ACCEPTED / REJECTED / EDITED train the online preference ranker (feedback
   * storage is always on and local); IGNORED is ambiguous and never treated as a negative. Unknown or
   * already answered ids are ignored. Returns the learning snapshot (null when unknown).
   */
  async feedback(suggestionId: string, outcome: FeedbackOutcome): Promise<FeedbackSnapshot | null> {
    const entry = this.shown.get(suggestionId);
    if (!entry || entry.resolved) return null;
    entry.resolved = true;
    const snapshot = this.feedbackBook.take(suggestionId);
    if (outcome === FeedbackOutcome.REJECTED) this.gate.recordDismissed(candidateKey(entry.candidate));
    if (this.ranker !== null && this.ranker.updateCandidate(entry.candidate, entry.app, outcome)) this.onLearned?.();
    return snapshot;
  }
}
