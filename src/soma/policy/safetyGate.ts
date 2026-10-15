/**
 * Confidence / interruptibility gate: decides whether SOMA speaks at all (port of
 * SOMA/soma/policy/safety_gate.py).
 *
 * NO_SUGGESTION must often win. The gate stays silent when:
 * - the shared app is in `quietApps`,
 * - there are no (eligible) candidates,
 * - a suggestion was shown less than `cooldownS` ago,
 * - the best non-abstain probability is below `relevanceThreshold`,
 * - it does not beat NO_SUGGESTION by `marginOverAbstain`,
 * - the candidate's key was dismissed or already shown within `duplicateWindowS`,
 * - the candidate references a blocked source / unauthorized file,
 * - its UI target no longer exists on the latest observation.
 *
 * Every decision returns a `Suggestion` with the full distribution, the screen version and a reason.
 */
import type { PolicyConfig } from "../config";
import {
  ActionType,
  type Candidate,
  candidateKey,
  newId,
  nowMs,
  type RetrievedMemory,
  type ScreenObservation,
  type Suggestion,
} from "../types";
import { verifyTarget } from "./grounding";

export const NO_SUGGESTION_LABEL = ActionType.NO_SUGGESTION;

/** Seconds, monotonic. */
export type Clock = () => number;
export const monotonicSeconds: Clock = () => performance.now() / 1000;

type Scores = ArrayLike<number> | ArrayLike<number>[];

function toVector(scores: Scores): number[] {
  let arr: ArrayLike<number> = scores as ArrayLike<number>;
  const first = (scores as ArrayLike<unknown>)[0];
  if (first !== undefined && typeof first !== "number") {
    if ((scores as ArrayLike<number>[]).length !== 1) throw new RangeError("expected scores shaped [A+1] or [1, A+1]");
    arr = (scores as ArrayLike<number>[])[0];
  }
  const out = Array.from(arr, Number);
  if (!out.length) throw new RangeError("expected scores shaped [A+1] or [1, A+1], got an empty vector");
  return out;
}

/**
 * Convert logits or probabilities to a probability vector. With `isLogits = null` the input is treated as
 * probabilities when it is non-negative, <= 1 and sums to ~1, otherwise as logits (masked -Infinity is
 * fine). Non-finite probabilities become 0; nothing valid means abstain.
 */
export function toProbs(scores: Scores, isLogits: boolean | null = null): number[] {
  let arr = toVector(scores);
  const abstain = () => arr.map((_, i) => (i === 0 ? 1 : 0));
  if (isLogits === null) {
    const sum = arr.reduce((a, b) => a + b, 0);
    const probLike = arr.every((v) => Number.isFinite(v) && v >= 0 && v <= 1) && Math.abs(sum - 1) < 1e-3;
    isLogits = !probLike;
  }
  if (isLogits) {
    arr = arr.map((v) => (Number.isNaN(v) ? -Infinity : v));
    const m = Math.max(...arr);
    if (!Number.isFinite(m)) return abstain();
    const e = arr.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  arr = arr.map((v) => (Number.isFinite(v) ? Math.max(v, 0) : 0));
  const s = arr.reduce((a, b) => a + b, 0);
  if (s <= 0) return abstain();
  return arr.map((v) => v / s);
}

/** Unique labels: NO_SUGGESTION then action names (`#2` for repeats). */
export function distributionLabels(candidates: readonly Candidate[]): string[] {
  const labels: string[] = [NO_SUGGESTION_LABEL];
  const counts = new Map<string, number>();
  for (const c of candidates) {
    const n = (counts.get(c.action) ?? 0) + 1;
    counts.set(c.action, n);
    labels.push(n === 1 ? c.action : `${c.action}#${n}`);
  }
  return labels;
}

export function normApp(name: string | null | undefined): string {
  const n = (name ?? "").trim().toLowerCase();
  return n.endsWith(".exe") ? n.slice(0, -4) : n;
}

export interface DecideOptions {
  /** Shared app; defaults to `obs.window.appName`. */
  appName?: string | null;
  /** Per-candidate block flags (see `SafetyGate.blockedMask`). */
  blocked?: readonly boolean[] | null;
  /** Force the interpretation of `scores` (auto-detected when null). */
  isLogits?: boolean | null;
  /** When the gate speaks, record the key as shown (starts the cooldown). Default true. */
  record?: boolean;
}

/** Stateful interruption policy around the scorer's output. `clock` (seconds) is injectable for tests. */
export class SafetyGate {
  private lastShownAt: number | null = null;
  private readonly shown = new Map<string, number>();
  private readonly dismissed = new Map<string, number>();

  constructor(readonly cfg: PolicyConfig, readonly clock: Clock = monotonicSeconds, readonly iouThreshold = 0.5) {}

  // ------------------------------------------------------------ history

  /** Mark a candidate key as shown now (starts cooldown + duplicate window). */
  recordShown(key: string): void {
    const now = this.clock();
    this.shown.set(key, now);
    this.lastShownAt = now;
  }

  /** Mark a candidate key as dismissed; it is suppressed for the duplicate window. */
  recordDismissed(key: string): void {
    this.dismissed.set(key, this.clock());
  }

  reset(): void {
    this.lastShownAt = null;
    this.shown.clear();
    this.dismissed.clear();
  }

  inCooldown(): boolean {
    return this.cooldownAt(this.clock());
  }

  isDuplicate(key: string): boolean {
    return this.duplicateReason(key, this.clock()) !== null;
  }

  private cooldownAt(now: number): boolean {
    return this.lastShownAt !== null && now - this.lastShownAt < this.cfg.cooldownS;
  }

  private duplicateReason(key: string, now: number): string | null {
    const window = this.cfg.duplicateWindowS;
    const d = this.dismissed.get(key);
    if (d !== undefined && now - d < window) return "dismissed recently";
    const s = this.shown.get(key);
    if (s !== undefined && now - s < window) return "already shown recently";
    return null;
  }

  private prune(now: number): void {
    for (const store of [this.shown, this.dismissed]) {
      for (const [k, t] of [...store]) if (now - t >= this.cfg.duplicateWindowS) store.delete(k);
    }
  }

  // ------------------------------------------------------------ blocking

  /**
   * True for candidates that must never be shown: a file `authorizedFile` does not allow (no predicate
   * means deny), a memory id or source URI in `blockedSources`, a memory that is not user-approved, or
   * (when `memoriesById` is given) a memory that was not retrieved.
   */
  static blockedMask(
    candidates: readonly Candidate[],
    authorizedFile: ((path: string) => boolean) | null = null,
    blockedSources: ReadonlySet<string> | readonly string[] = [],
    memoriesById: ReadonlyMap<string, RetrievedMemory> | null = null,
  ): boolean[] {
    const blockedSet = new Set(blockedSources);
    return candidates.map((c) => {
      let blocked = false;
      if (c.fileRef !== null) {
        blocked = authorizedFile === null || !authorizedFile(c.fileRef);
        blocked = blocked || blockedSet.has(c.fileRef);
      }
      if (c.memoryRef !== null && !blocked) {
        if (blockedSet.has(c.memoryRef)) blocked = true;
        else if (memoriesById !== null) {
          const mem = memoriesById.get(c.memoryRef);
          if (!mem || !mem.item.userApproved) blocked = true;
          else if (mem.item.sourceUri && (blockedSet.has(mem.item.sourceUri)
            || (authorizedFile !== null && !authorizedFile(mem.item.sourceUri)))) blocked = true;
        }
      }
      return blocked;
    });
  }

  // ------------------------------------------------------------ decision

  /**
   * Pick one candidate to show, or stay silent.
   *
   * @param scores logits or probabilities over `[NO_SUGGESTION, ...candidates]` (`[A+1]` or `[1, A+1]`;
   *   extra padded slots are ignored).
   * @param candidates the real candidates, in the scorer's order.
   * @param obs latest observation; used for grounding and `screenVersion`.
   */
  decide(scores: Scores, candidates: readonly Candidate[], obs: ScreenObservation | null,
    opts: DecideOptions = {}): Suggestion {
    const { blocked = null, isLogits = null, record = true } = opts;
    const cands = [...candidates];
    const vec = toVector(scores);
    if (vec.length < cands.length + 1) {
      throw new RangeError(`got ${vec.length} scores for ${cands.length} candidates (+1 abstain)`);
    }
    let probs = toProbs(vec, isLogits).slice(0, cands.length + 1);
    const total = probs.reduce((a, b) => a + b, 0);
    if (total > 0) probs = probs.map((p) => p / total);
    const labels = distributionLabels(cands);
    const dist: Record<string, number> = {};
    labels.forEach((l, i) => (dist[l] = probs[i]));
    const version = obs !== null ? obs.version : -1;
    const pAbstain = probs[0];

    const make = (candidate: Candidate | null, confidence: number, reason: string): Suggestion =>
      ({ candidate, confidence, screenVersion: version, reason, distribution: dist, id: newId("sug_"), createdMs: nowMs() });
    const silent = (reason: string) => make(null, pAbstain, reason);

    const app = opts.appName ?? (obs !== null ? obs.window.appName : "");
    const quiet = new Set(this.cfg.quietApps.map(normApp));
    if (app && quiet.has(normApp(app))) return silent(`quiet mode for app '${app}'`);
    if (!cands.length) return silent("no candidates for this screen");

    const now = this.clock();
    this.prune(now);
    if (this.cooldownAt(now)) {
      const left = this.cfg.cooldownS - (now - (this.lastShownAt ?? now));
      return silent(`cooldown: ${left.toFixed(0)}s left since the last suggestion`);
    }

    const order = cands.map((_, i) => i).sort((a, b) => probs[b + 1] - probs[a + 1]);
    const bestP = probs[order[0] + 1];
    const thr = this.cfg.relevanceThreshold;
    if (bestP < thr) return silent(`low relevance: best ${bestP.toFixed(2)} < threshold ${thr.toFixed(2)}`);

    const rejected: string[] = [];
    for (const i of order) {
      const c = cands[i];
      const p = probs[i + 1];
      const label = labels[i + 1];
      if (p < thr) {
        rejected.push(`${label} below threshold (${p.toFixed(2)})`);
        break; // sorted: everything after is lower
      }
      if (p - pAbstain < this.cfg.marginOverAbstain) {
        rejected.push(`${label} does not beat NO_SUGGESTION by ${this.cfg.marginOverAbstain.toFixed(2)} `
          + `(${p.toFixed(2)} vs ${pAbstain.toFixed(2)})`);
        break;
      }
      if (blocked !== null && i < blocked.length && blocked[i]) {
        rejected.push(`${label} references a blocked source`);
        continue;
      }
      const key = candidateKey(c);
      const dup = this.duplicateReason(key, now);
      if (dup) {
        rejected.push(`${label} ${dup}`);
        continue;
      }
      if (obs !== null && !verifyTarget(c, obs, this.iouThreshold)) {
        rejected.push(`${label} target no longer on screen`);
        continue;
      }
      if (record) {
        this.shown.set(key, now);
        this.lastShownAt = now;
      }
      const skipped = rejected.length ? ` (skipped: ${rejected.join("; ")})` : "";
      return make(c, p, `${label} p=${p.toFixed(2)} >= threshold ${thr.toFixed(2)} `
        + `and beats NO_SUGGESTION (${pAbstain.toFixed(2)})${skipped}`);
    }
    return silent(rejected.join("; ") || "no eligible candidate");
  }
}
