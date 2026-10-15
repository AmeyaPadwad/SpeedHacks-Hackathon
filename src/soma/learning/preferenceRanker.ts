/**
 * Stage C: lightweight online preference model that never touches base weights (port of
 * `online_ranker.OnlinePreferenceRanker`).
 *
 * Per-app Beta-Bernoulli "arms" track explicit feedback for
 *  - each action type in that app (`type` arms),
 *  - each concrete candidate key in that app (`key` arms), and
 *  - abstention in that app (`abstain` arm: a rejection is evidence that silence was better; an
 *    accept/edit is evidence against silence).
 *
 * `bias(candidates, app)` returns an additive `[A + 1]` logit bias (index 0 = NO_SUGGESTION) equal to the
 * clipped posterior log-odds of each arm. With no evidence every bias is exactly 0, magnitudes are bounded
 * by `maxBias`, and all evidence decays geometrically so stale preferences fade. IGNORED outcomes are
 * ambiguous and are not counted. Pure arithmetic: no tfjs.
 */
import { ActionType, type Candidate, candidateKey, FeedbackOutcome } from "../types";

/** Logits at or below this are treated as blocked candidates (the model uses ~-1e4). */
export const BLOCKED_THRESHOLD = -1e3;

/** (success, failure) increments for a shown candidate's arms. */
export const CANDIDATE_EVIDENCE: Partial<Record<FeedbackOutcome, [number, number]>> = {
  [FeedbackOutcome.ACCEPTED]: [1, 0],
  [FeedbackOutcome.EDITED]: [0.5, 0],
  [FeedbackOutcome.REJECTED]: [0, 1],
};

/** (success, failure) for the abstain arm: success = "silence would have been better". */
export const ABSTAIN_EVIDENCE: Partial<Record<FeedbackOutcome, [number, number]>> = {
  [FeedbackOutcome.ACCEPTED]: [0, 1],
  [FeedbackOutcome.EDITED]: [0, 0.5],
  [FeedbackOutcome.REJECTED]: [1, 0],
};

export interface PreferenceRankerOptions {
  /** Pseudo-count on each side of every arm (Beta(prior, prior)). */
  prior?: number;
  /** Multiplier on candidate log-odds. */
  scale?: number;
  /** Multiplier on the abstain arm's log-odds. */
  abstainScale?: number;
  /** Share of the key arm relative to the type arm. */
  keyWeight?: number;
  /** Absolute clip on every returned bias. */
  maxBias?: number;
  /** Evidence multiplier applied to all arms on each update, in (0, 1]. */
  decay?: number;
}

export interface PreferenceRankerState {
  prior: number;
  scale: number;
  abstainScale: number;
  keyWeight: number;
  maxBias: number;
  decay: number;
  updates: number;
  arms: Record<string, [number, number]>;
}

/**
 * `logits + bias`, leaving blocked entries (non-finite or at/below {@link BLOCKED_THRESHOLD}) untouched.
 * Returns a new array; `bias` shorter than `logits` counts as 0 for the missing slots.
 */
export function applyBias(logits: Float32Array, bias: Float32Array): Float32Array {
  const out = Float32Array.from(logits);
  for (let i = 0; i < out.length; i++) {
    const open = Number.isFinite(out[i]) && out[i] > BLOCKED_THRESHOLD;
    if (open && i < bias.length) out[i] += bias[i];
  }
  return out;
}

const isOutcome = (o: unknown): o is FeedbackOutcome => Object.values(FeedbackOutcome).includes(o as FeedbackOutcome);

export class OnlinePreferenceRanker {
  prior: number;
  scale: number;
  abstainScale: number;
  keyWeight: number;
  maxBias: number;
  decay: number;
  updates = 0;
  private arms = new Map<string, [number, number]>();

  constructor(opts: PreferenceRankerOptions = {}) {
    this.prior = opts.prior ?? 1;
    this.scale = opts.scale ?? 1;
    this.abstainScale = opts.abstainScale ?? 1;
    this.keyWeight = opts.keyWeight ?? 0.5;
    this.maxBias = opts.maxBias ?? 2;
    this.decay = opts.decay ?? 0.98;
    if (!(this.decay > 0 && this.decay <= 1)) throw new Error("decay must be in (0, 1]");
  }

  private static arm(kind: string, app: string, name = ""): string {
    return `${kind}|${(app || "").toLowerCase()}|${name}`;
  }

  private add(arm: string, s: number, f: number): void {
    const a = this.arms.get(arm) ?? [0, 0];
    a[0] += s;
    a[1] += f;
    this.arms.set(arm, a);
  }

  private logOdds(arm: string): number {
    const [s, f] = this.arms.get(arm) ?? [0, 0];
    return Math.log((s + this.prior) / (f + this.prior));
  }

  private clip(x: number): number {
    return Math.max(-this.maxBias, Math.min(this.maxBias, x));
  }

  /** Count one explicit outcome. Returns false (no-op) for ignored/unknown outcomes. */
  update(app: string, action: ActionType | string, outcome: FeedbackOutcome | string, key?: string | null): boolean {
    if (!isOutcome(outcome) || !CANDIDATE_EVIDENCE[outcome]) return false;
    for (const a of this.arms.values()) {
      a[0] *= this.decay;
      a[1] *= this.decay;
    }
    if (action !== ActionType.NO_SUGGESTION) {
      const [s, f] = CANDIDATE_EVIDENCE[outcome]!;
      this.add(OnlinePreferenceRanker.arm("type", app, action), s, f);
      if (key) this.add(OnlinePreferenceRanker.arm("key", app, key), s, f);
    }
    const [s, f] = ABSTAIN_EVIDENCE[outcome]!;
    this.add(OnlinePreferenceRanker.arm("abstain", app), s, f);
    this.updates += 1;
    return true;
  }

  /** Convenience: update from a shown candidate. */
  updateCandidate(candidate: Candidate, app: string, outcome: FeedbackOutcome | string): boolean {
    return this.update(app, candidate.action, outcome, candidateKey(candidate));
  }

  candidateBias(app: string, action: ActionType | string, key?: string | null): number {
    let b = this.scale * this.logOdds(OnlinePreferenceRanker.arm("type", app, action));
    if (key) b += this.scale * this.keyWeight * this.logOdds(OnlinePreferenceRanker.arm("key", app, key));
    return this.clip(b);
  }

  abstainBias(app: string): number {
    return this.clip(this.abstainScale * this.logOdds(OnlinePreferenceRanker.arm("abstain", app)));
  }

  /**
   * Additive logit bias `[A + 1]` (index 0 = NO_SUGGESTION). `candidates` may contain null for empty slots
   * (bias 0); `nSlots` pads the result to a fixed `A`.
   */
  bias(candidates: (Candidate | null)[], app: string, nSlots = 0, keys?: (string | null)[]): Float32Array {
    const out = new Float32Array(Math.max(candidates.length, nSlots) + 1);
    out[0] = this.abstainBias(app);
    candidates.forEach((c, i) => {
      if (c) out[i + 1] = this.candidateBias(app, c.action, keys ? keys[i] : candidateKey(c));
    });
    return out;
  }

  /** Bias `[A + 1]` from action types only (no candidate keys). */
  biasForTypes(types: (ActionType | string | null)[], app: string): Float32Array {
    const out = new Float32Array(types.length + 1);
    out[0] = this.abstainBias(app);
    types.forEach((t, i) => {
      if (t && t !== ActionType.NO_SUGGESTION) out[i + 1] = this.candidateBias(app, t);
    });
    return out;
  }

  /** See {@link applyBias}. */
  static apply(logits: Float32Array, bias: Float32Array): Float32Array {
    return applyBias(logits, bias);
  }

  stateDict(): PreferenceRankerState {
    const arms: Record<string, [number, number]> = {};
    for (const [k, v] of this.arms) arms[k] = [v[0], v[1]];
    return {
      prior: this.prior, scale: this.scale, abstainScale: this.abstainScale, keyWeight: this.keyWeight,
      maxBias: this.maxBias, decay: this.decay, updates: this.updates, arms,
    };
  }

  loadStateDict(state: PreferenceRankerState): void {
    this.prior = Number(state.prior);
    this.scale = Number(state.scale);
    this.abstainScale = Number(state.abstainScale);
    this.keyWeight = Number(state.keyWeight);
    this.maxBias = Number(state.maxBias);
    this.decay = Number(state.decay);
    this.updates = Number(state.updates ?? 0);
    this.arms = new Map(Object.entries(state.arms).map(([k, v]) => [k, [Number(v[0]), Number(v[1])]]));
  }

  /** Forget all evidence, or only one app's. */
  reset(app?: string): void {
    if (app === undefined) {
      this.arms.clear();
      return;
    }
    const needle = (app || "").toLowerCase();
    for (const k of [...this.arms.keys()]) if (k.split("|")[1] === needle) this.arms.delete(k);
  }
}
