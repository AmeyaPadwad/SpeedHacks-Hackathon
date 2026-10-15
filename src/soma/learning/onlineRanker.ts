/**
 * Stage C online ranker over embeddings: a small tfjs logistic model trained incrementally from explicit
 * feedback that returns an additive per-candidate logit bias. It never touches the base model's weights.
 *
 * Same semantics as `online_ranker.py` (whose arm-based port is {@link OnlinePreferenceRanker}), but keyed
 * on what the model sees rather than on action types and candidate keys, so it generalises to candidates
 * it has never shown:
 *  - Candidate slot i gets `clip(w . f(context, candidate_i) + b)` where `f` is `[ctx (*) cand, cand]` of the
 *    L2-normalised vectors. Weights start at zero, so with no evidence every bias is exactly 0.
 *  - Slot 0 (NO_SUGGESTION) is a per-app Beta-Bernoulli abstain arm, exactly as in Python: a rejection is
 *    evidence that silence was better, an accept/edit evidence against it.
 *  - IGNORED outcomes are ambiguous and never train. Accepted = positive (weight 1), edited = positive
 *    (weight 0.5), rejected = negative (weight 1).
 *  - Every bias is bounded by `maxBias`, and all evidence decays (weights and arms shrink by `decay` on each
 *    update) so stale preferences fade.
 *  - Each update is one SGD step on the new example plus a few replayed past ones (in-memory
 *    {@link ReplayBuffer}, never persisted), which keeps old preferences from being forgotten at once.
 */
import * as tf from "@tensorflow/tfjs";
import { l2normalize, Rng } from "../models/math";
import { loadWeights, saveWeights, type WeightsTarget } from "../models/weightsIO";
import { FeedbackOutcome } from "../types";
import { ABSTAIN_EVIDENCE, applyBias } from "./preferenceRanker";
import { ReplayBuffer } from "./replayBuffer";

export const RANKER_VERSION = "soma-online-ranker-1";

/** (label, weight) per trainable outcome. */
const LABELS: Partial<Record<FeedbackOutcome, [number, number]>> = {
  [FeedbackOutcome.ACCEPTED]: [1, 1],
  [FeedbackOutcome.EDITED]: [1, 0.5],
  [FeedbackOutcome.REJECTED]: [0, 1],
};

export interface OnlineRankerOptions {
  /** SGD learning rate. */
  lr?: number;
  /** Absolute clip on every returned bias. */
  maxBias?: number;
  /** Evidence multiplier applied on each update, in (0, 1]. */
  decay?: number;
  /** Beta prior pseudo-count for the abstain arm. */
  prior?: number;
  /** Multiplier on the abstain arm's log-odds. */
  abstainScale?: number;
  /** Past examples replayed with each update. */
  replay?: number;
  /** Replay buffer capacity. */
  replayCapacity?: number;
  seed?: number;
}

interface RankerExample {
  app: string;
  task: string;
  usableForTraining: boolean;
  features: Float32Array;
  label: number;
  weight: number;
}

export class OnlineRanker {
  readonly lr: number;
  readonly maxBias: number;
  readonly decay: number;
  readonly prior: number;
  readonly abstainScale: number;
  readonly replayN: number;
  updates = 0;
  private w: tf.Variable;
  private b: tf.Variable;
  private optimizer: tf.SGDOptimizer;
  private abstain = new Map<string, [number, number]>();
  private replay: ReplayBuffer<RankerExample>;
  private rng: Rng;

  constructor(readonly dim: number, opts: OnlineRankerOptions = {}) {
    this.lr = opts.lr ?? 0.5;
    this.maxBias = opts.maxBias ?? 2;
    this.decay = opts.decay ?? 0.98;
    this.prior = opts.prior ?? 1;
    this.abstainScale = opts.abstainScale ?? 1;
    this.replayN = opts.replay ?? 7;
    if (!(this.decay > 0 && this.decay <= 1)) throw new Error("decay must be in (0, 1]");
    this.w = tf.variable(tf.zeros([2 * dim, 1]), true);
    this.b = tf.variable(tf.zeros([1]), true);
    this.optimizer = tf.train.sgd(this.lr);
    this.rng = new Rng(opts.seed ?? 0);
    this.replay = new ReplayBuffer<RankerExample>(opts.replayCapacity ?? 256, { seed: opts.seed ?? 0 });
  }

  /** `[ctx (*) cand * sqrt(D), cand]` of the unit vectors, `2D` long. */
  features(context: Float32Array, candidate: Float32Array): Float32Array {
    const D = this.dim;
    if (context.length !== D || candidate.length !== D) throw new Error(`expected vectors of dim ${D}`);
    const c = l2normalize(context);
    const a = l2normalize(candidate);
    const out = new Float32Array(2 * D);
    const s = Math.sqrt(D);
    for (let i = 0; i < D; i++) {
      out[i] = c[i] * a[i] * s;
      out[D + i] = a[i];
    }
    return out;
  }

  private abstainBias(app: string): number {
    const [s, f] = this.abstain.get((app || "").toLowerCase()) ?? [0, 0];
    return this.clip(this.abstainScale * Math.log((s + this.prior) / (f + this.prior)));
  }

  private clip(x: number): number {
    return Math.max(-this.maxBias, Math.min(this.maxBias, x));
  }

  /** Additive logit bias `[1 + A]` (index 0 = NO_SUGGESTION) for `candidateEmbeddings` in `context`. */
  bias(context: Float32Array, candidateEmbeddings: Float32Array[], app = ""): Float32Array {
    const out = new Float32Array(candidateEmbeddings.length + 1);
    out[0] = this.abstainBias(app);
    if (!candidateEmbeddings.length) return out;
    const feats = candidateEmbeddings.map((c) => this.features(context, c));
    const scores = tf.tidy(() => {
      const x = tf.tensor2d(concat(feats), [feats.length, 2 * this.dim]);
      return tf.add(tf.matMul(x, this.w as tf.Tensor2D), this.b).dataSync();
    });
    scores.forEach((s, i) => (out[i + 1] = this.clip(s)));
    return out;
  }

  /** `logits + bias` leaving blocked entries untouched (see `applyBias`). */
  static apply(logits: Float32Array, bias: Float32Array): Float32Array {
    return applyBias(logits, bias);
  }

  /**
   * Learn from one explicit outcome on a shown candidate. IGNORED (or unknown) outcomes are a no-op and
   * resolve to false. `app` keys the abstain arm and the replay stratum; `task` defaults to "".
   */
  async update(context: Float32Array, candidateEmbedding: Float32Array, outcome: FeedbackOutcome,
    opts: { app?: string; task?: string } = {}): Promise<boolean> {
    const lab = LABELS[outcome];
    if (!lab) return false;
    const app = (opts.app ?? "").toLowerCase();
    const ex: RankerExample = {
      app, task: opts.task ?? "", usableForTraining: true,
      features: this.features(context, candidateEmbedding), label: lab[0], weight: lab[1],
    };
    // decay all evidence, then add the new abstain evidence
    for (const a of this.abstain.values()) {
      a[0] *= this.decay;
      a[1] *= this.decay;
    }
    const [s, f] = ABSTAIN_EVIDENCE[outcome]!;
    const arm = this.abstain.get(app) ?? [0, 0];
    this.abstain.set(app, [arm[0] + s, arm[1] + f]);

    const batch = [ex, ...(this.replay.size ? this.replay.sample(Math.min(this.replayN, this.replay.size), this.rng, 0) : [])];
    this.replay.add(ex);
    tf.tidy(() => {
      this.w.assign(tf.mul(this.w, this.decay));
      this.b.assign(tf.mul(this.b, this.decay));
      const x = tf.tensor2d(concat(batch.map((e) => e.features)), [batch.length, 2 * this.dim]);
      const y = tf.tensor2d(batch.map((e) => e.label), [batch.length, 1]);
      const wt = tf.tensor2d(batch.map((e) => e.weight), [batch.length, 1]);
      this.optimizer.minimize(() => {
        const logits = tf.add(tf.matMul(x, this.w as tf.Tensor2D), this.b);
        const loss = tf.losses.sigmoidCrossEntropy(y, logits, undefined, 0, tf.Reduction.NONE);
        return tf.mean(tf.mul(loss, wt)) as tf.Scalar;
      }, false, [this.w, this.b]);
    });
    this.updates += 1;
    return true;
  }

  /** Forget every preference (weights, abstain arms, replay). */
  reset(): void {
    this.w.assign(tf.zerosLike(this.w));
    this.b.assign(tf.zerosLike(this.b));
    this.abstain.clear();
    this.replay.clear();
    this.updates = 0;
  }

  /** Save weights and abstain arms (never the replay buffer) to a tfjs URL or IOHandler. */
  async save(url: WeightsTarget): Promise<void> {
    await saveWeights(url, { w: this.w, b: this.b }, {
      format: RANKER_VERSION, dim: this.dim, updates: this.updates,
      abstain: Object.fromEntries(this.abstain),
    });
  }

  /** Load what {@link save} wrote; false when absent, another version, or another `dim`. */
  async load(url: WeightsTarget): Promise<boolean> {
    const got = await loadWeights(url, RANKER_VERSION);
    if (!got) return false;
    try {
      const { w, b } = got.tensors;
      if (got.meta.dim !== this.dim || !w || !b || !tf.util.arraysEqual(w.shape, this.w.shape)
        || !tf.util.arraysEqual(b.shape, this.b.shape)) return false;
      this.w.assign(w);
      this.b.assign(b);
      const arms = (got.meta.abstain ?? {}) as Record<string, [number, number]>;
      this.abstain = new Map(Object.entries(arms).map(([k, v]) => [k, [Number(v[0]), Number(v[1])]]));
      this.updates = Number(got.meta.updates ?? 0);
      return true;
    } finally {
      tf.dispose(got.tensors);
    }
  }

  dispose(): void {
    this.w.dispose();
    this.b.dispose();
    this.optimizer.dispose();
  }
}

function concat(rows: Float32Array[]): Float32Array {
  const n = rows[0]?.length ?? 0;
  const out = new Float32Array(rows.length * n);
  rows.forEach((r, i) => out.set(r, i * n));
  return out;
}
