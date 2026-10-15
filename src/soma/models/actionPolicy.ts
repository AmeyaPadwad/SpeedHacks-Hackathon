/**
 * Action policy / ranker (port of `action_policy.py`): context + candidates -> logits `[1 + A]`.
 *
 * Index 0 is the trainable NO_SUGGESTION logit; `1..A` score the candidates. Blocked candidates get
 * {@link BLOCKED_LOGIT}, so when every candidate is blocked the softmax puts ~all mass on NO_SUGGESTION.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import { NEG_INF } from "./math";
import { Embedding, gelu, Linear, LinearNorm, Params } from "./nn";

export const NO_SUGGESTION_INDEX = 0;
/** Logit for blocked candidates; finite (see `NEG_INF`). */
export const BLOCKED_LOGIT = NEG_INF;

export class ActionPolicy {
  readonly typeEmbed: Embedding;
  readonly candProj: LinearNorm;
  readonly ctxProj: LinearNorm;
  readonly s1: Linear;
  readonly s2: Linear;
  readonly s3: Linear;
  readonly a1: Linear;
  readonly a2: Linear;
  readonly abstainBias: tf.Variable;
  readonly nTypes: number;

  constructor(p: Params, cfg: ModelConfig) {
    const D = cfg.dModel;
    const H = Math.floor(D / 2);
    this.nTypes = cfg.nActionTypes;
    this.typeEmbed = new Embedding(p, "policy.type", cfg.nActionTypes, D, { std: 0.02 });
    this.candProj = new LinearNorm(p, "policy.cand", D, D);
    this.ctxProj = new LinearNorm(p, "policy.ctx", D, D);
    this.s1 = new Linear(p, "policy.score1", 3 * D, D);
    this.s2 = new Linear(p, "policy.score2", D, H);
    this.s3 = new Linear(p, "policy.score3", H, 1);
    this.a1 = new Linear(p, "policy.abstain1", D, H);
    this.a2 = new Linear(p, "policy.abstain2", H, 1);
    this.abstainBias = p.constant("policy.abstainBias", [1], 0);
  }

  /**
   * Score `candidates [A, D]` (`types [A]` ActionType indices, `blocked [A]` true = blocked) against
   * `context [D]`. `A` may be 0. Returns `[1 + A]`, index 0 = NO_SUGGESTION.
   */
  apply(context: tf.Tensor1D, candidates: tf.Tensor2D, types?: number[], blocked?: boolean[]): tf.Tensor1D {
    const ctx = this.ctxProj.apply(context);
    const noSugg = tf.add(this.a2.apply(gelu(this.a1.apply(ctx))), this.abstainBias) as tf.Tensor1D; // [1]
    const A = candidates.shape[0];
    if (A === 0) return noSugg;
    let cand: tf.Tensor2D = candidates;
    if (types) {
      if (types.length !== A) throw new Error("candidateTypes length does not match candidates");
      const ids = types.map((t) => Math.min(Math.max(Math.floor(t), 0), this.nTypes - 1));
      cand = tf.add(cand, this.typeEmbed.gather(ids));
    }
    cand = this.candProj.apply(cand);
    const ctxE = tf.tile(ctx.expandDims(0), [A, 1]) as tf.Tensor2D;
    const feats = tf.concat([ctxE, cand, tf.mul(ctxE, cand)], 1) as tf.Tensor2D;
    let scores = this.s3.apply(gelu(this.s2.apply(gelu(this.s1.apply(feats))))).reshape([A]) as tf.Tensor1D;
    if (blocked?.some(Boolean)) {
      scores = tf.where(tf.tensor1d(blocked.map((b) => (b ? 1 : 0)), "bool"), tf.fill([A], BLOCKED_LOGIT), scores);
    }
    return tf.concat([noSugg, scores]);
  }
}
