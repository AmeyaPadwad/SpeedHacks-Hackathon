/**
 * Temporal encoder (port of `temporal_encoder.py`): current screen state + recent history -> temporal
 * state `[D]`.
 *
 * The sequence is `history[0..T-1]` (oldest first) followed by the current state. Both variants are causal
 * (the GRU by construction, the transformer through a causal mask). Python left-pads and masks missing
 * steps; here the history is just the valid steps, which is the same computation. App identity enters as
 * a learned embedding added to the current step.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import { causalPaddingMask, Embedding, LayerNorm, Linear, LinearNorm, Params, TransformerEncoder } from "./nn";

/** `nn.GRU(D, D)`, one layer, PyTorch gate order (r, z, n). */
class Gru {
  readonly wi: tf.Variable;
  readonly wh: tf.Variable;
  readonly bi: tf.Variable;
  readonly bh: tf.Variable;

  constructor(p: Params, name: string, readonly d: number) {
    const k = 1 / Math.sqrt(d);
    this.wi = p.uniform(`${name}.wi`, [d, 3 * d], k);
    this.wh = p.uniform(`${name}.wh`, [d, 3 * d], k);
    this.bi = p.uniform(`${name}.bi`, [3 * d], k);
    this.bh = p.uniform(`${name}.bh`, [3 * d], k);
  }

  /** Run over `x [S, D]` from a zero state; returns the last hidden state `[1, D]`. */
  last(x: tf.Tensor2D): tf.Tensor2D {
    const d = this.d;
    const gi = tf.add(tf.matMul(x, this.wi as tf.Tensor2D), this.bi) as tf.Tensor2D; // [S, 3D]
    let h = tf.zeros([1, d]) as tf.Tensor2D;
    for (let t = 0; t < x.shape[0]; t++) {
      const it = tf.slice(gi, [t, 0], [1, 3 * d]);
      const ht = tf.add(tf.matMul(h, this.wh as tf.Tensor2D), this.bh);
      const [ir, iz, inn] = tf.split(it, 3, 1);
      const [hr, hz, hn] = tf.split(ht, 3, 1);
      const r = tf.sigmoid(tf.add(ir, hr));
      const z = tf.sigmoid(tf.add(iz, hz));
      const n = tf.tanh(tf.add(inn, tf.mul(r, hn)));
      h = tf.add(tf.mul(tf.sub(1, z), n), tf.mul(z, h)) as tf.Tensor2D;
    }
    return h;
  }
}

export class TemporalEncoder {
  readonly kind: "gru" | "transformer";
  readonly maxSteps: number;
  readonly dim: number;
  readonly appEmbed: Embedding;
  readonly inProj: LinearNorm;
  readonly gru?: Gru;
  readonly ageEmbed?: Embedding;
  readonly encoder?: TransformerEncoder;
  readonly outProj: Linear;
  readonly outNorm: LayerNorm;

  constructor(p: Params, cfg: ModelConfig) {
    if (cfg.temporalKind !== "gru" && cfg.temporalKind !== "transformer") {
      throw new Error(`unknown temporalKind ${String(cfg.temporalKind)}`);
    }
    const D = cfg.dModel;
    this.dim = D;
    this.kind = cfg.temporalKind;
    this.maxSteps = cfg.historyLen + 1;
    this.appEmbed = new Embedding(p, "temporal.app", cfg.nApps, D, { std: 0.02 });
    this.inProj = new LinearNorm(p, "temporal.in", D, D);
    if (this.kind === "gru") {
      this.gru = new Gru(p, "temporal.gru", D);
    } else {
      // step position counted back from "now", so the current step is always age 0
      this.ageEmbed = new Embedding(p, "temporal.age", this.maxSteps, D, { std: 0.02 });
      this.encoder = new TransformerEncoder(p, "temporal.encoder", D, cfg.nHeads, Math.max(1, cfg.fusionLayers));
    }
    this.outProj = new Linear(p, "temporal.out", D, D);
    this.outNorm = new LayerNorm(p, "temporal.outNorm", D);
  }

  /**
   * Encode `screenState [D]` with `history` (previous states, oldest first; only the most recent
   * `historyLen` are used) and an optional hashed `appId` bucket. Returns `[D]`.
   */
  apply(screenState: tf.Tensor1D, history: Float32Array[], appId?: number): tf.Tensor1D {
    const D = this.dim;
    const hist = history.slice(Math.max(0, history.length - (this.maxSteps - 1)));
    for (const h of hist) if (h.length !== D) throw new Error(`history state has dim ${h.length}, expected ${D}`);
    let current: tf.Tensor1D = screenState;
    if (appId !== undefined && appId !== null) {
      const id = ((Math.floor(appId) % this.appEmbed.table.shape[0]) + this.appEmbed.table.shape[0]) % this.appEmbed.table.shape[0];
      current = tf.add(current, this.appEmbed.row(id));
    }
    const S = hist.length + 1;
    const seq = hist.length
      ? tf.concat([tf.tensor2d(concat(hist, D), [hist.length, D]), current.expandDims(0)], 0)
      : (current.expandDims(0) as tf.Tensor2D);
    let x = this.inProj.apply(seq as tf.Tensor2D);
    let last: tf.Tensor2D;
    if (this.gru) {
      last = this.gru.last(x);
    } else {
      const ages = Array.from({ length: S }, (_, i) => S - 1 - i);
      x = tf.add(x, this.ageEmbed!.gather(ages)) as tf.Tensor2D;
      const out = this.encoder!.apply(x, causalPaddingMask(new Array<boolean>(S).fill(false)));
      last = tf.slice(out, [S - 1, 0], [1, D]);
    }
    return this.outNorm.apply(tf.add(screenState, this.outProj.apply(last.squeeze([0]) as tf.Tensor1D))) as tf.Tensor1D;
  }
}

function concat(rows: Float32Array[], d: number): Float32Array {
  const out = new Float32Array(rows.length * d);
  rows.forEach((r, i) => out.set(r, i * d));
  return out;
}
