/**
 * Minimal tfjs building blocks for SOMA (port of the torch.nn pieces SOMA uses and of `attention.py`).
 *
 * Everything works on one sample at a time: token sequences are `[S, D]` tensors, vectors `[D]`. Weights are
 * `tf.Variable`s registered in a {@link Params} store under stable names, initialised from a seeded
 * {@link Rng} with PyTorch's default schemes, so the same seed always builds the same model. Padding masks
 * are plain `boolean[]` with `true` = PAD, as in the Python code. Inference only: dropout is the identity.
 */
import * as tf from "@tensorflow/tfjs";
import { NEG_INF, Rng } from "./math";

// ---------------------------------------------------------------- parameters

/** Named, seeded weight store. Creation order fixes the RNG stream, so build modules deterministically. */
export class Params {
  readonly vars = new Map<string, tf.Variable>();

  constructor(readonly rng: Rng) {}

  private add(name: string, shape: number[], fill: (i: number) => number): tf.Variable {
    if (this.vars.has(name)) throw new Error(`duplicate parameter ${name}`);
    const n = shape.reduce((a, b) => a * b, 1);
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) values[i] = fill(i);
    const init = tf.tensor(values, shape, "float32");
    const v = tf.variable(init, true, `${name}_${Params.uid++}`);
    init.dispose();
    this.vars.set(name, v);
    return v;
  }

  private static uid = 0;

  normal(name: string, shape: number[], std = 1): tf.Variable {
    return this.add(name, shape, () => this.rng.normal() * std);
  }

  uniform(name: string, shape: number[], bound: number): tf.Variable {
    return this.add(name, shape, () => (this.rng.next() * 2 - 1) * bound);
  }

  constant(name: string, shape: number[], value: number): tf.Variable {
    return this.add(name, shape, () => value);
  }

  /** Snapshot of every weight (clones; caller disposes). */
  snapshot(): tf.NamedTensorMap {
    const out: tf.NamedTensorMap = {};
    for (const [k, v] of this.vars) out[k] = v.clone();
    return out;
  }

  /** Assign weights by name. Throws (and assigns nothing) if a name is missing or a shape differs. */
  assign(tensors: tf.NamedTensorMap): void {
    for (const [k, v] of this.vars) {
      const t = tensors[k];
      if (!t) throw new Error(`missing weight ${k}`);
      if (!tf.util.arraysEqual(t.shape, v.shape)) throw new Error(`shape mismatch for ${k}: ${t.shape} vs ${v.shape}`);
    }
    for (const [k, v] of this.vars) v.assign(tensors[k] as tf.Tensor);
  }

  dispose(): void {
    for (const v of this.vars.values()) v.dispose();
    this.vars.clear();
  }
}

// ---------------------------------------------------------------- masks

/** True where every position is PAD (or there are none). */
export const allMasked = (pad: boolean[]): boolean => pad.every(Boolean);

/** Unmask position 0 of an all-PAD mask so attention stays finite; callers zero the padded outputs after. */
export function safePaddingMask(pad: boolean[]): boolean[] {
  if (!pad.length || !allMasked(pad)) return pad;
  const out = pad.slice();
  out[0] = false;
  return out;
}

/** `[1, S]` additive key mask: 0 keep, `NEG_INF` drop. */
export function additiveKeyMask(pad: boolean[]): tf.Tensor2D {
  return tf.tensor2d(pad.map((p) => (p ? NEG_INF : 0)), [1, pad.length]);
}

/**
 * `[S, S]` additive causal mask that also hides PAD keys, except every query always sees itself so no row is
 * ever fully masked (port of `causal_padding_attn_mask`).
 */
export function causalPaddingMask(pad: boolean[]): tf.Tensor2D {
  const S = pad.length;
  const m = new Float32Array(S * S);
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < S; j++) {
      const blocked = (j > i || pad[j]) && i !== j;
      m[i * S + j] = blocked ? NEG_INF : 0;
    }
  }
  return tf.tensor2d(m, [S, S]);
}

/** Zero the rows of `x [S, D]` where `pad` is true. */
export function zeroPadded(x: tf.Tensor2D, pad: boolean[]): tf.Tensor2D {
  if (!pad.some(Boolean)) return x;
  const keep = tf.tensor2d(pad.map((p) => (p ? 0 : 1)), [pad.length, 1]);
  return tf.mul(x, keep);
}

// ---------------------------------------------------------------- layers

/** Exact (erf) GELU, as `nn.GELU()`. */
export function gelu<T extends tf.Tensor>(x: T): T {
  return tf.tidy(() => tf.mul(tf.mul(x, 0.5), tf.add(1, tf.erf(tf.div(x, Math.SQRT2))))) as T;
}

/** `nn.Linear`: weight stored `[in, out]`, kaiming-uniform init `U(+-1/sqrt(in))` for weight and bias. */
export class Linear {
  readonly w: tf.Variable;
  readonly b: tf.Variable;

  constructor(p: Params, name: string, inF: number, outF: number, opts: { wBound?: number; zeroBias?: boolean } = {}) {
    const bound = 1 / Math.sqrt(inF);
    this.w = p.uniform(`${name}.w`, [inF, outF], opts.wBound ?? bound);
    this.b = opts.zeroBias ? p.constant(`${name}.b`, [outF], 0) : p.uniform(`${name}.b`, [outF], bound);
  }

  apply<T extends tf.Tensor>(x: T): T {
    if (x.rank === 1) return tf.add(tf.matMul(x.expandDims(0) as tf.Tensor2D, this.w as tf.Tensor2D), this.b).squeeze([0]) as T;
    return tf.add(tf.matMul(x as unknown as tf.Tensor2D, this.w as tf.Tensor2D), this.b) as unknown as T;
  }
}

/** `nn.LayerNorm` over the last axis (eps 1e-5). */
export class LayerNorm {
  readonly g: tf.Variable;
  readonly b: tf.Variable;

  constructor(p: Params, name: string, dim: number, readonly eps = 1e-5) {
    this.g = p.constant(`${name}.g`, [dim], 1);
    this.b = p.constant(`${name}.b`, [dim], 0);
  }

  apply<T extends tf.Tensor>(x: T): T {
    const { mean, variance } = tf.moments(x, -1, true);
    return tf.add(tf.mul(tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, this.eps))), this.g), this.b) as T;
  }
}

/** `nn.Embedding` (default `N(0, std)`; `paddingIdx` row zeroed). */
export class Embedding {
  readonly table: tf.Variable;

  constructor(p: Params, name: string, n: number, dim: number, opts: { std?: number; paddingIdx?: number } = {}) {
    this.table = p.normal(`${name}.table`, [n, dim], opts.std ?? 1);
    if (opts.paddingIdx !== undefined) {
      const idx = opts.paddingIdx;
      tf.tidy(() => {
        const keep = tf.oneHot([idx], n).reshape([n, 1]);
        this.table.assign(tf.mul(this.table, tf.sub(1, keep)));
      });
    }
  }

  /** Rows `ids` -> `[len, dim]`. */
  gather(ids: number[] | Int32Array): tf.Tensor2D {
    return tf.gather(this.table, tf.tensor1d(Array.from(ids), "int32")) as tf.Tensor2D;
  }

  row(i: number): tf.Tensor1D {
    return tf.gather(this.table, tf.tensor1d([i], "int32")).squeeze([0]) as tf.Tensor1D;
  }
}

/** Sequential Linear + LayerNorm (`nn.Sequential(nn.Linear(D, D), nn.LayerNorm(D))`). */
export class LinearNorm {
  readonly lin: Linear;
  readonly norm: LayerNorm;

  constructor(p: Params, name: string, inF: number, outF: number) {
    this.lin = new Linear(p, `${name}.lin`, inF, outF);
    this.norm = new LayerNorm(p, `${name}.norm`, outF);
  }

  apply<T extends tf.Tensor>(x: T): T {
    return this.norm.apply(this.lin.apply(x));
  }
}

/** Split `[S, D]` into heads `[H, S, D/H]`. */
function heads(x: tf.Tensor2D, h: number): tf.Tensor3D {
  const [S, D] = x.shape;
  return tf.transpose(tf.reshape(x, [S, h, D / h]), [1, 0, 2]);
}

/** `[H, S, dh]` -> `[S, H*dh]`. */
function merge(x: tf.Tensor3D): tf.Tensor2D {
  const [h, S, dh] = x.shape;
  return tf.reshape(tf.transpose(x, [1, 0, 2]), [S, h * dh]);
}

/** Scaled dot-product attention per head; `bias` broadcasts to `[H, Sq, Sk]`. Returns output and probs. */
function sdpa(q: tf.Tensor3D, k: tf.Tensor3D, v: tf.Tensor3D, bias?: tf.Tensor): { out: tf.Tensor3D; probs: tf.Tensor3D } {
  let scores = tf.div(tf.matMul(q, k, false, true), Math.sqrt(q.shape[2])) as tf.Tensor3D;
  if (bias) scores = tf.add(scores, bias);
  const probs = tf.softmax(scores);
  return { out: tf.matMul(probs, v), probs };
}

/** `nn.MultiheadAttention` (xavier-uniform in-projection, zero biases). */
export class MultiheadAttention {
  readonly inW: tf.Variable;
  readonly inB: tf.Variable;
  readonly out: Linear;

  constructor(p: Params, name: string, readonly d: number, readonly h: number) {
    if (d % h) throw new Error(`d_model=${d} not divisible by n_heads=${h}`);
    this.inW = p.uniform(`${name}.in.w`, [d, 3 * d], Math.sqrt(6 / (4 * d)));
    this.inB = p.constant(`${name}.in.b`, [3 * d], 0);
    this.out = new Linear(p, `${name}.out`, d, d, { zeroBias: true });
  }

  /**
   * Attend `q [Sq, D]` over `k, v [Sk, D]`. `bias` is an additive mask broadcastable to `[Sq, Sk]`.
   * Returns the output `[Sq, D]` and head-averaged weights `[Sq, Sk]`.
   */
  apply(q: tf.Tensor2D, k: tf.Tensor2D, v: tf.Tensor2D, bias?: tf.Tensor): { out: tf.Tensor2D; weights: tf.Tensor2D } {
    const [wq, wk, wv] = tf.split(this.inW, 3, 1) as tf.Tensor2D[];
    const [bq, bk, bv] = tf.split(this.inB, 3, 0);
    const Q = heads(tf.add(tf.matMul(q, wq), bq), this.h);
    const K = heads(tf.add(tf.matMul(k, wk), bk), this.h);
    const V = heads(tf.add(tf.matMul(v, wv), bv), this.h);
    const { out, probs } = sdpa(Q, K, V, bias);
    return { out: this.out.apply(merge(out)), weights: tf.mean(probs, 0) as tf.Tensor2D };
  }
}

/** Pre-norm `nn.TransformerEncoderLayer` (GELU, FFN 4D). */
export class EncoderLayer {
  readonly attn: MultiheadAttention;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly lin1: Linear;
  readonly lin2: Linear;

  constructor(p: Params, name: string, d: number, h: number) {
    this.attn = new MultiheadAttention(p, `${name}.attn`, d, h);
    this.lin1 = new Linear(p, `${name}.lin1`, d, 4 * d);
    this.lin2 = new Linear(p, `${name}.lin2`, 4 * d, d);
    this.norm1 = new LayerNorm(p, `${name}.norm1`, d);
    this.norm2 = new LayerNorm(p, `${name}.norm2`, d);
  }

  apply(x: tf.Tensor2D, bias?: tf.Tensor): tf.Tensor2D {
    const n = this.norm1.apply(x);
    x = tf.add(x, this.attn.apply(n, n, n, bias).out);
    return tf.add(x, this.lin2.apply(gelu(this.lin1.apply(this.norm2.apply(x)))));
  }
}

/** `nn.TransformerEncoder` of pre-norm layers plus a final LayerNorm (`attention.transformer_encoder`). */
export class TransformerEncoder {
  readonly layers: EncoderLayer[] = [];
  readonly norm: LayerNorm;

  constructor(p: Params, name: string, d: number, h: number, nLayers: number) {
    for (let i = 0; i < nLayers; i++) this.layers.push(new EncoderLayer(p, `${name}.${i}`, d, h));
    this.norm = new LayerNorm(p, `${name}.norm`, d);
  }

  /** `bias`: additive `[1, S]` key mask or `[S, S]` attention mask. */
  apply(x: tf.Tensor2D, bias?: tf.Tensor): tf.Tensor2D {
    for (const l of this.layers) x = l.apply(x, bias);
    return this.norm.apply(x);
  }
}

/**
 * Learned-query attention pooling `[S, D] -> [D]` with an always-visible learned null key/value, so pooling
 * an all-PAD sequence is well defined (port of `AttentionPool`, one query).
 */
export class AttentionPool {
  readonly query: tf.Variable;
  readonly nullKv: tf.Variable;
  readonly k: Linear;
  readonly v: Linear;
  readonly out: Linear;
  readonly norm: LayerNorm;

  constructor(p: Params, name: string, readonly d: number, readonly h: number) {
    if (d % h) throw new Error(`d_model=${d} not divisible by n_heads=${h}`);
    this.query = p.normal(`${name}.query`, [1, d], 0.02);
    this.nullKv = p.constant(`${name}.null`, [1, d], 0);
    this.k = new Linear(p, `${name}.k`, d, d);
    this.v = new Linear(p, `${name}.v`, d, d);
    this.out = new Linear(p, `${name}.out`, d, d);
    this.norm = new LayerNorm(p, `${name}.norm`, d);
  }

  apply(tokens: tf.Tensor2D, pad: boolean[]): tf.Tensor1D {
    const kv = tf.concat([this.nullKv as tf.Tensor2D, tokens], 0);
    const bias = additiveKeyMask([false, ...pad]);
    const { out } = sdpa(heads(this.query as tf.Tensor2D, this.h), heads(this.k.apply(kv), this.h),
      heads(this.v.apply(kv), this.h), bias);
    return this.norm.apply(this.out.apply(merge(out))).squeeze([0]) as tf.Tensor1D;
  }
}
