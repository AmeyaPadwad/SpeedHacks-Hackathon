/**
 * Small numeric helpers shared by the model and the learning code: a seeded RNG (so a fresh page load builds
 * the same untrained model), softmax on plain arrays, and fixed-size vector fitting.
 */

/** Large negative logit used instead of `-Infinity` (port of `attention.NEG_INF`); `exp(-1e4)` is exactly 0. */
export const NEG_INF = -1e4;

/** Deterministic PRNG (mulberry32) with uniform and normal draws. Same seed, same sequence. */
export class Rng {
  private state: number;
  private spare: number | null = null;

  constructor(seed = 0) {
    this.state = (Math.floor(seed) ^ 0x9e3779b9) >>> 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Standard normal (Box-Muller). */
  normal(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u = 0;
    while (u <= 1e-12) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  /** In-place Fisher-Yates shuffle; returns `arr`. */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /** Raw generator state, for persistence. */
  getState(): number {
    return this.state;
  }

  setState(state: number): void {
    this.state = state >>> 0;
    this.spare = null;
  }
}

/** Numerically stable softmax. `NEG_INF` / `-Infinity` entries get ~0; an empty input gives an empty output. */
export function softmax(logits: Float32Array): Float32Array {
  const out = new Float32Array(logits.length);
  if (!logits.length) return out;
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  if (!Number.isFinite(max)) return out;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Zero-pad or truncate `v` to length `dim` (port of the pipeline's `fit_dim`). Always returns a copy. */
export function fitDim(v: Float32Array, dim: number): Float32Array {
  const out = new Float32Array(dim);
  out.set(v.length > dim ? v.subarray(0, dim) : v);
  return out;
}

/** L2-normalised copy of `v` (zero vector stays zero). */
export function l2normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n);
  const out = new Float32Array(v.length);
  if (n > 0) for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}
