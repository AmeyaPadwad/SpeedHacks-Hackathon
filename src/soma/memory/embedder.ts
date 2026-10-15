/**
 * Deterministic text embedder for memory items and retrieval queries (port of soma/memory/embedder.py).
 *
 * An untrained bootstrap encoder that preserves lexical overlap as cosine similarity: hashed word tokens
 * plus character trigrams (robust to inflection), each mapped to a fixed pseudo-random Gaussian vector
 * derived from `(seed, feature id)`, mean-pooled, layer-normalized and L2-normalized. Same text -> same
 * vector in every tab and browser, with no weights to ship.
 *
 * Differences from Python: the near-identity projection is the identity (its noise only perturbed
 * cosines slightly), LayerNorm has no learned affine, and feature vectors are generated on demand (and
 * cached) instead of living in a `vocabSize x dim` table. Images are embedded by their text summary in
 * `ingestion.ts`; there is no image conv net.
 */
import { tokenId, tokenize, UNK_ID } from "../perception/tokenizer";

export const DEFAULT_SEED = 1234;
const CACHE_LIMIT = 8192;
const LN_EPS = 1e-5;

/** Hashed word tokens plus `#<tri>` character trigrams of each alphabetic word longer than 3 letters. */
export function textFeatures(text: string, vocabSize: number, maxFeatures: number): number[] {
  const ids: number[] = [];
  for (const tok of tokenize(text)) {
    ids.push(tokenId(tok, vocabSize));
    if (/^[a-z]+$/.test(tok) && tok.length > 3) {
      const padded = `<${tok}>`;
      for (let i = 0; i < padded.length - 2; i++) ids.push(tokenId(`#${padded.slice(i, i + 3)}`, vocabSize));
    }
    if (ids.length >= maxFeatures) break;
  }
  return ids.slice(0, maxFeatures);
}

/** mulberry32 PRNG seeded from (seed, id). */
function rng(seed: number, id: number): () => number {
  let a = (Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(id + 0x632be5ab, 0xc2b2ae35)) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Truncate or zero-pad `v` to `dim`, then L2-normalize (a zero vector stays zero). */
export function fitDim(v: ArrayLike<number>, dim: number): Float32Array {
  const out = new Float32Array(dim);
  const n = Math.min(dim, v.length);
  for (let i = 0; i < n; i++) out[i] = v[i];
  return l2Normalize(out);
}

/** In-place L2 normalization; returns its argument. */
export function l2Normalize(v: Float32Array): Float32Array {
  let s = 0;
  for (const x of v) s += x * x;
  const norm = Math.sqrt(s);
  if (norm > 0) for (let i = 0; i < v.length; i++) v[i] /= norm;
  return v;
}

export interface TextEmbedderOptions {
  vocabSize?: number;
  seed?: number;
  maxFeatures?: number;
}

export class TextEmbedder {
  readonly dim: number;
  readonly vocabSize: number;
  readonly seed: number;
  readonly maxFeatures: number;
  private readonly cache = new Map<number, Float32Array>();

  constructor(dim: number, opts: TextEmbedderOptions = {}) {
    if (!Number.isInteger(dim) || dim <= 0) throw new Error(`bad embedding dim ${dim}`);
    this.dim = dim;
    this.vocabSize = opts.vocabSize ?? 16384;
    this.seed = opts.seed ?? DEFAULT_SEED;
    this.maxFeatures = opts.maxFeatures ?? 4096;
  }

  /** The fixed N(0, 1) vector of one feature id. */
  private featureVector(id: number): Float32Array {
    let v = this.cache.get(id);
    if (v) return v;
    const r = rng(this.seed, id);
    v = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i += 2) {
      const u1 = Math.max(r(), 1e-12);
      const u2 = r();
      const mag = Math.sqrt(-2 * Math.log(u1));
      v[i] = mag * Math.cos(2 * Math.PI * u2);
      if (i + 1 < this.dim) v[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }
    if (this.cache.size >= CACHE_LIMIT) this.cache.clear();
    this.cache.set(id, v);
    return v;
  }

  /** One string -> float32 `[dim]` unit vector (empty text -> the UNK vector). */
  embed(text: string): Float32Array {
    const feats = textFeatures(text ?? "", this.vocabSize, this.maxFeatures);
    if (!feats.length) feats.push(UNK_ID);
    const out = new Float32Array(this.dim);
    for (const id of feats) {
      const fv = this.featureVector(id);
      for (let i = 0; i < this.dim; i++) out[i] += fv[i];
    }
    // mean pool, then LayerNorm (no affine), then L2
    let mean = 0;
    for (let i = 0; i < this.dim; i++) mean += (out[i] /= feats.length);
    mean /= this.dim;
    let variance = 0;
    for (let i = 0; i < this.dim; i++) variance += (out[i] - mean) ** 2;
    const sd = Math.sqrt(variance / this.dim + LN_EPS);
    for (let i = 0; i < this.dim; i++) out[i] = (out[i] - mean) / sd;
    return l2Normalize(out);
  }

  embedMany(texts: string[]): Float32Array[] {
    return texts.map((t) => this.embed(t));
  }
}
