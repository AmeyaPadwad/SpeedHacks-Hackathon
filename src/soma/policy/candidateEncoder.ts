/**
 * Encode templated candidates into fixed-size `[A][D]` vectors for the scorer (port of
 * SOMA/soma/policy/candidate_encoder.py).
 *
 * Each candidate embedding is the LayerNorm'd sum of:
 * - an action-type embedding (index into `ACTION_TYPES`, so 0 is NO_SUGGESTION and never a candidate),
 * - the candidate's text (+ its target's text) embedded by the memory `TextEmbedder` (the Python version
 *   pooled a learned hashed-token table; the web build reuses the shared embedder so nothing big ships),
 * - a projection of the target box `[x1, y1, x2, y2, hasTarget]`,
 * - a projection of the referenced memory embedding (`[D + 1]`, last slot = "has memory").
 *
 * Weights are fixed and seeded (an untrained but real encoder, as in the Python runtime); the online ranker
 * is what learns from feedback. Padding slots and blocked candidates are marked 1 in the mask so the model
 * excludes them. The NO_SUGGESTION logit is the model's job; it is never a candidate here.
 */
import { ACTION_TYPES, type ActionType, type Candidate, type RetrievedMemory } from "../types";

/** Anything that embeds text into a fixed-size vector (`soma/memory/embedder.ts` `TextEmbedder`). */
export interface TextEmbedderLike {
  embed(text: string): Float32Array;
}

export type MemoryLike = RetrievedMemory | Float32Array | ArrayLike<number>;

export interface EncodedCandidates {
  /** `A` vectors of `D` floats (zeros in pad slots). */
  embeddings: Float32Array[];
  /** `[A]` action-type indices (pads use 0). */
  types: number[];
  /** `[A]`, 1 = pad or blocked. */
  mask: Uint8Array;
}

const TYPE_INDEX = new Map<ActionType, number>(ACTION_TYPES.map((a, i) => [a, i]));

/** Index of `candidate.action` in `ACTION_TYPES`. */
export function actionIndex(candidate: Candidate): number {
  const i = TYPE_INDEX.get(candidate.action);
  if (i === undefined) throw new RangeError(`unknown action ${candidate.action}`);
  return i;
}

/** mulberry32: small, fast, seedable PRNG in [0, 1). */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Truncate / zero-pad to `dim` (no re-normalisation). */
function fitTo(v: ArrayLike<number>, dim: number): Float32Array {
  const out = new Float32Array(dim);
  const n = Math.min(dim, v.length);
  for (let i = 0; i < n; i++) out[i] = Number.isFinite(v[i]) ? v[i] : 0;
  return out;
}

/** Candidates -> `{ embeddings [A][D], types [A], mask [A] }`. */
export class CandidateEncoder {
  private readonly typeEmbedding: Float32Array[];
  /** `[5][D]` + bias `[D]`. */
  private readonly boxW: Float32Array[];
  private readonly boxB: Float32Array;
  /** `[D + 1][D]` + bias `[D]`. */
  private readonly memW: Float32Array[];
  private readonly memB: Float32Array;
  /** Scales the unit-length text embedding to the magnitude of the other terms. */
  private readonly textScale: number;

  constructor(readonly dModel: number, readonly embedder: TextEmbedderLike,
    readonly nActionTypes = ACTION_TYPES.length, seed = 0) {
    const rand = seededRandom(seed ^ 0x5eed);
    const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    const uniform = (bound: number) => (rand() * 2 - 1) * bound;
    const D = dModel;
    this.typeEmbedding = Array.from({ length: nActionTypes }, () => Float32Array.from({ length: D }, normal));
    const b5 = 1 / Math.sqrt(5);
    this.boxW = Array.from({ length: 5 }, () => Float32Array.from({ length: D }, () => uniform(b5)));
    this.boxB = Float32Array.from({ length: D }, () => uniform(b5));
    const bm = 1 / Math.sqrt(D + 1);
    this.memW = Array.from({ length: D + 1 }, () => Float32Array.from({ length: D }, () => uniform(bm)));
    this.memB = Float32Array.from({ length: D }, () => uniform(bm));
    this.textScale = Math.sqrt(D);
  }

  private textOf(c: Candidate): string {
    return c.target ? `${c.text} ${c.target.text}` : c.text;
  }

  private memoryVector(mem: MemoryLike | undefined): Float32Array {
    const D = this.dModel;
    const vec = new Float32Array(D + 1);
    if (mem === undefined) return vec;
    const raw: ArrayLike<number> = "item" in (mem as object) ? (mem as RetrievedMemory).embedding : (mem as ArrayLike<number>);
    vec.set(fitTo(raw, D));
    vec[D] = 1;
    return vec;
  }

  /** Embed one real candidate -> `[D]`. */
  embedOne(c: Candidate, mem?: MemoryLike): Float32Array {
    const D = this.dModel;
    const h = Float32Array.from(this.typeEmbedding[actionIndex(c)]);
    const text = fitTo(this.embedder.embed(this.textOf(c)), D);
    for (let d = 0; d < D; d++) h[d] += text[d] * this.textScale;
    const box = c.target
      ? [...c.target.bbox.map((v) => Math.min(Math.max(v, 0), 1)), 1]
      : [0, 0, 0, 0, 0];
    for (let d = 0; d < D; d++) h[d] += this.boxB[d];
    box.forEach((v, k) => {
      if (v) for (let d = 0; d < D; d++) h[d] += v * this.boxW[k][d];
    });
    const mv = this.memoryVector(mem);
    for (let d = 0; d < D; d++) h[d] += this.memB[d];
    mv.forEach((v, k) => {
      if (v) for (let d = 0; d < D; d++) h[d] += v * this.memW[k][d];
    });
    // LayerNorm (unit weight, zero bias, eps 1e-5)
    let mean = 0;
    for (let d = 0; d < D; d++) mean += h[d];
    mean /= D;
    let varSum = 0;
    for (let d = 0; d < D; d++) varSum += (h[d] - mean) ** 2;
    const inv = 1 / Math.sqrt(varSum / D + 1e-5);
    for (let d = 0; d < D; d++) h[d] = (h[d] - mean) * inv;
    return h;
  }

  /**
   * Encode up to `A` candidates (`A` defaults to `candidates.length`; extras are dropped). Pads are zero
   * vectors with type 0 and mask 1; `blocked[i]` also sets mask 1.
   */
  encode(candidates: readonly Candidate[], memoriesById: ReadonlyMap<string, MemoryLike> | null = null,
    A: number = candidates.length, D: number = this.dModel, blocked: readonly boolean[] | null = null): EncodedCandidates {
    if (D !== this.dModel) throw new RangeError(`D=${D} does not match encoder dModel=${this.dModel}`);
    const embeddings = Array.from({ length: A }, () => new Float32Array(D));
    const types = new Array<number>(A).fill(0);
    const mask = new Uint8Array(A).fill(1);
    candidates.slice(0, A).forEach((c, i) => {
      const mem = c.memoryRef && memoriesById ? memoriesById.get(c.memoryRef) : undefined;
      embeddings[i] = this.embedOne(c, mem);
      types[i] = actionIndex(c);
      mask[i] = blocked && blocked[i] ? 1 : 0;
    });
    return { embeddings, types, mask };
  }
}
