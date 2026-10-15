/**
 * In-memory cosine vector index (port of soma/memory/vector_index.py, brute force only).
 *
 * Permission filtering happens before scoring: with `allowedIds`, only those rows are ever compared with
 * the query, so a blocked item can never influence (or leak through) a score. Persistence is a plain
 * `{dim, ids, data}` record (structured-clone friendly for IndexedDB); `MemoryStore` also rebuilds the
 * index from its per-item embeddings.
 */

export interface SerializedIndex {
  dim: number;
  ids: string[];
  /** Row-major `[ids.length * dim]` unit vectors. */
  data: Float32Array;
}

export interface SearchHit {
  id: string;
  score: number;
}

function unit(v: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(v);
  let s = 0;
  for (const x of out) s += x * x;
  const n = Math.sqrt(s);
  if (n > 0) for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

export class VectorIndex {
  readonly dim: number;
  private readonly rows = new Map<string, Float32Array>();

  constructor(dim: number) {
    this.dim = dim;
  }

  get size(): number {
    return this.rows.size;
  }

  has(id: string): boolean {
    return this.rows.has(id);
  }

  ids(): string[] {
    return [...this.rows.keys()];
  }

  /** Stored unit vector (a copy) or null. */
  get(id: string): Float32Array | null {
    const v = this.rows.get(id);
    return v ? v.slice() : null;
  }

  /** Insert or replace `id`'s vector (stored L2-normalized). */
  add(id: string, vec: ArrayLike<number>): void {
    if (vec.length !== this.dim) throw new Error(`expected dim ${this.dim}, got ${vec.length}`);
    this.rows.set(id, unit(vec));
  }

  /** Delete `id`; false when absent. */
  remove(id: string): boolean {
    return this.rows.delete(id);
  }

  clear(): void {
    this.rows.clear();
  }

  /** Replace all contents with `[id, vector]` pairs. */
  rebuild(items: Iterable<[string, ArrayLike<number>]>): void {
    const next = new Map<string, Float32Array>();
    for (const [id, v] of items) {
      if (v.length !== this.dim) throw new Error(`expected dim ${this.dim}, got ${v.length}`);
      next.set(String(id), unit(v));
    }
    this.rows.clear();
    for (const [k, v] of next) this.rows.set(k, v);
  }

  /** Top-`k` hits by cosine, best first (ties keep insertion order). Only `allowedIds` are scored when given. */
  search(query: ArrayLike<number>, k: number, allowedIds?: ReadonlySet<string> | null): SearchHit[] {
    if (k <= 0) return [];
    if (query.length !== this.dim) throw new Error(`expected dim ${this.dim}, got ${query.length}`);
    const q = unit(query);
    const hits: SearchHit[] = [];
    const score = (id: string, v: Float32Array) => {
      let s = 0;
      for (let i = 0; i < v.length; i++) s += v[i] * q[i];
      hits.push({ id, score: s });
    };
    if (allowedIds) {
      for (const [id, v] of this.rows) if (allowedIds.has(id)) score(id, v);
    } else {
      for (const [id, v] of this.rows) score(id, v);
    }
    // Array.prototype.sort is stable
    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, k);
  }

  serialize(): SerializedIndex {
    const ids = this.ids();
    const data = new Float32Array(ids.length * this.dim);
    ids.forEach((id, i) => data.set(this.rows.get(id)!, i * this.dim));
    return { dim: this.dim, ids, data };
  }

  static deserialize(s: SerializedIndex): VectorIndex {
    const idx = new VectorIndex(s.dim);
    idx.rebuild(s.ids.map((id, i) => [id, s.data.subarray(i * s.dim, (i + 1) * s.dim)] as [string, Float32Array]));
    return idx;
  }
}
