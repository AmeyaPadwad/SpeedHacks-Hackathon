/**
 * Bounded, stratified replay buffer for continual updates (port of `replay_buffer.py`).
 *
 *  - Per-(app, task) reservoir sampling keeps every stratum an unbiased sample of what it has seen, so one
 *    busy app cannot crowd out the others.
 *  - A global `capacity` bound evicts from the largest stratum first.
 *  - "Anchor" examples (curated, never raw user content) live in their own bounded reservoir and are mixed
 *    into every sample to limit forgetting.
 *  - Ambiguous (ignored) and non-consented examples are refused at `add`.
 *
 * In memory only; never persisted. Deterministic for a given seed.
 */
import { Rng } from "../models/math";

/** What the buffer needs from an example. `LabeledExample` satisfies it. */
export interface Replayable {
  app: string;
  task: string;
  usableForTraining: boolean;
}

const SEP = "\x1f";
const keyOf = (e: Replayable): string => `${e.app}${SEP}${e.task}`;

export interface ReplayOptions {
  perStratumCapacity?: number;
  anchorCapacity?: number;
  anchorFraction?: number;
  seed?: number;
}

export class ReplayBuffer<T extends Replayable> {
  readonly capacity: number;
  readonly perStratumCapacity: number;
  readonly anchorCapacity: number;
  readonly anchorFraction: number;
  rejected = 0;
  private rng: Rng;
  private strataMap = new Map<string, T[]>();
  private seen = new Map<string, number>();
  private anchorPool: T[] = [];
  private anchorSeen = 0;

  constructor(capacity = 2000, opts: ReplayOptions = {}) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.capacity = Math.floor(capacity);
    this.perStratumCapacity = Math.floor(opts.perStratumCapacity ?? capacity);
    this.anchorCapacity = Math.floor(opts.anchorCapacity ?? 500);
    this.anchorFraction = opts.anchorFraction ?? 0.25;
    this.rng = new Rng(opts.seed ?? 0);
  }

  /** True only for consented, unambiguous examples. */
  static accepts(e: Replayable): boolean {
    return e.usableForTraining;
  }

  private reservoir(pool: T[], seen: number, cap: number, e: T): void {
    if (pool.length < cap) {
      pool.push(e);
      return;
    }
    const j = this.rng.int(seen);
    if (j < cap) pool[j] = e;
  }

  /** Add one example; false when refused (ambiguous / no consent). */
  add(e: T): boolean {
    if (!ReplayBuffer.accepts(e)) {
      this.rejected += 1;
      return false;
    }
    const k = keyOf(e);
    const pool = this.strataMap.get(k) ?? [];
    this.strataMap.set(k, pool);
    this.seen.set(k, (this.seen.get(k) ?? 0) + 1);
    this.reservoir(pool, this.seen.get(k)!, this.perStratumCapacity, e);
    while (this.size > this.capacity) {
      let biggest = "";
      let n = -1;
      for (const [key, v] of this.strataMap) {
        if (v.length > n || (v.length === n && key > biggest)) {
          biggest = key;
          n = v.length;
        }
      }
      const victims = this.strataMap.get(biggest)!;
      victims.splice(this.rng.int(victims.length), 1);
    }
    return true;
  }

  /** Add many; returns how many were accepted. */
  extend(examples: T[]): number {
    return examples.reduce((n, e) => n + (this.add(e) ? 1 : 0), 0);
  }

  /** Add a prior-competency anchor (must be unambiguous and consented/curated). */
  addAnchor(e: T): boolean {
    if (!ReplayBuffer.accepts(e)) {
      this.rejected += 1;
      return false;
    }
    this.anchorSeen += 1;
    this.reservoir(this.anchorPool, this.anchorSeen, this.anchorCapacity, e);
    return true;
  }

  get size(): number {
    let n = 0;
    for (const v of this.strataMap.values()) n += v.length;
    return n;
  }

  get nAnchors(): number {
    return this.anchorPool.length;
  }

  private sortedKeys(): string[] {
    return [...this.strataMap.keys()].sort();
  }

  /** Stratum sizes keyed by `[app, task]`, sorted. */
  strata(): Array<{ app: string; task: string; count: number }> {
    return this.sortedKeys().map((k) => {
      const [app, task] = k.split(SEP);
      return { app, task, count: this.strataMap.get(k)!.length };
    });
  }

  examples(): T[] {
    return this.sortedKeys().flatMap((k) => this.strataMap.get(k)!);
  }

  anchors(): T[] {
    return [...this.anchorPool];
  }

  /** `count` picks without replacement, cycling fresh permutations if needed. */
  private take(pool: T[], count: number, rng: Rng): T[] {
    const out: T[] = [];
    while (out.length < count && pool.length) {
      const perm = rng.shuffle(pool.map((_, i) => i));
      for (const i of perm.slice(0, count - out.length)) out.push(pool[i]);
    }
    return out;
  }

  /**
   * Stratified sample of `n` with anchors mixed in. Strata get equal shares (remainder to random strata).
   * Deterministic for a given buffer state and `rng` seed.
   */
  sample(n: number, rng: Rng = new Rng(0), anchorFraction?: number): T[] {
    const frac = anchorFraction ?? this.anchorFraction;
    const keys = this.sortedKeys().filter((k) => this.strataMap.get(k)!.length);
    let nAnchor = this.anchorPool.length ? Math.round(n * frac) : 0;
    if (!keys.length) nAnchor = this.anchorPool.length ? n : 0;
    const nMain = n - nAnchor;
    const out: T[] = [];
    if (keys.length && nMain > 0) {
      const share = Math.floor(nMain / keys.length);
      const extra = new Set(rng.shuffle(keys.map((_, i) => i)).slice(0, nMain % keys.length));
      keys.forEach((k, i) => out.push(...this.take(this.strataMap.get(k)!, share + (extra.has(i) ? 1 : 0), rng)));
    }
    out.push(...this.take(this.anchorPool, nAnchor, rng));
    return rng.shuffle(out);
  }

  /** Forget everything except anchors (e.g. when training consent is revoked). */
  clear(): void {
    this.strataMap.clear();
    this.seen.clear();
  }
}
