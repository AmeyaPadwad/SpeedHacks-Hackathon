/**
 * Permission-filtered retrieval and reranking of local memories (port of soma/memory/retrieval.py).
 *
 * 1. Permission prefilter over metadata only: user-approved, not expired, documents/images still inside
 *    a folder the user has not removed, feedback memories only when allowed, superseded conflicting
 *    preferences dropped.
 * 2. Vector search over the allowed ids only (no other row is ever scored).
 * 3. Rerank = relevance + recency (half-life) + source trust + preference boost.
 * 4. A final ACL re-check; failures are counted in `permissionViolations` (expected to stay 0).
 *
 * Retrieved summaries are untrusted task data, never instructions.
 */
import type { MemoryConfig } from "../config";
import { SourceType, type MemoryItem, type RetrievedMemory } from "../types";
import type { TextEmbedder } from "./embedder";
import { folderOf, type MemoryStore } from "./store";
import type { VectorIndex } from "./vectorIndex";

export const DEFAULT_TRUST = 0.5;
export const CONFLICT_PREFIX = "conflict:";
const FILE_TYPES: ReadonlySet<SourceType> = new Set([SourceType.DOCUMENT, SourceType.IMAGE]);

export interface RerankWeights {
  relevance: number;
  recency: number;
  trust: number;
  preference: number;
}

export const DEFAULT_WEIGHTS: RerankWeights = { relevance: 1.0, recency: 0.15, trust: 0.1, preference: 0.1 };

/** The `conflict:<key>` tag value of an item, if any. */
export function conflictKey(item: MemoryItem): string | null {
  for (const t of item.tags) if (t.startsWith(CONFLICT_PREFIX) && t.length > CONFLICT_PREFIX.length) return t.slice(CONFLICT_PREFIX.length);
  return null;
}

const parseIso = (s: string | null | undefined): number | null => {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isNaN(t) ? null : t;
};

export interface RetrieverOptions {
  weights?: RerankWeights;
  oversample?: number;
  /** Whether FEEDBACK memories may be used (the Python FEEDBACK_STORAGE consent). Default true. */
  allowFeedback?: () => boolean;
  /** Per-folder trust in [0, 1] (default `DEFAULT_TRUST`). */
  folderTrust?: (folderId: string) => number;
}

export class MemoryRetriever {
  permissionViolations = 0;
  lastAllowedCount = 0;
  readonly weights: RerankWeights;
  private readonly oversample: number;
  private readonly allowFeedback: () => boolean;
  private readonly folderTrust: (folderId: string) => number;

  constructor(
    private readonly store: MemoryStore,
    private readonly index: VectorIndex,
    private readonly embedder: TextEmbedder,
    private readonly cfg: MemoryConfig,
    opts: RetrieverOptions = {},
  ) {
    this.weights = opts.weights ?? DEFAULT_WEIGHTS;
    this.oversample = Math.max(1, opts.oversample ?? 4);
    this.allowFeedback = opts.allowFeedback ?? (() => true);
    this.folderTrust = opts.folderTrust ?? (() => DEFAULT_TRUST);
  }

  /** Metadata-only ACL check for one item against the current folder ids. */
  isPermitted(item: MemoryItem, folderIds: ReadonlySet<string>, now: number = Date.now()): boolean {
    if (!item.userApproved) return false;
    const exp = parseIso(item.expiresAt);
    if (exp !== null && exp <= now) return false;
    const folder = folderOf(item.sourceUri);
    // anything that came from a folder the user removed is gone; files must come from a current folder
    if (folder !== null && !folderIds.has(folder)) return false;
    if (FILE_TYPES.has(item.sourceType) && folder === null) return false;
    if (item.sourceType === SourceType.FEEDBACK && !this.allowFeedback()) return false;
    return true;
  }

  /** All items passing the ACL, with superseded conflicting memories removed. */
  async allowedItems(now: number = Date.now()): Promise<Map<string, MemoryItem>> {
    const folderIds = new Set((await this.store.listFolders()).map((f) => f.id));
    const items = (await this.store.allItems()).filter((i) => this.isPermitted(i, folderIds, now));
    const stamp = (i: MemoryItem) => `${i.updatedAt ?? ""}\u0000${i.createdAt ?? ""}`;
    const winners = new Map<string, MemoryItem>();
    for (const it of items) {
      const key = conflictKey(it);
      if (key === null) continue;
      const cur = winners.get(key);
      if (!cur || stamp(it) > stamp(cur)) winners.set(key, it);
    }
    const out = new Map<string, MemoryItem>();
    for (const it of items) {
      const key = conflictKey(it);
      if (key !== null && winners.get(key)!.id !== it.id) continue; // an older, conflicting preference
      out.set(it.id, it);
    }
    return out;
  }

  private recency(item: MemoryItem, now: number): number {
    const t = parseIso(item.updatedAt) ?? parseIso(item.createdAt);
    if (t === null) return 0;
    const ageDays = Math.max(0, (now - t) / 86_400_000);
    return 0.5 ** (ageDays / Math.max(1e-6, this.cfg.recencyHalfLifeDays));
  }

  private trust(item: MemoryItem): number {
    if (item.sourceType === SourceType.PREFERENCE || item.sourceType === SourceType.FEEDBACK) return 1;
    const folder = folderOf(item.sourceUri);
    return folder ? this.folderTrust(folder) : DEFAULT_TRUST;
  }

  score(item: MemoryItem, relevance: number, now: number): number {
    const w = this.weights;
    const pref = item.sourceType === SourceType.PREFERENCE ? 1 : 0;
    return w.relevance * relevance + w.recency * this.recency(item, now) + w.trust * this.trust(item) + w.preference * pref;
  }

  /** Top-`k` permitted memories for `queryText` (embedded with the shared embedder), best first. */
  async retrieve(queryText: string, k: number = this.cfg.topK, now: number = Date.now()): Promise<RetrievedMemory[]> {
    if (k <= 0) return [];
    const allowed = await this.allowedItems(now);
    this.lastAllowedCount = allowed.size;
    if (!allowed.size) return [];
    const q = this.embedder.embed(queryText);
    const hits = this.index.search(q, k * this.oversample, new Set(allowed.keys()));
    const folderIds = new Set((await this.store.listFolders()).map((f) => f.id));
    const results: RetrievedMemory[] = [];
    for (const { id, score } of hits) {
      const item = allowed.get(id);
      const vec = this.index.get(id);
      if (!item || !vec || !this.isPermitted(item, folderIds, now)) {
        this.permissionViolations += 1;
        continue;
      }
      results.push({ item, embedding: vec, score: this.score(item, score, now) });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, k);
  }
}

/**
 * Pack memories into `(embeddings [K*D], mask [K])` (1 = padding). Extra memories beyond `K` are dropped;
 * missing slots are zero vectors.
 */
export function memoriesToArrays(retrieved: readonly RetrievedMemory[], K: number, D: number): { embeddings: Float32Array; mask: Uint8Array } {
  const embeddings = new Float32Array(K * D);
  const mask = new Uint8Array(K).fill(1);
  retrieved.slice(0, K).forEach((r, i) => {
    if (r.embedding.length !== D) throw new Error(`memory embedding has dim ${r.embedding.length}, expected ${D}`);
    embeddings.set(r.embedding, i * D);
    mask[i] = 0;
  });
  return { embeddings, mask };
}
