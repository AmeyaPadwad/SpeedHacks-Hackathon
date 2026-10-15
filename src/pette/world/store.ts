/**
 * Pette's fact log (port of db/store.py) over IndexedDB.
 *
 * Three small tables, facts only, 30-day retention:
 *   events          webcam + UI signals: face_away, away_ended (with duration), eyes_closed, pinch, ...
 *   offers          which kind of SOMA suggestion was shown and what you did: shown | accepted | dismissed | ignored
 *   blink_baseline  30-second summaries (mean, p95, sample count) of open-eye blink scores
 * Never suggestion text, file names or paths, screen content or camera frames: the row types below have no
 * field that could hold them, and `test` checks that every stored record keeps to these keys.
 *
 * `FactLog` is synchronous (the hub and the webcam call it from event handlers), so rows are kept in memory
 * as well and written through to IndexedDB in the background; `flush()` waits for those writes. Opened with
 * `FactStore.open()`; `new FactStore()` is an in-memory store (tests, or when IndexedDB is unavailable).
 */
import { openDB, type IDBPDatabase } from "idb";
import type { FactLog, Outcome } from "../contracts";

export const DB_NAME = "pette-facts";
export const RETENTION_DAYS = 30;
const DAY_MS = 86_400_000;

/** What the offers table stores. "shown" is logged when a card appears; "rejected" is stored as "dismissed". */
export type OfferOutcome = "shown" | "accepted" | "dismissed" | "ignored";

export interface EventRow { ts: number; kind: string; duration: number | null }
export interface OfferRow { ts: number; kind: string; outcome: OfferOutcome }
export interface BlinkRow { ts: number; mean: number; p95: number; samples: number }

/** Every key a stored record may have, per table (checked by the facts-only test). */
export const ROW_KEYS = {
  events: ["ts", "kind", "duration"],
  offers: ["ts", "kind", "outcome"],
  blink_baseline: ["ts", "mean", "p95", "samples"],
} as const;
export type Table = keyof typeof ROW_KEYS;
export const TABLES = Object.keys(ROW_KEYS) as Table[];

/** The fact log the hub uses: `FactLog` that also takes "shown" (and the stored name "dismissed"). */
export interface OfferLog extends FactLog {
  logOffer(kind: string, outcome: Outcome | OfferOutcome): void;
}

type Rows = { events: EventRow[]; offers: OfferRow[]; blink_baseline: BlinkRow[] };

export class FactStore implements OfferLog {
  /** "Remember my patterns": off = nothing is written. */
  enabled = true;
  private rows: Rows = { events: [], offers: [], blink_baseline: [] };
  private pending: Promise<unknown> = Promise.resolve();
  private listeners = new Set<() => void>();

  constructor(private readonly db: IDBPDatabase | null = null, private readonly clock: () => number = Date.now) {}

  /** Open (or create) the IndexedDB log, load the last 30 days and prune older rows. */
  static async open(name = DB_NAME, clock: () => number = Date.now): Promise<FactStore> {
    const db = await openDB(name, 1, {
      upgrade(d) {
        for (const t of TABLES) {
          if (!d.objectStoreNames.contains(t)) d.createObjectStore(t, { autoIncrement: true });
        }
      },
    });
    const store = new FactStore(db, clock);
    await store.load();
    await store.prune();
    return store;
  }

  /** IndexedDB if it opens, else an in-memory log (private windows, blocked storage). */
  static async openOrMemory(name = DB_NAME): Promise<FactStore> {
    try {
      return await FactStore.open(name);
    } catch {
      return new FactStore();
    }
  }

  private async load() {
    if (!this.db) return;
    for (const t of TABLES) {
      (this.rows[t] as unknown[]) = (await this.db.getAll(t)).sort((a, b) => a.ts - b.ts);
    }
  }

  // --- writes ---------------------------------------------------------------
  private insert<T extends Table>(table: T, row: Rows[T][number]) {
    if (!this.enabled) return;
    (this.rows[table] as Rows[T][number][]).push(row);
    if (this.db) {
      const db = this.db;
      this.pending = this.pending.then(() => db.add(table, row)).catch(() => undefined);
    }
    this.emit();
  }

  logEvent(kind: string, duration: number | null = null): void {
    this.insert("events", { ts: this.clock(), kind, duration: duration ?? null });
  }

  /** kind: "soma:<template or action>"; outcome: shown | accepted | rejected (stored as dismissed) | ignored. */
  logOffer(kind: string, outcome: Outcome | OfferOutcome): void {
    this.insert("offers", { ts: this.clock(), kind, outcome: outcome === "rejected" ? "dismissed" : outcome });
  }

  logBlinkBaseline(mean: number, p95: number, samples: number): void {
    this.insert("blink_baseline", { ts: this.clock(), mean, p95, samples });
  }

  // --- reads ----------------------------------------------------------------
  events(): readonly EventRow[] { return this.rows.events; }
  offers(): readonly OfferRow[] { return this.rows.offers; }
  blinkBaselines(): readonly BlinkRow[] { return this.rows.blink_baseline; }

  counts(): Record<Table, number> {
    return { events: this.rows.events.length, offers: this.rows.offers.length, blink_baseline: this.rows.blink_baseline.length };
  }

  /** Every record as stored in IndexedDB (tests; the facts-only check). */
  async stored(): Promise<Record<Table, Record<string, unknown>[]>> {
    await this.flush();
    const out = { events: [], offers: [], blink_baseline: [] } as Record<Table, Record<string, unknown>[]>;
    for (const t of TABLES) out[t] = this.db ? await this.db.getAll(t) : [...this.rows[t]] as never;
    return out;
  }

  /** Called on every change (world panel refresh). Returns an unsubscribe function. */
  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    for (const fn of this.listeners) fn();
  }

  // --- housekeeping ---------------------------------------------------------
  /** Wait for the background writes. */
  async flush(): Promise<void> {
    await this.pending;
  }

  async prune(days = RETENTION_DAYS): Promise<void> {
    const cutoff = this.clock() - days * DAY_MS;
    for (const t of TABLES) (this.rows[t] as { ts: number }[]) = this.rows[t].filter((r) => r.ts >= cutoff);
    if (this.db) {
      await this.flush();
      for (const t of TABLES) {
        const tx = this.db.transaction(t, "readwrite");
        let cursor = await tx.store.openCursor();
        while (cursor) {
          if ((cursor.value as { ts: number }).ts < cutoff) await cursor.delete();
          cursor = await cursor.continue();
        }
        await tx.done;
      }
    }
    this.emit();
  }

  /** "Clear Pette memory": delete everything (also un-mutes every kind). */
  async clear(): Promise<void> {
    this.rows = { events: [], offers: [], blink_baseline: [] };
    if (this.db) {
      await this.flush();
      for (const t of TABLES) await this.db.clear(t);
    }
    this.emit();
  }

  close(): void {
    this.db?.close();
  }
}
