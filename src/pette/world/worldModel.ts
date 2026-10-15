/**
 * World model (port of db/world_model.py): what Pette has learned about this user from the fact log, and
 * how it changes her behaviour.
 *
 * Each method is a small, explainable rule over the last 30 days of facts. Defaults apply until there's
 * enough evidence, so a fresh install behaves exactly like the spec.
 */
import type { BlinkRow, EventRow, OfferRow } from "./store";

export const DEFAULT_BLINK_ON = 0.6;
export const MIN_BASELINE_SAMPLES = 300;
/** Decisions on one kind of suggestion before Pette may stop showing it ... */
export const MUTE_MIN_DECIDED = 4;
/** ... and the share of them that were "Not now". */
export const MUTE_DISMISS_RATE = 0.8;
export const SOMA_PREFIX = "soma:";
const DAY_MS = 86_400_000;

export const KIND_LABELS: Record<string, string> = {
  review_changes: "Review changes", write_tests: "Write a test", commit_work: "Commit work",
  take_notes: "Save notes", explain_error: "Explain error", open_relevant_file: "Open file",
  compare_with_prior_work: "Your files", suggest_next_step: "Next step",
  ask_clarifying_question: "Question",
};

/** "soma:write_tests" -> "Write a test"; unknown kinds -> "Some kind". */
export function kindLabel(kind: string): string {
  const k = kind.startsWith(SOMA_PREFIX) ? kind.slice(SOMA_PREFIX.length) : kind;
  if (KIND_LABELS[k]) return KIND_LABELS[k];
  const s = k.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

/** The reads the world model needs (implemented by `FactStore`). */
export interface FactReader {
  events(): readonly EventRow[];
  offers(): readonly OfferRow[];
  blinkBaselines(): readonly BlinkRow[];
}

export interface SomaCounts { shown: number; accepted: number; dismissed: number; ignored: number }
export interface KindStats extends SomaCounts { kind: string; label: string }
export interface Rule { when: string; then: string; why: string }
export interface Presence {
  awayEpisodes: number;
  medianAwayS: number | null;
  eyesClosed: number;
  pinches: number;
  /** Point left / right: paging through recent suggestions. */
  steps: number;
  /** Pinch-and-drag scrolls. */
  scrolls: number;
}

/** Everything the world panel shows. Facts only: kinds, counts, outcomes, event names and times. */
export interface WorldSnapshot {
  soma: SomaCounts;
  acceptRate: number | null;
  kinds: KindStats[];
  /** Labels of muted kinds. */
  muted: string[];
  blinkThreshold: number;
  blinkCalibrated: boolean;
  baselineSamples: number;
  presence: Presence;
  rules: Rule[];
  recent: { ts: number; kind: string; outcome: string }[];
  events: { ts: number; kind: string }[];
  updated: number;
}

const OUTCOMES = ["shown", "accepted", "dismissed", "ignored"] as const;
const isSoma = (r: OfferRow) => r.kind.startsWith(SOMA_PREFIX);
const newest = <T extends { ts: number }>(rows: readonly T[], n: number) => [...rows].sort((a, b) => b.ts - a.ts).slice(0, n);

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const round = (x: number, digits: number) => Math.round(x * 10 ** digits) / 10 ** digits;

export class WorldModel {
  constructor(private readonly store: FactReader, private readonly clock: () => number = Date.now) {}

  // --- webcam calibration ---------------------------------------------------
  /**
   * Eyes-closed threshold tuned to this user's open-eye baseline (glasses, lighting, eye shape):
   * 0.15 above the open-eye 95th percentile, kept within 0.5..0.85.
   */
  blinkThreshold(): number {
    const rows = newest(this.store.blinkBaselines(), 20);
    const total = rows.reduce((a, r) => a + r.samples, 0);
    if (total < MIN_BASELINE_SAMPLES) return DEFAULT_BLINK_ON;
    const p95 = rows.reduce((a, r) => a + r.p95 * r.samples, 0) / total;
    return round(Math.min(0.85, Math.max(0.5, p95 + 0.15)), 2);
  }

  // --- SOMA suggestions -----------------------------------------------------
  /** Suggestion outcomes (the hub logs `offers` rows with kind "soma:<kind>"). */
  somaCounts(): SomaCounts {
    const c: SomaCounts = { shown: 0, accepted: 0, dismissed: 0, ignored: 0 };
    for (const r of this.store.offers()) if (isSoma(r) && r.outcome in c) c[r.outcome]++;
    return c;
  }

  acceptRate(): number | null {
    const c = this.somaCounts();
    const decided = c.accepted + c.dismissed;
    return decided ? c.accepted / decided : null;
  }

  /** Per kind of suggestion: how often it was shown, done or put off (most shown first). */
  kinds(): KindStats[] {
    const out = new Map<string, KindStats>();
    for (const r of this.store.offers()) {
      if (!isSoma(r)) continue;
      let k = out.get(r.kind);
      if (!k) {
        k = { kind: r.kind, label: kindLabel(r.kind), shown: 0, accepted: 0, dismissed: 0, ignored: 0 };
        out.set(r.kind, k);
      }
      if ((OUTCOMES as readonly string[]).includes(r.outcome)) k[r.outcome]++;
    }
    return [...out.values()].sort((a, b) => b.shown - a.shown || (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : 0));
  }

  /** Kinds you nearly always answer "Not now": Pette stops showing them (Clear memory resets this). */
  mutedKinds(): Set<string> {
    const muted = new Set<string>();
    for (const k of this.kinds()) {
      const decided = k.accepted + k.dismissed;
      if (decided >= MUTE_MIN_DECIDED && k.dismissed / decided >= MUTE_DISMISS_RATE) muted.add(k.kind);
    }
    return muted;
  }

  // --- presence -------------------------------------------------------------
  presenceToday(): Presence {
    const start = this.clock() - DAY_MS;
    const today = this.store.events().filter((e) => e.ts >= start);
    const away = today.filter((e) => e.kind === "away_ended" && e.duration != null).map((e) => e.duration as number);
    const count = (...kinds: string[]) => today.filter((e) => kinds.includes(e.kind)).length;
    return {
      awayEpisodes: away.length,
      medianAwayS: away.length ? round(median(away), 1) : null,
      eyesClosed: count("eyes_closed"),
      pinches: count("pinch"),
      steps: count("point_left", "point_right"),
      scrolls: count("drag_start"),
    };
  }

  /** Plain-text summary (fallback view). */
  summary(): string {
    const c = this.somaCounts();
    const lines = [`SOMA suggestions: ${c.shown} shown, ${c.accepted} done, ${c.dismissed} not now`];
    const muted = this.mutedKinds();
    if (muted.size) lines.push("Not shown any more: " + [...muted].map(kindLabel).sort().join(", "));
    const thr = this.blinkThreshold();
    lines.push(`Eyes-closed threshold: ${thr}` + (thr !== DEFAULT_BLINK_ON ? " (calibrated)" : " (default)"));
    const p = this.presenceToday();
    lines.push(`Last 24h: away ${p.awayEpisodes}x` + (p.medianAwayS ? ` (median ${p.medianAwayS}s)` : "")
      + `, eyes closed ${p.eyesClosed}x, pinches ${p.pinches}, steps ${p.steps}, scrolls ${p.scrolls}`);
    return lines.join("\n");
  }

  // --- UI -------------------------------------------------------------------
  /** Everything the world panel shows: learned facts plus the behaviour each one drives. */
  snapshot(): WorldSnapshot {
    const samples = newest(this.store.blinkBaselines(), 20).reduce((a, r) => a + r.samples, 0);
    const thr = this.blinkThreshold();
    const soma = this.somaCounts();
    const kinds = this.kinds();
    const muted = this.mutedKinds();
    const rules: Rule[] = [{
      when: "SOMA sees a next step on your screen",
      then: "Pette shows it as a card; nothing happens until you press “Do it” or pinch",
      why: soma.shown
        ? `${soma.shown} shown: ${soma.accepted} done, ${soma.dismissed} not now, ${soma.ignored} ignored`
        : "no suggestions yet",
    }];
    for (const k of kinds) {
      if (muted.has(k.kind)) {
        rules.push({ when: `SOMA suggests “${k.label}”`, then: "Pette doesn't show it any more",
          why: `you said Not now ${k.dismissed} of ${k.accepted + k.dismissed} times` });
      }
    }
    rules.push({ when: "Your eyes look closed on camera", then: `blink score above ${thr}`,
      why: thr !== DEFAULT_BLINK_ON ? `calibrated from ${samples} open-eye samples`
        : "default until the camera has seen enough open-eye frames" });
    const recent = newest(this.store.offers().filter((r) => isSoma(r) && r.outcome !== "shown"), 8)
      .map((r) => ({ ts: r.ts, kind: kindLabel(r.kind), outcome: r.outcome }));
    const events = newest(this.store.events(), 8).map((e) => ({ ts: e.ts, kind: e.kind }));
    return {
      soma, acceptRate: this.acceptRate(), kinds: kinds.slice(0, 8),
      muted: [...muted].map(kindLabel).sort(),
      blinkThreshold: thr, blinkCalibrated: thr !== DEFAULT_BLINK_ON, baselineSamples: samples,
      presence: this.presenceToday(), rules, recent, events, updated: this.clock(),
    };
  }
}
