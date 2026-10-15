import { describe, expect, it } from "vitest";
import { InferenceScheduler, type Timers } from "./scheduler";

interface R { skippedReason: string | null; suggestion: { candidate: unknown | null } }
const result = (silent: boolean, skipped: string | null = null): R => ({ skippedReason: skipped, suggestion: { candidate: silent ? null : {} } });

/** Manual timers: `fire()` runs the pending timeout (if any). */
class ManualTimers implements Timers {
  pending: { fn: () => void; ms: number } | null = null;
  setTimeout(fn: () => void, ms: number) {
    this.pending = { fn, ms };
    return this.pending;
  }
  clearTimeout(h: unknown) {
    if (h === this.pending) this.pending = null;
  }
  fire() {
    const p = this.pending;
    this.pending = null;
    p?.fn();
  }
}

class ScriptedRuntime {
  calls = 0;
  forced: boolean[] = [];
  paused = false;
  inFlight = 0;
  maxInFlight = 0;
  release: (() => void) | null = null;
  hold = false;
  constructor(public results: R[]) {}
  async step(force = false): Promise<R> {
    this.calls++;
    this.forced.push(force);
    this.inFlight++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    if (this.hold) await new Promise<void>((r) => (this.release = r));
    this.inFlight--;
    return this.results[Math.min(this.calls - 1, this.results.length - 1)];
  }
  pause() { this.paused = true; }
  resume() { this.paused = false; }
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("InferenceScheduler", () => {
  it("start / stop are idempotent; periodic steps are not forced, trigger is", async () => {
    const rt = new ScriptedRuntime([result(true)]);
    const timers = new ManualTimers();
    const s = new InferenceScheduler(rt, 2, { timers });
    s.start();
    s.start();
    expect(s.running).toBe(true);
    expect(timers.pending?.ms).toBe(2000);
    timers.fire();
    await s.idle();
    expect(rt.forced).toEqual([false]);
    expect(s.trigger()).toBe(true);
    timers.fire();
    await s.idle();
    expect(rt.forced).toEqual([false, true]);
    s.stop();
    s.stop();
    expect(s.running).toBe(false);
    expect(timers.pending).toBeNull();
  });

  it("backs off on unchanged / silent steps and resets on a shown suggestion", async () => {
    const rt = new ScriptedRuntime([...Array(6).fill(result(true, "unchanged")), result(false)]);
    const timers = new ManualTimers();
    const got: R[] = [];
    const s = new InferenceScheduler(rt, 1, { timers, maxIntervalS: 5, backoff: 2, onResult: (r) => got.push(r) });
    s.start();
    const seen: number[] = [];
    for (let i = 0; i < 7; i++) {
      timers.fire();
      await s.idle();
      seen.push(s.intervalS);
    }
    expect(seen).toEqual([2, 4, 5, 5, 5, 5, 1]);
    expect(got).toHaveLength(7);
    expect(timers.pending?.ms).toBe(1000);
  });

  it("never overlaps steps; triggers while busy coalesce into one forced follow-up", async () => {
    const rt = new ScriptedRuntime([result(true)]);
    rt.hold = true;
    const timers = new ManualTimers();
    const s = new InferenceScheduler(rt, 1, { timers });
    s.start();
    timers.fire();
    await flush();
    expect(s.inFlight).toBe(true);
    s.trigger();
    s.trigger();
    expect(rt.calls).toBe(1);
    rt.release!();
    await flush();
    await flush();
    expect(timers.pending?.ms).toBe(0);
    timers.fire();
    await flush();
    rt.release!();
    await s.idle();
    expect(rt.calls).toBe(2);
    expect(rt.forced).toEqual([false, true]);
    expect(rt.maxInFlight).toBe(1);
  });

  it("pause drops the in-flight result and stops stepping; resume restarts", async () => {
    const rt = new ScriptedRuntime([result(false)]);
    rt.hold = true;
    const timers = new ManualTimers();
    const got: R[] = [];
    const s = new InferenceScheduler(rt, 1, { timers, onResult: (r) => got.push(r) });
    s.start();
    timers.fire();
    await flush();
    s.pause();
    expect(rt.paused).toBe(true);
    rt.release!();
    await s.idle();
    expect(got).toEqual([]);
    expect(s.droppedStale).toBe(1);
    expect(timers.pending).toBeNull();
    expect(s.trigger()).toBe(false);
    rt.hold = false;
    s.resume();
    expect(rt.paused).toBe(false);
    timers.fire();
    await s.idle();
    expect(got).toHaveLength(1);
  });

  it("errors never stop the loop; trigger works without start", async () => {
    let n = 0;
    const errors: unknown[] = [];
    const timers = new ManualTimers();
    const rt = { step: async () => { if (n++ === 0) throw new Error("boom"); return result(true); }, pause() {}, resume() {} };
    const s = new InferenceScheduler<R>(rt, 1, { timers, onError: (e) => errors.push(e) });
    s.start();
    timers.fire();
    await s.idle();
    expect(errors).toHaveLength(1);
    expect(timers.pending?.ms).toBe(1000);
    s.stop();
    const once = new InferenceScheduler<R>(new ScriptedRuntime([result(true)]), 1, { timers: new ManualTimers() });
    expect(once.trigger()).toBe(true);
    await once.idle();
    expect(once.steps).toBe(1);
  });
});
