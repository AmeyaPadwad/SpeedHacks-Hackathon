/**
 * The step loop (port of SOMA/soma/runtime/scheduler.py), as a `setTimeout` chain on the page's event loop.
 *
 * - **Backpressure:** at most one step is in flight; the next one is scheduled only after it finishes.
 *   A trigger arriving while a step runs is coalesced into one forced follow-up step.
 * - **Stale results are dropped:** `pause()` / `stop()` bump a generation counter; a result computed under
 *   an older generation is never delivered.
 * - **Adaptive interval:** unchanged / rate-limited / silent steps back the interval off geometrically up to
 *   `maxIntervalS`; a shown suggestion resets it. `trigger()` runs immediately with `force = true`, which
 *   bypasses rate limit and change detection but never the pause / screen-consent gates.
 */

/** Skip reasons that mean "nothing new on screen" (back off). */
export const IDLE_SKIPS = ["unchanged", "rate-limited"];

interface ResultLike {
  skippedReason: string | null;
  suggestion: { candidate: unknown | null };
}

export interface SchedulerRuntime<R extends ResultLike = ResultLike> {
  step(force?: boolean): Promise<R>;
  pause(): void;
  resume(): void;
}

export interface Timers {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export interface SchedulerOptions<R> {
  onResult?: (result: R) => void;
  /** Errors never stop the loop. */
  onError?: (err: unknown) => void;
  /** Back-off ceiling (default `4 * intervalS`). */
  maxIntervalS?: number;
  /** Multiplier applied after an idle / silent step (>= 1, default 1.5). */
  backoff?: number;
  timers?: Timers;
}

const defaultTimers: Timers = {
  setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
  clearTimeout: (h) => globalThis.clearTimeout(h as ReturnType<typeof setTimeout>),
};

export class InferenceScheduler<R extends ResultLike = ResultLike> {
  readonly baseIntervalS: number;
  readonly maxIntervalS: number;
  readonly backoff: number;
  steps = 0;
  delivered = 0;
  droppedStale = 0;
  errors = 0;

  private interval: number;
  private timer: unknown = null;
  private runningFlag = false;
  private pausedFlag = false;
  private forced = false;
  private busy: Promise<void> | null = null;
  private gen = 0;
  private readonly timers: Timers;

  constructor(readonly runtime: SchedulerRuntime<R>, intervalS = 1, readonly opts: SchedulerOptions<R> = {}) {
    if (!(intervalS > 0)) throw new RangeError("intervalS must be > 0");
    this.baseIntervalS = intervalS;
    this.maxIntervalS = opts.maxIntervalS ?? 4 * intervalS;
    this.backoff = Math.max(1, opts.backoff ?? 1.5);
    this.interval = intervalS;
    this.timers = opts.timers ?? defaultTimers;
  }

  get running(): boolean {
    return this.runningFlag;
  }

  get paused(): boolean {
    return this.pausedFlag;
  }

  get inFlight(): boolean {
    return this.busy !== null;
  }

  /** Current (adaptive) interval in seconds. */
  get intervalS(): number {
    return this.interval;
  }

  /** Resolves when the in-flight step (if any) has finished and been delivered. */
  idle(): Promise<void> {
    return this.busy ?? Promise.resolve();
  }

  start(): void {
    if (this.runningFlag) return;
    this.runningFlag = true;
    this.schedule(this.forced ? 0 : this.interval * 1000);
  }

  /** Stop the loop; an in-flight result is discarded. */
  stop(): void {
    this.gen += 1;
    this.runningFlag = false;
    this.clearTimer();
  }

  /** Pause instantly: the runtime stops capturing and in-flight results go stale. */
  pause(): void {
    this.pausedFlag = true;
    this.gen += 1;
    this.clearTimer();
    this.runtime.pause();
  }

  resume(): void {
    this.runtime.resume();
    this.pausedFlag = false;
    this.gen += 1;
    this.interval = this.baseIntervalS;
    if (this.runningFlag && !this.busy) this.schedule(0);
  }

  /**
   * Request an immediate forced step. Coalesces while a step runs. Works without `start()` too (one step).
   * Returns false while paused.
   */
  trigger(): boolean {
    if (this.pausedFlag) return false;
    this.forced = true;
    if (this.busy) return true;
    this.clearTimer();
    if (this.runningFlag) this.schedule(0);
    else void this.tick();
    return true;
  }

  private clearTimer(): void {
    if (this.timer !== null) this.timers.clearTimeout(this.timer);
    this.timer = null;
  }

  private schedule(ms: number): void {
    this.clearTimer();
    this.timer = this.timers.setTimeout(() => {
      this.timer = null;
      void this.tick();
    }, ms);
  }

  private adapt(result: R): void {
    const reason = result.skippedReason;
    const silent = result.suggestion?.candidate == null;
    const idle = (reason !== null && IDLE_SKIPS.includes(reason.split(":")[0])) || (reason === null && silent);
    if (reason === null && !silent) this.interval = this.baseIntervalS;
    else if (idle) this.interval = Math.min(this.maxIntervalS, this.interval * this.backoff);
  }

  private tick(): Promise<void> {
    if (this.busy || this.pausedFlag) return this.busy ?? Promise.resolve();
    const forced = this.forced;
    this.forced = false;
    const gen = this.gen;
    const run = (async () => {
      let result: R | null = null;
      try {
        result = await this.runtime.step(forced);
      } catch (err) {
        this.errors += 1;
        try {
          this.opts.onError?.(err);
        } catch {
          // a UI bug must not stop inference
        }
      }
      this.steps += 1;
      if (result !== null) {
        if (gen !== this.gen || this.pausedFlag) {
          this.droppedStale += 1;
        } else {
          this.adapt(result);
          this.delivered += 1;
          try {
            this.opts.onResult?.(result);
          } catch (err) {
            this.errors += 1;
            void err;
          }
        }
      }
    })();
    this.busy = run.finally(() => {
      this.busy = null;
      if (this.pausedFlag) return;
      if (this.forced) {
        if (this.runningFlag) this.schedule(0);
        else void this.tick();
      } else if (this.runningFlag) {
        this.schedule(this.interval * 1000);
      }
    });
    return this.busy;
  }
}
