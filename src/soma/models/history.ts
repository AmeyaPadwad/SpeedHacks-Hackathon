/**
 * Rolling, in-memory buffer of recent screen states for the temporal encoder (port of
 * `runtime/history.py`).
 *
 * Holds at most `len` copies of `[D]` screen-state vectors (the `SomaModel.encodeScreen` output of previous
 * frames), oldest first. Never persisted, no pixels or text; cleared on pause so a resumed session never
 * mixes with what was on screen before.
 */
export class ScreenHistory {
  private buf: Float32Array[] = [];
  private app: string | null = null;

  /**
   * @param len  maximum number of states kept (`ModelConfig.historyLen`).
   * @param opts `resetOnAppSwitch` clears the buffer when {@link observeApp} sees a different app.
   */
  constructor(readonly len: number, readonly opts: { dim?: number; resetOnAppSwitch?: boolean } = {}) {
    if (len < 1) throw new Error("len must be >= 1");
  }

  get size(): number {
    return this.buf.length;
  }

  /** Append a copy of one state; the oldest is dropped beyond `len`. */
  push(state: Float32Array): void {
    if (this.opts.dim !== undefined && state.length !== this.opts.dim) {
      throw new Error(`state has dim ${state.length}, expected ${this.opts.dim}`);
    }
    this.buf.push(Float32Array.from(state));
    if (this.buf.length > this.len) this.buf.splice(0, this.buf.length - this.len);
  }

  /** Copies of the stored states, oldest first. */
  states(): Float32Array[] {
    return this.buf.map((s) => Float32Array.from(s));
  }

  /** Note the foreground app; clears on a switch when `resetOnAppSwitch` is set. */
  observeApp(app: string): void {
    const a = (app || "").toLowerCase();
    if (this.opts.resetOnAppSwitch && this.app !== null && a !== this.app) this.buf = [];
    this.app = a;
  }

  clear(): void {
    this.buf = [];
    this.app = null;
  }
}
