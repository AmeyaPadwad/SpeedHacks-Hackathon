/**
 * `SomaModel` (port of `soma_model.py`): the six modules wired as
 *
 *   pixels ──E_v──┐
 *                 ├─ScreenFusion─> screenState ─Temporal─> temporal ─MemoryFusion─> context ─Policy─> logits [1+A]
 *   OCR ────E_t──┘
 *
 * with a NO_SUGGESTION output at logits index 0. The screen is encoded once per step ({@link encodeScreen});
 * {@link score} runs temporal -> memory fusion -> policy on that state, and {@link step} does both.
 *
 * Weights are seeded (`seed`), so a fresh page load builds the same untrained model. Inference runs inside
 * `tf.tidy` and returns plain typed arrays; no tensors leak. Inference only: there is no offline training
 * in the web app, so dropout is never applied.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import { ActionPolicy } from "./actionPolicy";
import type { ScreenInputs, StepInputs } from "./batch";
import { fitDim, Rng } from "./math";
import { MemoryFusion } from "./memoryFusion";
import { Params } from "./nn";
import { ScreenFusion } from "./screenFusion";
import { TemporalEncoder } from "./temporalEncoder";
import { TextLayoutEncoder } from "./textLayoutEncoder";
import { VisualEncoder } from "./visualEncoder";
import { loadWeights, saveWeights, type WeightsTarget } from "./weightsIO";

export { softmax } from "./math";
export { BLOCKED_LOGIT, NO_SUGGESTION_INDEX } from "./actionPolicy";

/** Stored with the weights; a different version is never loaded. */
export const CHECKPOINT_VERSION = "soma-model-1";

export interface StepOutput {
  /** `[D]` encoded screen; push it to `ScreenHistory` after the step. */
  screenState: Float32Array;
  /** `[D]` screen + history (also a good retrieval query). */
  temporalState: Float32Array;
  /** `[1 + nCandidates]`, index 0 = NO_SUGGESTION. */
  logits: Float32Array;
  /** How much memory was let through, in (0, 1). */
  memoryGate: number;
  /** `[1 + nMemories]`, index 0 = NO_MEMORY. */
  memoryAttention: Float32Array;
}

export class SomaModel {
  readonly params: Params;
  readonly vision: VisualEncoder;
  readonly text: TextLayoutEncoder;
  readonly fusion: ScreenFusion;
  readonly temporal: TemporalEncoder;
  readonly memory: MemoryFusion;
  readonly policy: ActionPolicy;
  private disposed = false;

  constructor(readonly cfg: ModelConfig, readonly seed = 0) {
    if (cfg.dModel % cfg.nHeads) throw new Error(`dModel=${cfg.dModel} must be divisible by nHeads=${cfg.nHeads}`);
    this.params = new Params(new Rng(seed));
    const p = this.params;
    // creation order fixes the RNG stream: keep it stable
    this.vision = new VisualEncoder(p, cfg);
    this.text = new TextLayoutEncoder(p, cfg);
    this.fusion = new ScreenFusion(p, cfg);
    this.temporal = new TemporalEncoder(p, cfg);
    this.memory = new MemoryFusion(p, cfg);
    this.policy = new ActionPolicy(p, cfg);
  }

  /** Number of weights. */
  get size(): number {
    let n = 0;
    for (const v of this.params.vars.values()) n += v.size;
    return n;
  }

  private check(): void {
    if (this.disposed) throw new Error("SomaModel was disposed");
  }

  private screenTensor(inp: ScreenInputs): tf.Tensor1D {
    const pad = Array.from(inp.ocrPadding, (x) => x !== 0);
    const visual = this.vision.apply(inp.pixels);
    const text = this.text.apply(inp.ocrIds, inp.ocrBoxes, pad);
    return this.fusion.apply(visual, text, pad);
  }

  /** Screen state `[D]` for one screenshot + OCR (what the history stores per step). */
  encodeScreen(inp: ScreenInputs): Float32Array {
    this.check();
    let out = new Float32Array(0);
    tf.tidy(() => {
      out = this.screenTensor(inp).dataSync() as Float32Array;
    });
    return Float32Array.from(out);
  }

  private scoreTensors(screen: tf.Tensor1D, inp: Omit<StepInputs, keyof ScreenInputs>, history: Float32Array[]) {
    const D = this.cfg.dModel;
    const temporal = this.temporal.apply(screen, history, inp.appId);
    const mems = inp.memoryEmbeddings.slice(0, this.cfg.maxMemories).map((m) => fitDim(m, D));
    const memT = tf.tensor2d(flat(mems, D), [mems.length, D]);
    const mem = this.memory.apply(temporal, memT);
    const A = inp.candidateEmbeddings.length;
    const cands = inp.candidateEmbeddings.map((c) => fitDim(c, D));
    const blocked = inp.candidateMask ? Array.from({ length: A }, (_, i) => inp.candidateMask![i] !== 0) : undefined;
    const logits = this.policy.apply(mem.context, tf.tensor2d(flat(cands, D), [A, D]), inp.candidateTypes, blocked);
    return { temporal, logits, gate: mem.gate, attention: mem.attention };
  }

  /**
   * Temporal -> memory fusion -> policy on an already encoded `screenState` (the pipeline's `score`).
   * `history` holds previous screen states, oldest first.
   */
  score(screenState: Float32Array, inp: Omit<StepInputs, keyof ScreenInputs>, history: Float32Array[]): StepOutput {
    this.check();
    let res!: StepOutput;
    tf.tidy(() => {
      const r = this.scoreTensors(tf.tensor1d(screenState), inp, history);
      res = {
        screenState: Float32Array.from(screenState),
        temporalState: Float32Array.from(r.temporal.dataSync()),
        logits: Float32Array.from(r.logits.dataSync()),
        memoryGate: r.gate.dataSync()[0],
        memoryAttention: Float32Array.from(r.attention.dataSync()),
      };
    });
    return res;
  }

  /** Full step: encode the screen once, then score. Does not touch any history buffer. */
  step(inp: StepInputs, history: Float32Array[]): StepOutput {
    this.check();
    let res!: StepOutput;
    tf.tidy(() => {
      const screen = this.screenTensor(inp);
      const r = this.scoreTensors(screen, inp, history);
      res = {
        screenState: Float32Array.from(screen.dataSync()),
        temporalState: Float32Array.from(r.temporal.dataSync()),
        logits: Float32Array.from(r.logits.dataSync()),
        memoryGate: r.gate.dataSync()[0],
        memoryAttention: Float32Array.from(r.attention.dataSync()),
      };
    });
    return res;
  }

  /** Save weights + config to a tfjs URL (e.g. `indexeddb://soma-model`) or an IOHandler. */
  async save(url: WeightsTarget): Promise<void> {
    this.check();
    const snap = this.params.snapshot();
    try {
      await saveWeights(url, snap, { format: CHECKPOINT_VERSION, config: this.cfg, seed: this.seed });
    } finally {
      tf.dispose(snap);
    }
  }

  /**
   * Load weights saved by {@link save}. Returns false (weights unchanged) when nothing is stored there, the
   * version differs, or the stored shapes do not match this model's config.
   */
  async load(url: WeightsTarget): Promise<boolean> {
    this.check();
    const got = await loadWeights(url, CHECKPOINT_VERSION);
    if (!got) return false;
    try {
      this.params.assign(got.tensors);
      return true;
    } catch {
      return false;
    } finally {
      tf.dispose(got.tensors);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.params.dispose();
  }
}

function flat(rows: Float32Array[], d: number): Float32Array {
  const out = new Float32Array(rows.length * d);
  rows.forEach((r, i) => out.set(r, i * d));
  return out;
}
