/**
 * Memory fusion (port of `memory_fusion.py`): gated cross-attention from the temporal state to retrieved
 * memories. A learned NO_MEMORY key/value is always prepended and never masked, so an empty retrieval
 * attends fully to it instead of softmaxing over nothing.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import { gelu, LayerNorm, Linear, LinearNorm, MultiheadAttention, Params } from "./nn";

export interface MemoryFusionOutput {
  /** Personalised context `[D]`. */
  context: tf.Tensor1D;
  /** How much memory was let through, a scalar in (0, 1). */
  gate: tf.Scalar;
  /** Attention `[K+1]`; index 0 = NO_MEMORY. */
  attention: tf.Tensor1D;
}

export class MemoryFusion {
  readonly noMemory: tf.Variable;
  readonly memoryProj: LinearNorm;
  readonly queryNorm: LayerNorm;
  readonly attn: MultiheadAttention;
  readonly gate: Linear;
  readonly norm1: LayerNorm;
  readonly ffn1: Linear;
  readonly ffn2: Linear;
  readonly norm2: LayerNorm;
  readonly dim: number;

  constructor(p: Params, cfg: ModelConfig) {
    const D = cfg.dModel;
    this.dim = D;
    this.noMemory = p.normal("memory.null", [1, D], 0.02);
    this.memoryProj = new LinearNorm(p, "memory.proj", D, D);
    this.queryNorm = new LayerNorm(p, "memory.queryNorm", D);
    this.attn = new MultiheadAttention(p, "memory.attn", D, cfg.nHeads);
    this.gate = new Linear(p, "memory.gate", 2 * D, 1);
    this.norm1 = new LayerNorm(p, "memory.norm1", D);
    this.ffn1 = new Linear(p, "memory.ffn1", D, 4 * D);
    this.ffn2 = new Linear(p, "memory.ffn2", 4 * D, D);
    this.norm2 = new LayerNorm(p, "memory.norm2", D);
  }

  /** Fuse `state [D]` with `memories [K, D]` (K may be 0). */
  apply(state: tf.Tensor1D, memories: tf.Tensor2D): MemoryFusionOutput {
    const kv = memories.shape[0]
      ? tf.concat([this.noMemory as tf.Tensor2D, this.memoryProj.apply(memories)], 0)
      : (this.noMemory as tf.Tensor2D);
    const q = this.queryNorm.apply(state).expandDims(0) as tf.Tensor2D;
    const { out, weights } = this.attn.apply(q, kv, kv);
    const attended = out.squeeze([0]) as tf.Tensor1D;
    const gate = tf.sigmoid(this.gate.apply(tf.concat([state, attended]))).squeeze() as tf.Scalar;
    const x = this.norm1.apply(tf.add(state, tf.mul(gate, attended))) as tf.Tensor1D;
    const context = this.norm2.apply(tf.add(x, this.ffn2.apply(gelu(this.ffn1.apply(x))))) as tf.Tensor1D;
    return { context, gate, attention: weights.squeeze([0]) as tf.Tensor1D };
  }
}
