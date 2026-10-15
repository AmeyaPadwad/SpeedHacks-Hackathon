/**
 * Screen fusion `F` (port of `screen_fusion.py`): visual + OCR tokens with modality-type embeddings -> small
 * transformer -> learned-query pool -> screen state `[D]`. Visual tokens are never masked, so attention
 * always has valid keys even when OCR is empty.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import { additiveKeyMask, AttentionPool, Embedding, Params, TransformerEncoder, zeroPadded } from "./nn";

export const VISUAL_TYPE = 0;
export const TEXT_TYPE = 1;

export class ScreenFusion {
  readonly typeEmbed: Embedding;
  readonly encoder: TransformerEncoder;
  readonly pool: AttentionPool;

  constructor(p: Params, cfg: ModelConfig) {
    const D = cfg.dModel;
    this.typeEmbed = new Embedding(p, "fusion.type", 2, D, { std: 0.02 });
    this.encoder = new TransformerEncoder(p, "fusion.encoder", D, cfg.nHeads, cfg.fusionLayers);
    this.pool = new AttentionPool(p, "fusion.pool", D, cfg.nHeads);
  }

  /** Fuse `visual [N, D]` and `text [L, D]` (`ocrPad [L]`, true = PAD) into the screen state `[D]`. */
  apply(visual: tf.Tensor2D, text: tf.Tensor2D, ocrPad: boolean[]): tf.Tensor1D {
    const N = visual.shape[0];
    const vis = tf.add(visual, this.typeEmbed.row(VISUAL_TYPE)) as tf.Tensor2D;
    const txt = zeroPadded(tf.add(text, this.typeEmbed.row(TEXT_TYPE)) as tf.Tensor2D, ocrPad);
    const tokens = tf.concat([vis, txt], 0);
    const pad = [...new Array<boolean>(N).fill(false), ...ocrPad];
    const fused = zeroPadded(this.encoder.apply(tokens, additiveKeyMask(pad)), pad);
    return this.pool.apply(fused, pad);
  }
}
