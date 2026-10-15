/**
 * Text/layout encoder `E_t` (port of `text_layout_encoder.py`): OCR ids `[L]` + boxes `[L, 4]` -> `[L, D]`.
 * Padded positions are zero in the output; no OCR at all gives an all-zero block, never NaN.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import {
  additiveKeyMask, Embedding, LayerNorm, Linear, Params, safePaddingMask, TransformerEncoder, zeroPadded,
} from "./nn";

/** Token id 0 is PAD (see the hashed tokenizer). */
export const PAD_ID = 0;

/** Normalised xyxy boxes `[L*4]` -> `[L, 6]` = (x0, y0, x1, y1, w, h). */
export function boxFeatures(boxes: Float32Array, L: number): tf.Tensor2D {
  const out = new Float32Array(L * 6);
  for (let i = 0; i < L; i++) {
    const [x0, y0, x1, y1] = [boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
    out.set([x0, y0, x1, y1, x1 - x0, y1 - y0], i * 6);
  }
  return tf.tensor2d(out, [L, 6]);
}

export class TextLayoutEncoder {
  readonly tokenEmbed: Embedding;
  readonly boxProj: Linear;
  readonly posEmbed: Embedding;
  readonly norm: LayerNorm;
  readonly encoder: TransformerEncoder;
  readonly maxLen: number;
  readonly dim: number;

  constructor(p: Params, cfg: ModelConfig) {
    const D = cfg.dModel;
    this.dim = D;
    this.maxLen = cfg.maxOcrTokens;
    this.tokenEmbed = new Embedding(p, "text.token", cfg.vocabSize, D, { paddingIdx: PAD_ID });
    this.boxProj = new Linear(p, "text.box", 6, D);
    this.posEmbed = new Embedding(p, "text.pos", cfg.maxOcrTokens, D, { std: 0.02 });
    this.norm = new LayerNorm(p, "text.norm", D);
    this.encoder = new TransformerEncoder(p, "text.encoder", D, cfg.nHeads, cfg.textLayers);
  }

  /** Encode `ids [L]`, `boxes [L*4]` with `pad [L]` (true = PAD) into `[L, D]`. */
  apply(ids: Int32Array, boxes: Float32Array, pad: boolean[]): tf.Tensor2D {
    const L = ids.length;
    if (L > this.maxLen) throw new Error(`${L} OCR tokens exceed maxOcrTokens=${this.maxLen}`);
    if (boxes.length !== L * 4 || pad.length !== L) throw new Error("ocrBoxes / ocrPadding do not match ocrIds");
    if (L === 0) return tf.zeros([0, this.dim]);
    const pos = this.posEmbed.gather(Array.from({ length: L }, (_, i) => i));
    let x = tf.add(tf.add(this.tokenEmbed.gather(ids), this.boxProj.apply(boxFeatures(boxes, L))), pos) as tf.Tensor2D;
    x = zeroPadded(this.norm.apply(x), pad);
    x = this.encoder.apply(x, additiveKeyMask(safePaddingMask(pad)));
    return zeroPadded(x, pad);
  }
}
