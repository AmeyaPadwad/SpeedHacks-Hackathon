/**
 * Visual encoder `E_v` (port of `visual_encoder.VisualEncoder`, the from-scratch ViT; the torchvision
 * backbone is dropped): an `S x S x 3` screenshot -> patch tokens `[N, D]`, `N = (S / patch)^2`.
 *
 * Pixels are HWC row-major floats (already normalised). The patch "conv" is a linear map over flattened
 * `patch x patch x 3` patches, which is the same operation as a stride-`patch` Conv2d.
 */
import * as tf from "@tensorflow/tfjs";
import type { ModelConfig } from "../config";
import { Embedding, Linear, Params, TransformerEncoder } from "./nn";

export class VisualEncoder {
  readonly grid: number;
  readonly patch: number;
  readonly patchEmbed: Linear;
  readonly rowEmbed: Embedding;
  readonly colEmbed: Embedding;
  readonly encoder: TransformerEncoder;

  constructor(p: Params, cfg: ModelConfig) {
    if (cfg.imageSize % cfg.patchSize) throw new Error("imageSize must be a multiple of patchSize");
    const D = cfg.dModel;
    this.patch = cfg.patchSize;
    this.grid = cfg.imageSize / cfg.patchSize;
    this.patchEmbed = new Linear(p, "vision.patch", 3 * this.patch * this.patch, D);
    this.rowEmbed = new Embedding(p, "vision.row", this.grid, D, { std: 0.02 });
    this.colEmbed = new Embedding(p, "vision.col", this.grid, D, { std: 0.02 });
    this.encoder = new TransformerEncoder(p, "vision.encoder", D, cfg.nHeads, cfg.visionLayers);
  }

  /** `N` for a full-size input. */
  get numTokens(): number {
    return this.grid * this.grid;
  }

  /**
   * Encode `pixels` (`S*S*3`, square) into `[N, D]`. Smaller inputs use a prefix of the positional tables;
   * larger ones (or non-multiples of the patch size) are rejected.
   */
  apply(pixels: Float32Array): tf.Tensor2D {
    const side = Math.round(Math.sqrt(pixels.length / 3));
    if (side * side * 3 !== pixels.length) throw new Error("pixels must be a square S*S*3 image");
    if (side % this.patch) throw new Error(`image side ${side} is not a multiple of patch ${this.patch}`);
    const g = side / this.patch;
    if (g > this.grid) throw new Error(`image grid ${g}x${g} exceeds configured ${this.grid}x${this.grid}`);
    const p = this.patch;
    const img = tf.tensor(pixels, [g, p, g, p, 3]);
    const patches = tf.reshape(tf.transpose(img, [0, 2, 1, 3, 4]), [g * g, p * p * 3]) as tf.Tensor2D;
    const idx = Array.from({ length: g }, (_, i) => i);
    const pos = tf.add(this.rowEmbed.gather(idx).expandDims(1), this.colEmbed.gather(idx).expandDims(0));
    const x = tf.add(this.patchEmbed.apply(patches), tf.reshape(pos, [g * g, -1])) as tf.Tensor2D;
    return this.encoder.apply(x);
  }
}
