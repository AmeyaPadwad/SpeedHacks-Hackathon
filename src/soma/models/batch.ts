/**
 * Tensor contract for {@link SomaModel} (port of `batch.py`), as plain typed arrays for one step.
 *
 * Masks follow the Python convention: 1 / true = PAD (or blocked), ignored by the model. The model never
 * keeps any of these arrays: pixels are used for one encode and dropped.
 */
import type { ModelConfig } from "../config";

export interface StepInputs {
  /** `S*S*3` HWC row-major floats, already normalised; `S` <= `imageSize`, a multiple of `patchSize`. */
  pixels: Float32Array;
  /** `[L]` hashed OCR token ids (0 = PAD). */
  ocrIds: Int32Array;
  /** `[L*4]` normalised xyxy boxes. */
  ocrBoxes: Float32Array;
  /** `[L]`, 1 = PAD. */
  ocrPadding: Uint8Array;
  /** Hashed app bucket in `[0, nApps)`. */
  appId: number;
  /** Up to `maxMemories` retrieved memory embeddings (each fitted to `dModel`). */
  memoryEmbeddings: Float32Array[];
  /** One embedding per candidate (each fitted to `dModel`). */
  candidateEmbeddings: Float32Array[];
  /** ActionType index per candidate (1.., 0 is NO_SUGGESTION). */
  candidateTypes: number[];
  /** Optional `[A]`, 1 = blocked by the safety gate; blocked candidates get `BLOCKED_LOGIT`. */
  candidateMask?: Uint8Array;
}

/** Screen-only part of {@link StepInputs}. */
export type ScreenInputs = Pick<StepInputs, "pixels" | "ocrIds" | "ocrBoxes" | "ocrPadding">;

/** Seeded, well-formed random inputs for shape tests and smoke runs (port of `random_batch`, one sample). */
export function randomInputs(cfg: ModelConfig, opts: {
  seed?: number; L?: number; K?: number; A?: number; imageSize?: number;
  emptyOcr?: boolean; allBlocked?: boolean;
} = {}): StepInputs {
  let s = ((opts.seed ?? 0) * 2654435761 + 1) % 4294967296;
  const rand = (): number => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  const S = opts.imageSize ?? cfg.imageSize;
  const L = opts.L ?? 8;
  const K = opts.K ?? 3;
  const A = opts.A ?? 4;
  const D = cfg.dModel;
  const vec = (): Float32Array => Float32Array.from({ length: D }, () => rand() * 2 - 1);
  const ocrPadding = new Uint8Array(L);
  for (let i = 0; i < L; i++) ocrPadding[i] = opts.emptyOcr || i >= L / 2 ? 1 : 0;
  const ocrBoxes = new Float32Array(L * 4);
  for (let i = 0; i < L; i++) {
    const x0 = rand() * 0.8;
    const y0 = rand() * 0.8;
    ocrBoxes.set([x0, y0, x0 + 0.1, y0 + 0.1], i * 4);
  }
  return {
    pixels: Float32Array.from({ length: S * S * 3 }, () => rand() * 2 - 1),
    ocrIds: Int32Array.from({ length: L }, (_, i) => (ocrPadding[i] ? 0 : 1 + Math.floor(rand() * (cfg.vocabSize - 1)))),
    ocrBoxes,
    ocrPadding,
    appId: Math.floor(rand() * cfg.nApps),
    memoryEmbeddings: Array.from({ length: K }, vec),
    candidateEmbeddings: Array.from({ length: A }, vec),
    candidateTypes: Array.from({ length: A }, () => 1 + Math.floor(rand() * (cfg.nActionTypes - 1))),
    candidateMask: opts.allBlocked ? new Uint8Array(A).fill(1) : undefined,
  };
}
