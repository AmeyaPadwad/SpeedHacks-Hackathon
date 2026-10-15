/**
 * Plain typed-array features for the model (port of soma/perception/featurize.py; no tfjs here).
 *
 * One token entry per word (hashed with `encode`); each token carries the normalized xyxy box of the
 * element it came from. Sequences are truncated / padded to `cfg.maxOcrTokens` (L). In the mask,
 * 1 = padding (PyTorch's `key_padding_mask` convention). Errors, file refs and buttons are moved to the
 * front before truncation so they survive long screens.
 */
import type { ModelConfig } from "../config";
import type { OCRElement, ScreenObservation } from "../types";
import { letterbox, type Pixels } from "./preprocess";
import { PAD_ID, encode, hashBucket } from "./tokenizer";

/** Roles moved to the front before truncation. */
export const PRIORITY_ROLES: readonly string[] = ["error", "file_ref", "button"];

/** Stable hashed bucket of the app name (case-insensitive). */
export const appBucket = (appName: string, nApps: number): number => hashBucket((appName ?? "").toLowerCase(), nApps);

function ordered(elements: readonly OCRElement[], prioritize: boolean): OCRElement[] {
  if (!prioritize) return [...elements];
  const rank = (e: OCRElement) => {
    const i = PRIORITY_ROLES.indexOf(e.uiRole ?? "");
    return i < 0 ? PRIORITY_ROLES.length : i;
  };
  // stable: keeps reading order within each group
  return elements.map((e, i) => ({ e, i })).sort((a, b) => rank(a.e) - rank(b.e) || a.i - b.i).map((x) => x.e);
}

export interface OcrArrays {
  /** [L] token ids, PAD_ID (0) in padding. */
  ids: Int32Array;
  /** [L*4] normalized xyxy per token, 0 in padding. */
  boxes: Float32Array;
  /** [L], 1 = padding. Padding is always a suffix. */
  mask: Uint8Array;
}

/** OCR arrays for one observation. `null` or an empty observation is all padding. */
export function observationToArrays(obs: ScreenObservation | null, cfg: Pick<ModelConfig, "maxOcrTokens" | "vocabSize">, prioritize = true): OcrArrays {
  const L = cfg.maxOcrTokens;
  const ids = new Int32Array(L).fill(PAD_ID);
  const boxes = new Float32Array(L * 4);
  const mask = new Uint8Array(L).fill(1);
  if (!obs) return { ids, boxes, mask };
  let n = 0;
  for (const e of ordered(obs.elements, prioritize)) {
    if (n >= L) break;
    const toks = encode(e.text, cfg.vocabSize, L - n);
    if (!toks.length) continue;
    const box = e.bbox.map((v) => Math.min(1, Math.max(0, Number(v))));
    for (const t of toks) {
      ids[n] = t;
      boxes.set(box, n * 4);
      mask[n] = 0;
      n += 1;
    }
  }
  return { ids, boxes, mask };
}

/**
 * Letterbox `image` into `size x size` RGB, ImageNet-normalized, HWC (`[size*size*3]`), like
 * `preprocess.frame_to_tensor` (which was CHW; tfjs wants HWC).
 */
export const imageToPixels = (image: ImageData | Pixels, size: number): Float32Array => letterbox(image, size, true).pixels;
