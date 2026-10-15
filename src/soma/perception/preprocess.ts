/**
 * Image preprocessing on plain typed arrays (port of soma/perception/preprocess.py; no tfjs here).
 *
 * - `letterboxGeometry` / `LetterboxTransform` map boxes between a frame and the square model input.
 * - `letterbox` resizes RGBA pixels into a square float RGB image with antialiased bilinear filtering
 *   (like `F.interpolate(..., antialias=True)`), black padding, optional ImageNet normalization.
 * - `changeRatio` is the cheap downsampled frame difference used for event-triggered capture.
 */

export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

/** Minimal image shape: RGBA bytes, row-major (what `ImageData` provides). */
export interface Pixels {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

export type Box = [number, number, number, number];

/** Geometry of a letterbox resize from `srcSize` (w, h) into a `dstSize` square. */
export interface LetterboxTransform {
  srcSize: [number, number];
  dstSize: number;
  scale: number;
  padX: number;
  padY: number;
  /** Scaled content (w, h) inside the destination. */
  newSize: [number, number];
}

/** Letterbox transform for a `(w, h)` frame into `size`². */
export function letterboxGeometry(srcSize: [number, number], size: number): LetterboxTransform {
  const w = Math.max(1, Math.trunc(srcSize[0]));
  const h = Math.max(1, Math.trunc(srcSize[1]));
  const scale = Math.min(size / w, size / h);
  const nw = Math.max(1, Math.round(w * scale));
  const nh = Math.max(1, Math.round(h * scale));
  return { srcSize: [w, h], dstSize: size, scale, padX: Math.floor((size - nw) / 2), padY: Math.floor((size - nh) / 2), newSize: [nw, nh] };
}

/** Map a normalized box on the source frame to the model input. */
export function srcToDst(tf: LetterboxTransform, box: Box): Box {
  const [sw, sh] = tf.srcSize;
  const d = tf.dstSize;
  const fx = (v: number) => (v * sw * tf.scale + tf.padX) / d;
  const fy = (v: number) => (v * sh * tf.scale + tf.padY) / d;
  return [fx(box[0]), fy(box[1]), fx(box[2]), fy(box[3])];
}

/** Map a normalized box on the model input back to the source frame (clipped to [0, 1]). */
export function dstToSrc(tf: LetterboxTransform, box: Box): Box {
  const [sw, sh] = tf.srcSize;
  const d = tf.dstSize;
  const clip = (v: number) => Math.min(1, Math.max(0, v));
  const fx = (v: number) => clip((v * d - tf.padX) / (tf.scale * sw));
  const fy = (v: number) => clip((v * d - tf.padY) / (tf.scale * sh));
  return [fx(box[0]), fy(box[1]), fx(box[2]), fy(box[3])];
}

/** Per-output-index source taps for an antialiased linear (triangle) filter, PyTorch-style. */
function taps(inSize: number, outSize: number): { start: Int32Array; weights: Float32Array[] } {
  const scale = inSize / outSize;
  const support = Math.max(1, scale);
  const start = new Int32Array(outSize);
  const weights: Float32Array[] = [];
  for (let i = 0; i < outSize; i++) {
    const center = scale * (i + 0.5);
    const lo = Math.max(0, Math.floor(center - support + 0.5));
    const hi = Math.min(inSize, Math.floor(center + support + 0.5));
    const w = new Float32Array(Math.max(1, hi - lo));
    let total = 0;
    for (let j = lo; j < hi; j++) {
      const v = Math.max(0, 1 - Math.abs((j - center + 0.5) / support));
      w[j - lo] = v;
      total += v;
    }
    if (total > 0) for (let k = 0; k < w.length; k++) w[k] /= total;
    else w[0] = 1;
    start[i] = hi > lo ? lo : Math.min(inSize - 1, Math.max(0, Math.floor(center)));
    weights.push(w);
  }
  return { start, weights };
}

/**
 * Resize the RGB channels of `img` to `outW x outH` floats in [0, 1] (HWC, 3 channels).
 * Separable antialiased bilinear; alpha is ignored.
 */
export function resizeRgb(img: Pixels, outW: number, outH: number): Float32Array {
  const { width: w, height: h, data } = img;
  const tx = taps(w, outW);
  const ty = taps(h, outH);
  // horizontal pass: [h, outW, 3]
  const mid = new Float32Array(h * outW * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < outW; x++) {
      const s = tx.start[x];
      const wt = tx.weights[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < wt.length; k++) {
        const p = (y * w + s + k) * 4;
        r += data[p] * wt[k];
        g += data[p + 1] * wt[k];
        b += data[p + 2] * wt[k];
      }
      const o = (y * outW + x) * 3;
      mid[o] = r; mid[o + 1] = g; mid[o + 2] = b;
    }
  }
  // vertical pass: [outH, outW, 3]
  const out = new Float32Array(outH * outW * 3);
  for (let y = 0; y < outH; y++) {
    const s = ty.start[y];
    const wt = ty.weights[y];
    for (let x = 0; x < outW; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < wt.length; k++) {
        const p = ((s + k) * outW + x) * 3;
        r += mid[p] * wt[k];
        g += mid[p + 1] * wt[k];
        b += mid[p + 2] * wt[k];
      }
      const o = (y * outW + x) * 3;
      out[o] = Math.min(1, Math.max(0, r / 255));
      out[o + 1] = Math.min(1, Math.max(0, g / 255));
      out[o + 2] = Math.min(1, Math.max(0, b / 255));
    }
  }
  return out;
}

/**
 * Letterbox `img` into a `size x size` RGB float image, HWC (`[size*size*3]`). Padding is black before
 * normalization; with `normalize` each channel becomes `(v - mean) / std` (ImageNet).
 */
export function letterbox(img: Pixels, size: number, normalize = true): { pixels: Float32Array; transform: LetterboxTransform } {
  if (img.width <= 0 || img.height <= 0 || img.data.length < img.width * img.height * 4) {
    throw new Error(`expected RGBA pixels, got ${img.width}x${img.height} with ${img.data.length} bytes`);
  }
  const tf = letterboxGeometry([img.width, img.height], size);
  const [nw, nh] = tf.newSize;
  const content = resizeRgb(img, nw, nh);
  const out = new Float32Array(size * size * 3);
  for (let y = 0; y < nh; y++) {
    out.set(content.subarray(y * nw * 3, (y + 1) * nw * 3), ((y + tf.padY) * size + tf.padX) * 3);
  }
  if (normalize) {
    for (let i = 0; i < out.length; i += 3) {
      for (let c = 0; c < 3; c++) out[i + c] = (out[i + c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }
  return { pixels: out, transform: tf };
}

/** Block-average a frame to a `grid x grid` grayscale thumbnail (0-255 floats), row-major. */
export function thumb(img: Pixels, grid: number): { values: Float32Array; size: number } {
  const step = Math.max(1, Math.floor(Math.min(img.height, img.width) / (grid * 4)));
  const h = Math.ceil(img.height / step);
  const w = Math.ceil(img.width / step);
  const g = Math.max(1, Math.min(grid, h, w));
  const sums = new Float64Array(g * g);
  const counts = new Uint32Array(g * g);
  for (let sy = 0, y = 0; y < img.height; y += step, sy++) {
    const cy = Math.min(g - 1, Math.floor((sy * g) / h));
    for (let sx = 0, x = 0; x < img.width; x += step, sx++) {
      const cx = Math.min(g - 1, Math.floor((sx * g) / w));
      const p = (y * img.width + x) * 4;
      sums[cy * g + cx] += (img.data[p] + img.data[p + 1] + img.data[p + 2]) / 3;
      counts[cy * g + cx] += 1;
    }
  }
  const values = new Float32Array(g * g);
  for (let i = 0; i < values.length; i++) values[i] = counts[i] ? sums[i] / counts[i] : 0;
  return { values, size: g };
}

/**
 * Fraction of cells in a `grid x grid` downsampled grey view that changed by more than `pixelThreshold`
 * (0-255). A missing frame or different frame sizes count as a full change (1.0).
 */
export function changeRatio(a: Pixels | null, b: Pixels | null, grid = 64, pixelThreshold = 8): number {
  if (!a || !b) return 1;
  if (a.width !== b.width || a.height !== b.height) return 1;
  const ta = thumb(a, grid);
  const tb = thumb(b, grid);
  let changed = 0;
  for (let i = 0; i < ta.values.length; i++) if (Math.abs(ta.values[i] - tb.values[i]) > pixelThreshold) changed++;
  return changed / ta.values.length;
}

/** Same as `changeRatio` against a precomputed thumbnail (cheaper when one side is kept). */
export function changeRatioThumb(prev: { values: Float32Array; size: number } | null, next: { values: Float32Array; size: number }, pixelThreshold = 8): number {
  if (!prev || prev.size !== next.size) return 1;
  let changed = 0;
  for (let i = 0; i < next.values.length; i++) if (Math.abs(prev.values[i] - next.values[i]) > pixelThreshold) changed++;
  return changed / next.values.length;
}
