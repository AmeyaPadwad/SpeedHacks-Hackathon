/**
 * Target grounding and screen-version staleness checks (port of SOMA/soma/policy/grounding.py).
 *
 * A suggestion that points at a UI element is only valid while that element is still on screen. Before
 * presenting, the target is re-verified against the **latest** observation:
 * - the text must still match (normalized, tolerant of small OCR noise), and
 * - the box must overlap the original (IoU >= threshold) or sit nearby (shifted slightly).
 *
 * Coordinates are normalized `xyxy` in [0, 1]; `groundPoint` maps them to pixels for a given frame size.
 */
import type { Candidate, OCRElement, ScreenObservation, Suggestion } from "../types";

export type Box = readonly [number, number, number, number];

const CTRL_RE = /[\x00-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]/g;
const WS_RE = /\s+/g;
const TRIM_CHARS = " .,:;!?'\"`()[]{}";

export const TEXT_SIMILARITY = 0.9;
export const MAX_CENTER_SHIFT = 0.05;

/** Lowercase, NFKC, no control chars, collapsed whitespace, trimmed punctuation. */
export function normalizeText(text: string | null | undefined): string {
  const t = (text ?? "").normalize("NFKC").replace(WS_RE, " ").replace(CTRL_RE, "").toLowerCase().trim();
  let a = 0;
  let b = t.length;
  while (a < b && TRIM_CHARS.includes(t[a])) a++;
  while (b > a && TRIM_CHARS.includes(t[b - 1])) b--;
  return t.slice(a, b);
}

/** Longest common block of a[alo:ahi], b[blo:bhi] (earliest in a, then in b, on ties) like difflib. */
function longestMatch(a: string, b: string, alo: number, ahi: number, blo: number, bhi: number): [number, number, number] {
  let besti = alo;
  let bestj = blo;
  let bestsize = 0;
  let prev = new Map<number, number>();
  for (let i = alo; i < ahi; i++) {
    const cur = new Map<number, number>();
    for (let j = blo; j < bhi; j++) {
      if (a[i] !== b[j]) continue;
      const k = (prev.get(j - 1) ?? 0) + 1;
      cur.set(j, k);
      if (k > bestsize) {
        besti = i - k + 1;
        bestj = j - k + 1;
        bestsize = k;
      }
    }
    prev = cur;
  }
  return [besti, bestj, bestsize];
}

/**
 * Ratcliff/Obershelp similarity, `difflib.SequenceMatcher(None, a, b).ratio()` without the autojunk
 * heuristic (which only applies to strings of 200+ characters).
 */
export function similarityRatio(a: string, b: string): number {
  const total = a.length + b.length;
  if (total === 0) return 1;
  let matches = 0;
  const queue: [number, number, number, number][] = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop()!;
    const [i, j, k] = longestMatch(a, b, alo, ahi, blo, bhi);
    if (k === 0) continue;
    matches += k;
    if (alo < i && blo < j) queue.push([alo, i, blo, j]);
    if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
  }
  return (2 * matches) / total;
}

/** Normalized equality, or a near-identical string (OCR jitter). */
export function textMatches(a: string, b: string, threshold = TEXT_SIMILARITY): boolean {
  const na = normalizeText(a);
  const nb = normalizeText(b);
  if (!na || !nb) return false;
  if (na === nb) return true;
  return similarityRatio(na, nb) >= threshold;
}

/** Intersection-over-union of two `xyxy` boxes (0 for degenerate boxes). */
export function iou(a: Box, b: Box): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

export function center(box: Box): [number, number] {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

export function centerDistance(a: Box, b: Box): number {
  const [ax, ay] = center(a);
  const [bx, by] = center(b);
  return Math.hypot(ax - bx, ay - by);
}

/** Best element on `latestObs` matching `target`, or null. */
export function findTarget(target: OCRElement, latestObs: ScreenObservation, iouThreshold = 0.5,
  maxCenterShift = MAX_CENTER_SHIFT): OCRElement | null {
  if (target.windowId && latestObs.window.windowId && target.windowId !== latestObs.window.windowId) return null;
  let best: OCRElement | null = null;
  let bestScore = -1;
  for (const el of latestObs.elements) {
    if (!textMatches(target.text, el.text)) continue;
    const overlap = iou(target.bbox, el.bbox);
    const shift = centerDistance(target.bbox, el.bbox);
    if (overlap < iouThreshold && shift > maxCenterShift) continue;
    const score = overlap - shift;
    if (score > bestScore) {
      best = el;
      bestScore = score;
    }
  }
  return best;
}

/** True if the candidate has no UI target, or its target is still on screen. */
export function verifyTarget(candidate: Candidate, latestObs: ScreenObservation, iouThreshold = 0.5,
  maxCenterShift = MAX_CENTER_SHIFT): boolean {
  if (candidate.target === null) return true;
  return findTarget(candidate.target, latestObs, iouThreshold, maxCenterShift) !== null;
}

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);
const boxOf = (t: OCRElement | Box): Box => ("bbox" in t ? t.bbox : t) as Box;

/** Pixel center `[x, y]` of a normalized target for a frame of `[width, height]`. */
export function groundPoint(target: OCRElement | Box, size: readonly [number, number]): [number, number] {
  const w = Math.trunc(size[0]);
  const h = Math.trunc(size[1]);
  if (w <= 0 || h <= 0) throw new RangeError(`invalid frame size ${size}`);
  const [cx, cy] = center(boxOf(target));
  return [Math.round(clamp01(cx) * (w - 1)), Math.round(clamp01(cy) * (h - 1))];
}

/** Pixel `xyxy` box of a normalized target, clamped to the frame. */
export function groundBox(target: OCRElement | Box, size: readonly [number, number]): [number, number, number, number] {
  const b = boxOf(target);
  const w = Math.trunc(size[0]);
  const h = Math.trunc(size[1]);
  return [Math.round(clamp01(b[0]) * w), Math.round(clamp01(b[1]) * h),
    Math.round(clamp01(b[2]) * w), Math.round(clamp01(b[3]) * h)];
}

export type ChangedFn = (suggestionVersion: number, currentVersion: number) => boolean;

/**
 * Whether a suggestion generated on an older screen version is no longer valid.
 *
 * @param changed whether the UI changed relevantly since the suggestion's version. `null` (default) is
 *   conservative: any newer version is stale. A function gets `(suggestionVersion, currentVersion)`.
 * @param latestObs if given and the suggestion has a UI target, the target must still verify on it.
 */
export function isStale(suggestion: Suggestion, currentVersion: number, changed: boolean | ChangedFn | null = null,
  latestObs: ScreenObservation | null = null): boolean {
  if (suggestion.screenVersion > currentVersion) return true; // from the future: never trust it
  if (suggestion.screenVersion < currentVersion) {
    if (changed === null) return true;
    const relevant = typeof changed === "function" ? changed(suggestion.screenVersion, currentVersion) : changed;
    if (relevant) return true;
  }
  if (latestObs !== null && suggestion.candidate !== null) return !verifyTarget(suggestion.candidate, latestObs);
  return false;
}
