/**
 * Screen parser: OCR elements -> `ScreenObservation` (port of soma/perception/screen_parser.py).
 *
 * OCR text is not assumed to be clickable. A web page has no accessibility tree for other apps and the
 * parser gets no pixels, so roles come from the element (when it already has a specific one) or from
 * text heuristics: error lines (`Traceback`, `...Error:`, `FAILED`), file references
 * (`File "x.py", line 12` / `x.py:12`), URLs (link) and short button words; otherwise `"text"`.
 * Also: geometry helpers and `findErrors` / `findFileRefs` / `findButtons`.
 */
import type { OCRElement, ScreenObservation, WindowInfo } from "../types";
import type { Box } from "./preprocess";

// ----------------------------------------------------------------- geometry

/** Normalized xyxy -> integer pixel xyxy for an image of `(w, h)`. */
export function toPixels(b: Box, size: [number, number]): Box {
  const [w, h] = size;
  return [Math.round(b[0] * w), Math.round(b[1] * h), Math.round(b[2] * w), Math.round(b[3] * h)];
}

/** Pixel xyxy -> normalized xyxy (clipped to [0, 1]) for `(w, h)`. */
export function toNormalized(b: Box, size: [number, number]): Box {
  const w = Math.max(1, size[0]);
  const h = Math.max(1, size[1]);
  const c = (v: number) => Math.min(1, Math.max(0, v));
  return [c(b[0] / w), c(b[1] / h), c(b[2] / w), c(b[3] / h)];
}

export const boxArea = (b: Box): number => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);

export function intersection(a: Box, b: Box): number {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  return Math.max(0, w) * Math.max(0, h);
}

/** Intersection over union of two xyxy boxes. */
export function iou(a: Box, b: Box): number {
  const inter = intersection(a, b);
  const union = boxArea(a) + boxArea(b) - inter;
  return union > 0 ? inter / union : 0;
}

// ----------------------------------------------------------------- text heuristics

const ERROR_RES: RegExp[] = [
  /^\s*Traceback\b/,
  /^\s*[A-Za-z_][\w.]*(?:Error|Exception|Warning)\s*(?::|$)/,
  /\b(?:error|fatal|panic)\s*(?:\[\w+\]|[A-Z]+\d+)?\s*:/i,
  /\bFAILED\b|^\s*FAIL\b|\bERROR\b/,
  /\b\d+\s*failed\b/i,
  /\bUncaught\s+\w*(?:Error|Exception)/,
  /^\s*Exception\b/,
];
const PY_FILE_RE = /File\s*"(?<path>[^"]+)"\s*,\s*line\s*(?<line>\d+)/g;
const EXTS = "py|pyi|ipynb|js|jsx|ts|tsx|mjs|go|rs|java|kt|c|cc|cpp|h|hpp|cs|rb|php|swift|" +
  "md|json|ya?ml|toml|txt|cfg|ini|html|css|sql|sh|ps1";
const PATH_RE = new RegExp(
  `(?<![\\w/\\\\.-])(?<path>(?:[A-Za-z]:[\\\\/])?[\\w.\\\\/-]*\\w\\.(?:${EXTS}))\\b` +
  "(?:(?:::[\\w\\[\\]-]+)|(?::(?<line>\\d+))(?::(?<col>\\d+))?|\\((?<pline>\\d+)(?:,(?<pcol>\\d+))?\\))?", "g");
const URL_RE = /\bhttps?:\/\/\S+|\bwww\.\S+\.\w+/g;
export const BUTTON_WORDS: ReadonlySet<string> = new Set([
  "ok", "cancel", "apply", "save", "run", "submit", "close", "yes", "no", "copy",
  "install", "run tests", "debug", "next", "back", "open", "continue", "retry", "delete",
  "accept", "reject", "done", "allow", "deny", "send", "search", "commit", "push", "sign out",
]);

/** Heuristic: does this line look like an error / traceback / test failure? */
export const looksLikeError = (text: string): boolean => ERROR_RES.some((r) => r.test(text ?? ""));

/** A file reference seen on screen (never opened by the parser). Untrusted text. */
export interface FileRef {
  path: string;
  line: number | null;
  column: number | null;
  element: OCRElement | null;
}

/** File references in one text line (Python tracebacks, `x.py:12:5`, `x.cs(12,5)`); URLs are ignored. */
export function extractFileRefs(text: string): FileRef[] {
  const t = text ?? "";
  const refs: FileRef[] = [];
  const spans: [number, number][] = [];
  for (const m of t.matchAll(URL_RE)) spans.push([m.index!, m.index! + m[0].length]);
  for (const m of t.matchAll(PY_FILE_RE)) {
    refs.push({ path: m.groups!.path.trim(), line: Number(m.groups!.line), column: null, element: null });
    spans.push([m.index!, m.index! + m[0].length]);
  }
  for (const m of t.matchAll(PATH_RE)) {
    const start = m.index!;
    if (spans.some(([s, e]) => s <= start && start < e)) continue;
    const g = m.groups!;
    const line = g.line ?? g.pline;
    const col = g.col ?? g.pcol;
    refs.push({ path: g.path, line: line ? Number(line) : null, column: col ? Number(col) : null, element: null });
  }
  return refs;
}

function isFileRefLine(text: string): boolean {
  PY_FILE_RE.lastIndex = 0;
  if (PY_FILE_RE.test(text)) return true;
  return extractFileRefs(text).some((r) => r.line !== null);
}

/** Heuristic `uiRole` for an element with no reliable role. */
export function inferRole(element: Pick<OCRElement, "text">): string {
  const text = element.text.trim();
  if (looksLikeError(text)) return "error";
  if (isFileRefLine(text)) return "file_ref";
  URL_RE.lastIndex = 0;
  if (URL_RE.test(text)) return "link";
  const words = text.split(/\s+/).filter(Boolean);
  const short = text.length > 0 && text.length <= 24 && words.length <= 3 && /\p{L}/u.test(text);
  const first = text[0] ?? "";
  if (short && first !== first.toLowerCase() && BUTTON_WORDS.has(text.toLowerCase().replace(/[.…]+$/, ""))) {
    return "button";
  }
  return "text";
}

/** Roles that yield to the heuristics. */
export const GENERIC_ROLES: ReadonlySet<string> = new Set(["", "text", "document", "pane", "group", "custom", "window", "edit", "textbox"]);

/** Sort top-to-bottom, then left-to-right (rows grouped by vertical centre). */
export function readingOrder(elements: readonly OCRElement[]): OCRElement[] {
  if (!elements.length) return [];
  const heights = elements.map((e) => Math.max(1e-6, e.bbox[3] - e.bbox[1])).sort((a, b) => a - b);
  const tol = heights[Math.floor(heights.length / 2)] * 0.5;
  const row = (e: OCRElement) => (tol > 0 ? Math.trunc((e.bbox[1] + e.bbox[3]) / 2 / tol) : 0);
  return elements
    .map((e, i) => ({ e, i, r: row(e) }))
    .sort((a, b) => a.r - b.r || a.e.bbox[0] - b.e.bbox[0] || a.i - b.i)
    .map((x) => x.e);
}

/**
 * Turn OCR elements into a `ScreenObservation`: empty elements are dropped, generic roles are replaced
 * by heuristics, and elements come back in reading order. Text is not redacted here (OCR engines
 * already redact; the runtime also calls `redactObservation`).
 */
export function parseScreen(window: WindowInfo, elements: readonly OCRElement[], version: number, timestampMs: number = Date.now()): ScreenObservation {
  const out: OCRElement[] = [];
  for (const e of elements) {
    if (!e.text.trim()) continue;
    const base = e.uiRole;
    let role = base;
    if (base === null || GENERIC_ROLES.has(base)) {
      const heuristic = inferRole(e);
      role = heuristic !== "text" || !base ? heuristic : base;
    }
    out.push({ ...e, uiRole: role });
  }
  return { window, elements: readingOrder(out), version, timestampMs };
}

/** Elements that are error lines (by role or by text), in reading order. */
export const findErrors = (obs: ScreenObservation): OCRElement[] =>
  obs.elements.filter((e) => e.uiRole === "error" || looksLikeError(e.text));

/** Unique file references (path, line) visible on screen, in reading order. */
export function findFileRefs(obs: ScreenObservation): FileRef[] {
  const seen = new Set<string>();
  const out: FileRef[] = [];
  for (const e of obs.elements) {
    for (const r of extractFileRefs(e.text)) {
      const k = `${r.path}\u0000${r.line ?? ""}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ ...r, element: e });
    }
  }
  return out;
}

/** Elements with a button-like role. */
export const findButtons = (obs: ScreenObservation): OCRElement[] =>
  obs.elements.filter((e) => ["button", "link", "menuitem", "tab"].includes(e.uiRole ?? ""));
