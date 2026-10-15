/**
 * Privacy filters applied before anything is captured, stored or shown (port of soma/capture/redaction.py).
 *
 * - `isExcluded` decides whether a shared window may be captured at all (password managers, private
 *   browsing, banking, auth prompts, the user's own list). Lives in `window.ts` next to `inferWindow`.
 * - `redactText` masks secrets in recognized text; `redactElements` / `redactObservation` apply it.
 * - `sensitiveBoxes` finds the boxes whose text held a secret.
 */
import type { OCRElement, ScreenObservation } from "../types";

const mask = (kind: string): string => `[REDACTED:${kind}]`;

/** Luhn checksum for a string of digits (credit-card validation). Needs at least 13 digits. */
export function luhnValid(digits: string): boolean {
  const ds = Array.from(digits).filter((c) => c >= "0" && c <= "9").map(Number);
  if (ds.length < 13) return false;
  let total = 0;
  ds.reverse().forEach((d, i) => {
    let v = d;
    if (i % 2 === 1) {
      v *= 2;
      if (v > 9) v -= 9;
    }
    total += v;
  });
  return total % 10 === 0;
}

const SECRET_KEYS =
  "(?:password|passwd|pwd|passphrase|secret|client[_-]?secret|token|auth[_-]?token|" +
  "access[_-]?token|api[_-]?key|apikey|access[_-]?key|private[_-]?key)";
const PAIR_RE = new RegExp(`(\\b${SECRET_KEYS}\\b["']?\\s*[:=]\\s*)("[^"\\n]*"|'[^'\\n]*'|[^\\s,;&]+)`, "gi");

/** (kind, pattern) applied in order after key/value pairs. */
const PATTERNS: [string, RegExp][] = [
  ["private-key", /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)/g],
  ["jwt", /\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}/g],
  ["api-key", /\bsk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{16,}/g],
  ["api-key", /\b(?:rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}/g],
  ["aws-key", /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g],
  ["github-token", /\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})/g],
  ["slack-token", /\bxox[abprs]-[A-Za-z0-9-]{10,}/g],
  ["google-key", /\bAIza[0-9A-Za-z_-]{35}\b/g],
  ["email", /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g],
];
const CARD_RE = /(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])/g;
const HEX_RE = /\b[0-9a-fA-F]{32,}\b/g;
const B64_RE = /(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{40,}={0,2}/g;

/** A long token containing upper, lower and digits is treated as a secret. */
const looksRandom = (tok: string): boolean => /\d/.test(tok) && /[A-Z]/.test(tok) && /[a-z]/.test(tok);

/**
 * Mask secrets: API keys, AWS/GitHub/Slack tokens, JWTs, emails, Luhn-valid card numbers, long
 * hex/base64 tokens and `password=...` pairs. Ordinary text is returned unchanged.
 */
export function redactText(text: string): string {
  if (!text) return text ?? "";
  let out = text.replace(PAIR_RE, (_m, key: string) => key + mask("secret"));
  for (const [kind, pat] of PATTERNS) out = out.replace(pat, mask(kind));
  out = out.replace(CARD_RE, (m) => (luhnValid(m) ? mask("card") : m));
  out = out.replace(HEX_RE, mask("hex"));
  out = out.replace(B64_RE, (m) => (looksRandom(m) ? mask("token") : m));
  return out;
}

/** True when `redactText` would change `text`. */
export const containsSecret = (text: string): boolean => redactText(text) !== (text ?? "");

/** Copies of `elements` with secrets in `text` masked (originals untouched). */
export const redactElements = (elements: readonly OCRElement[]): OCRElement[] =>
  elements.map((e) => ({ ...e, text: redactText(e.text) }));

/** A copy of `obs` with the window title and every element's text redacted. */
export function redactObservation(obs: ScreenObservation): ScreenObservation {
  return {
    ...obs,
    window: { ...obs.window, title: redactText(obs.window.title) },
    elements: redactElements(obs.elements),
  };
}

/** Normalized boxes of elements whose text contains (or already shows) a redacted secret. */
export const sensitiveBoxes = (elements: readonly OCRElement[]): [number, number, number, number][] =>
  elements.filter((e) => containsSecret(e.text) || e.text.includes("[REDACTED:")).map((e) => [...e.bbox]);
