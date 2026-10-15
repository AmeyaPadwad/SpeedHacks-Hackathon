/**
 * Deterministic hashed word-piece tokenizer for OCR text and memory summaries (port of
 * soma/models/tokenizer.py).
 *
 * No vocabulary file to ship or leak; ids are stable across runs and machines. The Python version used
 * blake2b; the web build uses 32-bit FNV-1a over UTF-8 bytes (ids differ from Python's, which is fine:
 * nothing trained in Python is loaded here). Id 0 is PAD, id 1 is UNK/empty.
 */

export const PAD_ID = 0;
export const UNK_ID = 1;

const TOKEN_RE = /[A-Za-z]+|\d+|[^\sA-Za-z\d]/g;
const encoder = new TextEncoder();

/** 32-bit FNV-1a of the UTF-8 bytes of `s` (unsigned). `seed` changes the offset basis. */
export function fnv1a(s: string, seed = 0x811c9dc5): number {
  let h = seed >>> 0;
  for (const b of encoder.encode(s)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Lower-cased word, number and punctuation tokens. */
export const tokenize = (text: string): string[] => (text ?? "").match(TOKEN_RE)?.map((t) => t.toLowerCase()) ?? [];

/** Id of one token in `[2, vocabSize)`. */
export const tokenId = (token: string, vocabSize: number): number => 2 + (fnv1a(token) % (vocabSize - 2));

/** Token ids of `text`, truncated to `maxLen` when given. */
export function encode(text: string, vocabSize: number, maxLen?: number): number[] {
  const ids = tokenize(text).map((t) => tokenId(t, vocabSize));
  return maxLen === undefined ? ids : ids.slice(0, Math.max(0, maxLen));
}

/** Stable bucket for categorical strings such as app names (case-insensitive). */
export const hashBucket = (name: string, nBuckets: number): number => fnv1a((name ?? "").toLowerCase()) % nBuckets;
