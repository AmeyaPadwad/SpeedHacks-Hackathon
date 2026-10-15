import { describe, expect, it } from "vitest";
import { ActionType, makeCandidate, SourceType } from "../types";
import { actionIndex, CandidateEncoder, type TextEmbedderLike } from "./candidateEncoder";
import { generateCandidates } from "./candidateGenerator";
import { mem, obsOf } from "./testUtil";

const D = 32;
/** Bag-of-characters stand-in for the memory TextEmbedder (unit length). */
const fakeEmbedder: TextEmbedderLike = {
  embed(text: string) {
    const v = new Float32Array(D);
    for (const ch of text) v[ch.charCodeAt(0) % D] += 1;
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  },
};
const TRACEBACK = ["Traceback (most recent call last):", '  File "src/app/calc.py", line 40, in divide',
  "ZeroDivisionError: division by zero"];
const sample = () => generateCandidates(obsOf(TRACEBACK), [mem("doc1", SourceType.DOCUMENT, "Q3 report draft")], 8,
  (p) => p.startsWith("src/"));
const enc = () => new CandidateEncoder(D, fakeEmbedder);

describe("CandidateEncoder", () => {
  it("shapes, types and mask", () => {
    const cands = sample().slice(0, 3);
    const { embeddings, types, mask } = enc().encode(cands, null, 5);
    expect(embeddings).toHaveLength(5);
    expect(embeddings.every((e) => e.length === D)).toBe(true);
    expect(Array.from(mask)).toEqual([0, 0, 0, 1, 1]);
    expect(types.slice(0, 3)).toEqual(cands.map(actionIndex));
    expect(types.slice(3)).toEqual([0, 0]);
    expect(embeddings.slice(3).every((e) => e.every((x) => x === 0))).toBe(true);
    expect(embeddings.flatMap((e) => Array.from(e)).every(Number.isFinite)).toBe(true);
    expect(types.slice(0, 3).every((t) => t >= 1)).toBe(true);
  });

  it("truncates, handles empty, blocks and rejects a bad dim", () => {
    const e = enc();
    const t = e.encode(sample(), null, 2);
    expect(t.embeddings).toHaveLength(2);
    expect(Array.from(t.mask)).toEqual([0, 0]);
    const empty = e.encode([], null, 3);
    expect(Array.from(empty.mask)).toEqual([1, 1, 1]);
    expect(Array.from(e.encode(sample().slice(0, 2), null, 3, D, [true, false]).mask)).toEqual([1, 0, 1]);
    expect(() => e.encode(sample(), null, 3, D + 1)).toThrow();
  });

  it("a memory embedding changes the output", () => {
    const c = makeCandidate({ action: ActionType.COMPARE_WITH_PRIOR_WORK, text: "Compare with earlier doc", memoryRef: "m1" });
    const e = enc();
    const a = e.encode([c]).embeddings[0];
    const b = e.encode([c], new Map([["m1", new Float32Array(50).fill(2)]])).embeddings[0];
    const r = e.encode([c], new Map([["m1", mem("m1", SourceType.DOCUMENT, "x")]])).embeddings[0];
    expect(Array.from(a)).not.toEqual(Array.from(b));
    expect(Array.from(a)).not.toEqual(Array.from(r));
  });

  it("is deterministic per seed and LayerNorm'd", () => {
    const a = enc().encode(sample(), null, 8).embeddings;
    const b = enc().encode(sample(), null, 8).embeddings;
    expect(a.map((x) => Array.from(x))).toEqual(b.map((x) => Array.from(x)));
    const v = a[0];
    const mean = v.reduce((s, x) => s + x, 0) / D;
    expect(Math.abs(mean)).toBeLessThan(1e-4);
    const other = new CandidateEncoder(D, fakeEmbedder, undefined, 7).encode(sample(), null, 8).embeddings[0];
    expect(Array.from(other)).not.toEqual(Array.from(v));
  });
});
