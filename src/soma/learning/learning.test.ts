import * as tf from "@tensorflow/tfjs";
import { Rng } from "../models/math";
import { memoryIO } from "../models/weightsIO";
import { ActionType, FeedbackOutcome, makeCandidate } from "../types";
import { exampleFromFeedback, FeedbackBook, type FeedbackSnapshot, type LabeledExample } from "./feedback";
import { OnlineRanker } from "./onlineRanker";
import { applyBias, OnlinePreferenceRanker } from "./preferenceRanker";
import { ReplayBuffer } from "./replayBuffer";

beforeAll(async () => {
  await tf.setBackend("cpu");
});

const D = 16;
const vec = (seed: number): Float32Array => {
  const r = new Rng(seed);
  return Float32Array.from({ length: D }, () => r.normal());
};

describe("OnlineRanker", () => {
  it("has zero bias without evidence", () => {
    const r = new OnlineRanker(D);
    const b = r.bias(vec(1), [vec(2), vec(3)]);
    expect([...b]).toEqual([0, 0, 0]);
    expect([...r.bias(vec(1), [])]).toEqual([0]);
    r.dispose();
  });

  it("learns to prefer an accepted candidate over a rejected one", async () => {
    const r = new OnlineRanker(D);
    const ctx = vec(1);
    const [good, bad] = [vec(2), vec(3)];
    for (let i = 0; i < 5; i++) {
      await r.update(ctx, good, FeedbackOutcome.ACCEPTED, { app: "code" });
      await r.update(ctx, bad, FeedbackOutcome.REJECTED, { app: "code" });
    }
    const b = r.bias(ctx, [good, bad], "code");
    expect(b[1]).toBeGreaterThan(0);
    expect(b[2]).toBeLessThan(0);
    expect(r.updates).toBe(10);
    r.dispose();
  });

  it("moves the abstain bias per app, like Python's abstain arm", async () => {
    const r = new OnlineRanker(D);
    for (let i = 0; i < 4; i++) await r.update(vec(1), vec(2), FeedbackOutcome.REJECTED, { app: "Code" });
    expect(r.bias(vec(1), [], "code")[0]).toBeGreaterThan(0);
    expect(r.bias(vec(1), [], "chrome")[0]).toBe(0);
    const acc = new OnlineRanker(D);
    for (let i = 0; i < 4; i++) await acc.update(vec(1), vec(2), FeedbackOutcome.ACCEPTED);
    expect(acc.bias(vec(1), [])[0]).toBeLessThan(0);
    r.dispose();
    acc.dispose();
  });

  it("does not train on ignored outcomes", async () => {
    const r = new OnlineRanker(D);
    expect(await r.update(vec(1), vec(2), FeedbackOutcome.IGNORED)).toBe(false);
    expect(r.updates).toBe(0);
    expect([...r.bias(vec(1), [vec(2)])]).toEqual([0, 0]);
    r.dispose();
  });

  it("edits count as a softer positive", async () => {
    const [acc, ed] = [new OnlineRanker(D), new OnlineRanker(D)];
    for (let i = 0; i < 3; i++) {
      await acc.update(vec(1), vec(2), FeedbackOutcome.ACCEPTED);
      await ed.update(vec(1), vec(2), FeedbackOutcome.EDITED);
    }
    const [a, e] = [acc.bias(vec(1), [vec(2)])[1], ed.bias(vec(1), [vec(2)])[1]];
    expect(e).toBeGreaterThan(0);
    expect(e).toBeLessThan(a);
    acc.dispose();
    ed.dispose();
  });

  it("stays bounded by maxBias and leaks no tensors", async () => {
    const r = new OnlineRanker(D, { maxBias: 1.5, lr: 5 });
    await r.update(vec(1), vec(2), FeedbackOutcome.ACCEPTED); // warm up the optimizer
    const before = tf.memory().numTensors;
    for (let i = 0; i < 30; i++) await r.update(vec(1), vec(2), FeedbackOutcome.REJECTED);
    const b = r.bias(vec(1), [vec(2), vec(4)]);
    expect(tf.memory().numTensors).toBe(before);
    expect(Math.max(...b.map(Math.abs))).toBeLessThanOrEqual(1.5 + 1e-6);
    r.reset();
    expect([...r.bias(vec(1), [vec(2)])]).toEqual([0, 0]);
    r.dispose();
  });

  it("round-trips through an IOHandler and indexeddb, and refuses another dim", async () => {
    const r = new OnlineRanker(D);
    for (let i = 0; i < 3; i++) await r.update(vec(1), vec(2), FeedbackOutcome.ACCEPTED, { app: "code" });
    const io = memoryIO();
    await r.save(io);
    const other = new OnlineRanker(D);
    expect(await other.load(memoryIO())).toBe(false);
    expect(await other.load(io)).toBe(true);
    expect([...other.bias(vec(1), [vec(2)], "code")]).toEqual([...r.bias(vec(1), [vec(2)], "code")]);
    expect(other.updates).toBe(3);
    const wrong = new OnlineRanker(D * 2);
    expect(await wrong.load(io)).toBe(false);
    expect(await wrong.load("indexeddb://soma-ranker-missing")).toBe(false);
    await r.save("indexeddb://soma-ranker-test");
    const third = new OnlineRanker(D);
    expect(await third.load("indexeddb://soma-ranker-test")).toBe(true);
    expect(third.bias(vec(1), [vec(2)], "code")[1]).toBeCloseTo(r.bias(vec(1), [vec(2)], "code")[1], 6);
    [r, other, wrong, third].forEach((x) => x.dispose());
  });
});

describe("OnlinePreferenceRanker (arm port)", () => {
  const cands = () => [
    makeCandidate({ action: ActionType.EXPLAIN_ERROR, text: "Explain this error" }),
    makeCandidate({ action: ActionType.SUGGEST_NEXT_STEP, text: "Run the tests" }),
    null,
  ];

  it("has no bias without evidence", () => {
    const b = new OnlinePreferenceRanker().bias(cands(), "code.exe", 4);
    expect(b.length).toBe(5);
    expect(b.every((x) => x === 0)).toBe(true);
  });

  it("rejections raise abstain and lower the candidate, per app", () => {
    const r = new OnlinePreferenceRanker();
    const c = cands();
    for (let i = 0; i < 5; i++) r.updateCandidate(c[0]!, "code.exe", FeedbackOutcome.REJECTED);
    const b = r.bias(c, "code.exe");
    expect(b[0]).toBeGreaterThan(0);
    expect(b[1]).toBeLessThan(0);
    expect(b[2]).toBe(0);
    expect(b[3]).toBe(0);
    expect(r.bias(c, "chrome.exe").every((x) => x === 0)).toBe(true);
  });

  it("accepts raise the candidate; edits less so", () => {
    const r = new OnlinePreferenceRanker();
    const e = new OnlinePreferenceRanker();
    const c = cands();
    for (let i = 0; i < 4; i++) {
      r.updateCandidate(c[1]!, "code.exe", FeedbackOutcome.ACCEPTED);
      e.updateCandidate(c[1]!, "code.exe", FeedbackOutcome.EDITED);
    }
    const b = r.bias(c, "code.exe");
    expect(b[2]).toBeGreaterThan(0);
    expect(b[0]).toBeLessThan(0);
    const eb = e.bias(c, "code.exe")[2];
    expect(eb).toBeGreaterThan(0);
    expect(eb).toBeLessThan(b[2]);
  });

  it("ignores IGNORED, bounds and decays", () => {
    const r = new OnlinePreferenceRanker({ maxBias: 1.5, decay: 0.5 });
    expect(r.update("code.exe", ActionType.EXPLAIN_ERROR, FeedbackOutcome.IGNORED)).toBe(false);
    expect(r.updates).toBe(0);
    for (let i = 0; i < 200; i++) r.update("code.exe", ActionType.EXPLAIN_ERROR, FeedbackOutcome.REJECTED);
    expect(Math.max(...r.bias(cands(), "code.exe").map(Math.abs))).toBeLessThanOrEqual(1.5 + 1e-6);
    const before = r.abstainBias("code.exe");
    for (let i = 0; i < 10; i++) r.update("chrome.exe", ActionType.SUGGEST_NEXT_STEP, FeedbackOutcome.ACCEPTED);
    expect(Math.abs(r.abstainBias("code.exe"))).toBeLessThan(before);
    expect(() => new OnlinePreferenceRanker({ decay: 0 })).toThrow();
  });

  it("applyBias leaves blocked entries untouched", () => {
    const out = applyBias(Float32Array.from([0, 1, -1e4, -Infinity]), Float32Array.from([0.5, -0.5, 2, 2]));
    expect([...out]).toEqual([0.5, 0.5, -1e4, -Infinity]);
  });

  it("state round-trips and resets per app", () => {
    const r = new OnlinePreferenceRanker();
    const c = cands();
    r.updateCandidate(c[0]!, "code.exe", FeedbackOutcome.REJECTED);
    r.updateCandidate(c[1]!, "chrome.exe", FeedbackOutcome.ACCEPTED);
    const other = new OnlinePreferenceRanker({ prior: 5 });
    other.loadStateDict(JSON.parse(JSON.stringify(r.stateDict())));
    expect([...other.bias(c, "code.exe")]).toEqual([...r.bias(c, "code.exe")]);
    r.reset("code.exe");
    expect(r.bias(c, "code.exe").every((x) => x === 0)).toBe(true);
    expect(r.bias(c, "chrome.exe").some((x) => x !== 0)).toBe(true);
    const t = r.biasForTypes([ActionType.SUGGEST_NEXT_STEP, null], "chrome.exe");
    expect(t.length).toBe(3);
    expect(t[1]).toBeGreaterThan(0);
    expect(t[2]).toBe(0);
  });
});

const snap = (over: Partial<FeedbackSnapshot> = {}): FeedbackSnapshot => ({
  app: "code", action: ActionType.EXPLAIN_ERROR, candidateKey: "k", shownIndex: 2,
  context: vec(1), candidateEmbedding: vec(2), ...over,
});

describe("feedback", () => {
  it("converts outcomes to labels; ignored is ambiguous, never a negative", () => {
    const acc = exampleFromFeedback(snap(), FeedbackOutcome.ACCEPTED, { consented: true });
    const ed = exampleFromFeedback(snap(), FeedbackOutcome.EDITED, { consented: true });
    const rej = exampleFromFeedback(snap(), FeedbackOutcome.REJECTED, { consented: true });
    const ign = exampleFromFeedback(snap(), FeedbackOutcome.IGNORED, { consented: true });
    expect([acc.target, acc.weight]).toEqual([2, 1]);
    expect([ed.target, ed.weight]).toEqual([2, 0.5]);
    expect([rej.target, rej.weight, rej.rejectedIndex]).toEqual([0, 0.5, 2]);
    expect([ign.ambiguous, ign.weight, ign.rejectedIndex, ign.usableForTraining]).toEqual([true, 0, null, false]);
    expect(exampleFromFeedback(snap(), FeedbackOutcome.ACCEPTED, { consented: false }).usableForTraining).toBe(false);
    expect(new ReplayBuffer<LabeledExample>(10).add(ign)).toBe(false);
  });

  it("FeedbackBook keeps copies without pixels, is single-use and bounded", () => {
    const book = new FeedbackBook(2);
    const s = snap() as FeedbackSnapshot & { pixels?: Float32Array };
    s.pixels = new Float32Array(12);
    book.remember("a", s);
    s.context[0] = 99; // stored a copy
    expect(book.peek("a")!.context[0]).not.toBe(99);
    expect("pixels" in book.peek("a")!).toBe(false);
    book.remember("b", snap());
    book.remember("c", snap());
    expect(book.size).toBe(2);
    expect(book.take("a")).toBeNull(); // evicted (oldest)
    expect(book.take("b")).not.toBeNull();
    expect(book.take("b")).toBeNull(); // single use
    book.clear();
    expect(book.size).toBe(0);
  });
});

describe("ReplayBuffer", () => {
  const ex = (app: string, task: string, n: number, consented = true): LabeledExample[] =>
    Array.from({ length: n }, (_, i) =>
      exampleFromFeedback(snap({ app, shownIndex: 1 + (i % 3) }), FeedbackOutcome.ACCEPTED, { consented, task, feedbackId: `${app}${i}` }));

  it("bounds and stratifies", () => {
    const buf = new ReplayBuffer<LabeledExample>(30, { perStratumCapacity: 20, seed: 0 });
    buf.extend(ex("code", "error", 100));
    buf.extend(ex("winword", "draft", 5));
    expect(buf.size).toBeLessThanOrEqual(30);
    const strata = Object.fromEntries(buf.strata().map((s) => [s.app, s.count]));
    expect(strata.winword).toBe(5);
    expect(strata.code).toBeLessThanOrEqual(20);
    expect(buf.add(ex("x", "y", 1, false)[0])).toBe(false);
    const sample = buf.sample(20, new Rng(0), 0);
    const word = sample.filter((e) => e.app === "winword");
    expect(sample.length).toBe(20);
    expect(word.length).toBe(10);
    expect(new Set(word).size).toBe(5);
  });

  it("mixes anchors in deterministically", () => {
    const build = () => {
      const b = new ReplayBuffer<LabeledExample>(50, { anchorCapacity: 10, anchorFraction: 0.25, seed: 4 });
      b.extend(ex("code", "error", 40));
      for (const a of ex("anchor", "anchor", 30)) b.addAnchor({ ...a, source: "anchor" });
      return b;
    };
    const [b1, b2] = [build(), build()];
    expect(b1.nAnchors).toBe(10);
    const s1 = b1.sample(16, new Rng(1)).map((e) => e.feedbackId);
    const s2 = b2.sample(16, new Rng(1)).map((e) => e.feedbackId);
    expect(s1).toEqual(s2);
    expect(b1.sample(16, new Rng(1)).filter((e) => e.source === "anchor").length).toBe(4);
    const only = new ReplayBuffer<LabeledExample>(5);
    only.addAnchor(ex("a", "b", 1)[0]);
    expect(only.sample(3).length).toBe(3);
    b1.clear();
    expect(b1.size).toBe(0);
    expect(b1.nAnchors).toBe(10);
  });
});
