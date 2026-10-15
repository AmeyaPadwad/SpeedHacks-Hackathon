import * as tf from "@tensorflow/tfjs";
import { DEFAULT_CONFIG, type ModelConfig } from "../config";
import { randomInputs } from "./batch";
import { ScreenHistory } from "./history";
import { softmax } from "./math";
import { BLOCKED_LOGIT, NO_SUGGESTION_INDEX, SomaModel } from "./somaModel";
import { memoryIO } from "./weightsIO";

const small = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  ...DEFAULT_CONFIG.model,
  dModel: 32, nHeads: 4, imageSize: 64, patchSize: 16, visionLayers: 1, vocabSize: 512,
  maxOcrTokens: 16, textLayers: 1, fusionLayers: 1, historyLen: 4, nApps: 16, maxMemories: 4,
  ...over,
});

beforeAll(async () => {
  await tf.setBackend("cpu");
});

const close = (a: Float32Array, b: Float32Array, tol = 1e-5) =>
  a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) <= tol);

describe.each(["gru", "transformer"] as const)("SomaModel (%s)", (kind) => {
  let model: SomaModel;
  const cfg = small({ temporalKind: kind });
  beforeAll(() => {
    model = new SomaModel(cfg, 0);
  });
  afterAll(() => model.dispose());

  it("returns well-formed shapes", () => {
    const inp = randomInputs(cfg, { seed: 1, A: 3, K: 2 });
    const hist = [new Float32Array(32).fill(0.1), new Float32Array(32).fill(-0.2)];
    const out = model.step(inp, hist);
    expect(out.screenState.length).toBe(32);
    expect(out.temporalState.length).toBe(32);
    expect(out.logits.length).toBe(4);
    expect(out.memoryAttention.length).toBe(3);
    expect(out.memoryGate).toBeGreaterThan(0);
    expect(out.memoryGate).toBeLessThan(1);
    expect([...out.logits].every(Number.isFinite)).toBe(true);
    expect(softmax(out.memoryAttention).length).toBe(3);
    const sum = out.memoryAttention.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("encodeScreen matches step's screen state and score matches step (single encode)", () => {
    const inp = randomInputs(cfg, { seed: 2 });
    const s = model.encodeScreen(inp);
    const out = model.step(inp, []);
    expect(close(s, out.screenState)).toBe(true);
    const scored = model.score(s, inp, []);
    expect(close(scored.logits, out.logits)).toBe(true);
  });

  it("puts all mass on NO_SUGGESTION with zero candidates", () => {
    const inp = randomInputs(cfg, { seed: 3, A: 0, K: 0, emptyOcr: true });
    const out = model.step(inp, []);
    expect(NO_SUGGESTION_INDEX).toBe(0);
    expect(out.logits.length).toBe(1);
    expect(softmax(out.logits)[0]).toBe(1);
    expect(out.memoryAttention.length).toBe(1);
    expect(out.memoryAttention[0]).toBeCloseTo(1, 6);
  });

  it("gives blocked candidates BLOCKED_LOGIT and ~0 probability", () => {
    const inp = randomInputs(cfg, { seed: 4, A: 3, allBlocked: true });
    const out = model.step(inp, []);
    expect([...out.logits.subarray(1)]).toEqual([BLOCKED_LOGIT, BLOCKED_LOGIT, BLOCKED_LOGIT]);
    expect(softmax(out.logits)[0]).toBeCloseTo(1, 6);
    inp.candidateMask = Uint8Array.from([0, 1, 0]);
    const part = model.step(inp, []);
    expect(part.logits[2]).toBe(BLOCKED_LOGIT);
    expect(part.logits[1]).not.toBe(BLOCKED_LOGIT);
  });

  it("ignores padded OCR tokens", () => {
    const inp = randomInputs(cfg, { seed: 5, L: 8 }); // second half padded
    const a = model.encodeScreen(inp);
    const changed = { ...inp, ocrIds: Int32Array.from(inp.ocrIds), ocrBoxes: Float32Array.from(inp.ocrBoxes) };
    for (let i = 4; i < 8; i++) {
      changed.ocrIds[i] = 7;
      changed.ocrBoxes.set([0.5, 0.5, 0.9, 0.9], i * 4);
    }
    expect(close(a, model.encodeScreen(changed))).toBe(true);
    // trimming the padding entirely gives the same state
    const trimmed = {
      ...inp, ocrIds: inp.ocrIds.slice(0, 4), ocrBoxes: inp.ocrBoxes.slice(0, 16), ocrPadding: inp.ocrPadding.slice(0, 4),
    };
    expect(close(a, model.encodeScreen(trimmed), 1e-4)).toBe(true);
    // but an unpadded token does change it
    const real = { ...changed, ocrPadding: new Uint8Array(8) };
    expect(close(a, model.encodeScreen(real))).toBe(false);
  });

  it("handles empty OCR and a smaller image", () => {
    const empty = randomInputs(cfg, { seed: 6, L: 0, imageSize: 32 });
    const s = model.encodeScreen(empty);
    expect(s.length).toBe(32);
    expect([...s].every(Number.isFinite)).toBe(true);
    expect(() => model.encodeScreen(randomInputs(cfg, { imageSize: 128 }))).toThrow();
  });

  it("uses history causally and truncates it to historyLen", () => {
    const inp = randomInputs(cfg, { seed: 7 });
    const h = Array.from({ length: 4 }, (_, i) => new Float32Array(32).fill(i / 4));
    const base = model.step(inp, h).temporalState;
    const longer = [new Float32Array(32).fill(9), ...h]; // oldest step beyond historyLen is dropped
    expect(close(base, model.step(inp, longer).temporalState)).toBe(true);
    expect(close(base, model.step(inp, []).temporalState)).toBe(false);
  });

  it("leaks no tensors", () => {
    const inp = randomInputs(cfg, { seed: 8 });
    model.step(inp, []); // warm up
    const before = tf.memory().numTensors;
    model.step(inp, [new Float32Array(32)]);
    model.encodeScreen(inp);
    model.score(new Float32Array(32), inp, []);
    model.step(randomInputs(cfg, { A: 0, K: 0, L: 0 }), []);
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe("SomaModel determinism and persistence", () => {
  const cfg = small();

  it("same seed -> same outputs, different seed -> different outputs", () => {
    const [a, b, c] = [new SomaModel(cfg, 3), new SomaModel(cfg, 3), new SomaModel(cfg, 4)];
    const inp = randomInputs(cfg, { seed: 9 });
    const [la, lb, lc] = [a, b, c].map((m) => m.step(inp, []).logits);
    expect([...la]).toEqual([...lb]);
    expect(close(la, lc)).toBe(false);
    [a, b, c].forEach((m) => m.dispose());
  });

  it("round-trips weights through an IOHandler and refuses mismatched configs", async () => {
    const a = new SomaModel(cfg, 1);
    const b = new SomaModel(cfg, 2);
    const inp = randomInputs(cfg, { seed: 10 });
    const io = memoryIO();
    await a.save(io);
    const before = tf.memory().numTensors;
    expect(await b.load(io)).toBe(true);
    expect(tf.memory().numTensors).toBe(before);
    expect([...b.step(inp, []).logits]).toEqual([...a.step(inp, []).logits]);
    const other = new SomaModel(small({ dModel: 16 }), 1);
    expect(await other.load(io)).toBe(false);
    expect(await other.load(memoryIO())).toBe(false);
    [a, b, other].forEach((m) => m.dispose());
  });

  it("round-trips through indexeddb:// and reports absence", async () => {
    const a = new SomaModel(cfg, 5);
    const b = new SomaModel(cfg, 6);
    expect(await b.load("indexeddb://soma-model-missing")).toBe(false);
    await a.save("indexeddb://soma-model-test");
    expect(await b.load("indexeddb://soma-model-test")).toBe(true);
    const inp = randomInputs(cfg, { seed: 11 });
    expect([...b.step(inp, []).logits]).toEqual([...a.step(inp, []).logits]);
    a.dispose();
    b.dispose();
  });

  it("dispose frees every weight", () => {
    const before = tf.memory().numTensors;
    const m = new SomaModel(cfg, 0);
    expect(tf.memory().numTensors).toBeGreaterThan(before);
    m.dispose();
    expect(tf.memory().numTensors).toBe(before);
    expect(() => m.encodeScreen(randomInputs(cfg))).toThrow();
  });
});

describe("ScreenHistory", () => {
  it("keeps the most recent len copies, oldest first", () => {
    const h = new ScreenHistory(2, { dim: 2 });
    const s = Float32Array.from([1, 1]);
    h.push(s);
    s[0] = 5; // a copy was stored
    h.push(Float32Array.from([2, 2]));
    h.push(Float32Array.from([3, 3]));
    expect(h.states().map((x) => x[0])).toEqual([2, 3]);
    expect(() => h.push(new Float32Array(3))).toThrow();
    h.clear();
    expect(h.states()).toEqual([]);
  });

  it("optionally resets on app switch", () => {
    const h = new ScreenHistory(4, { resetOnAppSwitch: true });
    h.observeApp("code");
    h.push(new Float32Array(2));
    h.observeApp("CODE");
    expect(h.size).toBe(1);
    h.observeApp("chrome");
    expect(h.size).toBe(0);
    expect(() => new ScreenHistory(0)).toThrow();
  });
});

describe("SomaModel default config", () => {
  it("builds and steps with the shipped config", () => {
    const cfg = DEFAULT_CONFIG.model;
    const m = new SomaModel(cfg, DEFAULT_CONFIG.seed);
    const out = m.step(randomInputs(cfg, { seed: 1, L: 64, A: cfg.maxCandidates, K: cfg.maxMemories }),
      [new Float32Array(cfg.dModel)]);
    expect(out.logits.length).toBe(1 + cfg.maxCandidates);
    expect([...out.logits].every(Number.isFinite)).toBe(true);
    m.dispose();
  }, 60_000);
});
