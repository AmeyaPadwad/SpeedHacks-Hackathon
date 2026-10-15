import { describe, expect, it } from "vitest";
import { ActionType, candidateKey, FeedbackOutcome, SourceType } from "../types";
import { mem } from "../policy/testUtil";
import { REVIEW_CHANGES, WRITE_TESTS } from "../policy/contextCandidates";
import { OnlinePreferenceRanker } from "../learning/preferenceRanker";
import { SomaRuntime } from "./pipeline";
import {
  CharEmbedder,
  FakeCapturer,
  FakeClock,
  FakeModel,
  FakeRanker,
  FakeRetriever,
  LinesOcr,
  testConfig,
} from "./testFakes";

const EDITOR = "main.py - Pette - Visual Studio Code";
const TERMINAL = "dev@box: ~/proj - Terminal";
const TRACEBACK = [
  "Traceback (most recent call last):",
  '  File "C:/proj/app/calc.py", line 40, in divide',
  "ZeroDivisionError: division by zero",
];

function setup(opts: { label?: string; lines?: string[]; policy?: Parameters<typeof testConfig>[0];
  retriever?: FakeRetriever; ranker?: FakeRanker; model?: FakeModel; fileRefs?: string[] } = {}) {
  const cfg = testConfig(opts.policy);
  const clock = new FakeClock();
  const capturer = new FakeCapturer(opts.label ?? EDITOR);
  const ocr = new LinesOcr(opts.lines ?? ["def total(items):", "    return sum(items)"]);
  const model = opts.model ?? new FakeModel(cfg.model.dModel);
  const ranker = opts.ranker ?? new FakeRanker();
  const rt = new SomaRuntime(cfg, {
    capturer, ocr, model, embedder: new CharEmbedder(cfg.model.dModel), retriever: opts.retriever ?? null,
    ranker, clock: clock.now, fileRefs: opts.fileRefs ? async () => opts.fileRefs! : undefined,
  });
  return { rt, cfg, clock, capturer, ocr, model, ranker };
}

describe("SomaRuntime", () => {
  it("an error on screen -> EXPLAIN_ERROR", async () => {
    const { rt } = setup({ label: TERMINAL, lines: TRACEBACK });
    const r = await rt.step();
    expect(r.skippedReason).toBeNull();
    expect(r.window.appName).toBe("windowsterminal");
    expect(r.suggestion.candidate?.action).toBe(ActionType.EXPLAIN_ERROR);
    expect(r.suggestion.screenVersion).toBe(r.frameVersion);
    expect(Object.values(r.suggestion.distribution).reduce((a, b) => a + b)).toBeCloseTo(1, 4);
    expect(Object.keys(r.timingsMs).sort()).toEqual(["capture", "encode", "gate", "ocr", "parse", "policy", "retrieve", "total"]);
    expect(rt.processed).toBe(1);
    expect(rt.lastResult).toBe(r);
  });

  it("an editor -> review / write-tests templates", async () => {
    const { rt } = setup();
    const r = await rt.step();
    expect(r.window.appName).toBe("code");
    expect(r.candidates.map((c) => c.template)).toEqual([REVIEW_CHANGES, WRITE_TESTS]);
    expect([REVIEW_CHANGES, WRITE_TESTS]).toContain(r.suggestion.candidate?.template);
    expect(r.suggestion.candidate?.text).toContain('"main.py"');
  });

  it("templates are off without policy.contextTemplates", async () => {
    const { rt } = setup({ policy: { contextTemplates: false } });
    const r = await rt.step();
    expect(r.candidates).toEqual([]);
    expect(r.suggestion.candidate).toBeNull();
  });

  it("a quiet app stays silent", async () => {
    const { rt } = setup({ policy: { quietApps: ["code"] } });
    const r = await rt.step();
    expect(r.candidates.length).toBeGreaterThan(0);
    expect(r.suggestion.candidate).toBeNull();
    expect(r.suggestion.reason).toContain("quiet");
  });

  it("cooldown, then the shown candidate is not repeated within the duplicate window", async () => {
    const { rt, clock } = setup();
    const first = await rt.step();
    const key = candidateKey(first.suggestion.candidate!);
    clock.t += 5;
    const second = await rt.step(true);
    expect(second.suggestion.candidate).toBeNull();
    expect(second.suggestion.reason).toContain("cooldown");
    clock.t += 60;
    const third = await rt.step(true);
    expect(third.suggestion.candidate).not.toBeNull();
    expect(candidateKey(third.suggestion.candidate!)).not.toBe(key);
  });

  it("a dismissal suppresses the candidate; ignored is not a dismissal", async () => {
    const { rt, clock, ranker } = setup({ policy: { cooldownS: 0, duplicateWindowS: 600 } });
    const first = await rt.step();
    const key = candidateKey(first.suggestion.candidate!);
    await rt.feedback(first.suggestion.id, FeedbackOutcome.REJECTED);
    expect(ranker.updates.map((u) => u.outcome)).toEqual([FeedbackOutcome.REJECTED]);
    for (let i = 0; i < 4; i++) {
      clock.t += 5;
      const r = await rt.step(true);
      expect(r.suggestion.candidate === null || candidateKey(r.suggestion.candidate) !== key).toBe(true);
      expect(r.candidates.map(candidateKey)).toContain(key); // still a candidate, never shown
    }

    const b = setup({ policy: { cooldownS: 0, duplicateWindowS: 1 } });
    const s1 = await b.rt.step();
    await b.rt.feedback(s1.suggestion.id, FeedbackOutcome.IGNORED);
    expect(b.ranker.updates).toEqual([]); // ignored never trains
    b.clock.t += 2;
    const s2 = await b.rt.step(true);
    expect(candidateKey(s2.suggestion.candidate!)).toBe(candidateKey(s1.suggestion.candidate!));
  });

  it("paused -> skipped without OCR or model work; resume processes again", async () => {
    const { rt, ocr, model, capturer } = setup();
    await rt.step();
    expect(rt.history.size).toBe(1);
    rt.pause();
    expect(rt.history.size).toBe(0);
    const r = await rt.step(true);
    expect(r.skippedReason).toBe("paused");
    expect(r.suggestion.candidate).toBeNull();
    expect(ocr.calls).toBe(1);
    expect(model.encodes).toBe(1);
    expect(rt.currentSuggestionValid(rt.frameVersion)).toBe(false);
    rt.resume();
    expect(capturer.resets).toBe(1);
    expect((await rt.step(true)).skippedReason).toBeNull();
  });

  it("no screen / unchanged / not ready / excluded are skipped before OCR", async () => {
    const { rt, ocr, capturer } = setup();
    capturer.active = false;
    expect((await rt.step()).skippedReason).toBe("no-consent");
    capturer.active = true;
    await rt.step();
    capturer.changed = false;
    expect((await rt.step()).skippedReason).toBe("unchanged");
    expect(ocr.calls).toBe(1);

    const ex = setup({ label: "Bitwarden - Google Chrome" });
    const r = await ex.rt.step(true);
    expect(r.skippedReason?.startsWith("excluded")).toBe(true);
    expect(r.window.title).toBe(""); // excluded titles are not echoed
    expect(ex.ocr.calls).toBe(0);
    expect(ex.rt.lastResult).toBeNull();

    const inc = setup({ label: "", lines: ["Incognito window"] });
    inc.ocr.lines = ["You've gone Incognito"]; // at the top of the screen once OCR'd
    inc.ocr.recognize = async (_i, windowId) => [{ text: "You've gone Incognito", bbox: [0.1, 0.01, 0.5, 0.04],
      confidence: 1, source: "ocr", uiRole: null, windowId, timestampMs: 0 }];
    expect((await inc.rt.step()).skippedReason?.startsWith("excluded")).toBe(true);
    expect(inc.model.encodes).toBe(0);

    const nr = new SomaRuntime(testConfig(), { capturer: new FakeCapturer(EDITOR), embedder: new CharEmbedder() });
    expect((await nr.step()).skippedReason).toBe("not-ready");
  });

  it("pausing mid-step drops the result and records nothing as shown", async () => {
    const { rt, ocr } = setup();
    const orig = ocr.recognize.bind(ocr);
    ocr.recognize = async (img, id) => {
      rt.pause();
      return orig(img, id);
    };
    const r = await rt.step();
    expect(r.skippedReason).toBe("paused-during-step");
    expect(rt.lastResult).toBeNull();
    expect(rt.gate.inCooldown()).toBe(false);
  });

  it("validity follows the frame version and feedback", async () => {
    const { rt } = setup();
    const r = await rt.step();
    expect(rt.currentSuggestionValid(r.frameVersion)).toBe(true);
    expect(rt.currentSuggestionValid(r.frameVersion + 1)).toBe(false);
    expect(rt.knownSuggestion(r.suggestion.id)).toBe(true);
    await rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED);
    expect(rt.currentSuggestionValid(r.frameVersion)).toBe(false);
    await rt.feedback("sug_unknown", FeedbackOutcome.ACCEPTED); // ignored, no throw
  });

  it("retrieval uses redacted text and memories become candidates", async () => {
    const retriever = new FakeRetriever([mem("doc1", SourceType.DOCUMENT, "Q3 report draft", 0.9, true, { sourceUri: "f1/report.md" })]);
    const { rt } = setup({ retriever, lines: ["Quarterly report", "contact me: alice@example.com"] });
    rt.setAllowedFolders(["f1"]);
    const r = await rt.step();
    expect(retriever.queries[0]).not.toContain("alice@example.com");
    expect(r.retrieved.map((m) => m.item.id)).toEqual(["doc1"]);
    expect(r.candidates.some((c) => c.action === ActionType.COMPARE_WITH_PRIOR_WORK && c.memoryRef === "doc1")).toBe(true);
    // a memory from a folder the user removed is dropped
    rt.setAllowedFolders([]);
    expect((await rt.step(true)).retrieved).toEqual([]);
  });

  it("on-screen paths become authorized refs only inside allowed folders", async () => {
    const lines = TRACEBACK;
    const { rt } = setup({ label: TERMINAL, lines, fileRefs: ["f1/app/calc.py", "f2/other.py"],
      policy: { cooldownS: 0 } });
    let r = await rt.step();
    expect(r.candidates.some((c) => c.action === ActionType.OPEN_RELEVANT_FILE)).toBe(false);
    rt.setAllowedFolders(["f1"]);
    r = await rt.step(true);
    const open = r.candidates.find((c) => c.action === ActionType.OPEN_RELEVANT_FILE)!;
    expect(open.fileRef).toBe("f1/app/calc.py");
    expect(open.line).toBe(40);
    expect(rt.authorizedFile("f1/app/calc.py")).toBe(true);
    expect(rt.authorizedFile("f2/other.py")).toBe(false);
    expect(rt.authorizedFile("f1/../f2/other.py")).toBe(false);
  });

  it("blocked candidates are masked for the model and never shown", async () => {
    const model = new FakeModel(32, 3, (inp) => [0, ...inp.candidateEmbeddings.map(() => 3)]);
    const { rt } = setup({ model });
    const r = await rt.step();
    expect(Array.from(model.lastInputs!.candidateMask!)).toEqual([0, 0, 1, 1, 1, 1, 1, 1]);
    expect(model.lastInputs!.candidateTypes.slice(0, 2).every((t) => t > 0)).toBe(true);
    expect(r.suggestion.candidate).not.toBeNull();
  });

  it("the ranker bias can change the winner", async () => {
    const ranker = new FakeRanker();
    ranker.biasFn = (i) => (i === 1 ? 5 : 0);
    const { rt } = setup({ ranker });
    const r = await rt.step();
    expect(r.suggestion.candidate?.template).toBe(WRITE_TESTS);
  });

  it("steps are serialised", async () => {
    const { rt, ocr } = setup();
    let inFlight = 0;
    let maxInFlight = 0;
    const orig = ocr.recognize.bind(ocr);
    ocr.recognize = async (img, id) => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      return orig(img, id);
    };
    await Promise.all([rt.step(true), rt.step(true), rt.step(true)]);
    expect(maxInFlight).toBe(1);
    expect(rt.steps).toBe(3);
  });

  it("the ledger never executes anything", async () => {
    const { rt } = setup();
    const r = await rt.step();
    await rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED);
    expect(rt.ledger.executors.size).toBe(0);
    expect(rt.ledger.executedCount).toBe(0);
    expect(rt.ledger.unapprovedExecutionCount).toBe(0);
  });

  it("feedback hands back a text-free snapshot and trains the real preference ranker", async () => {
    const cfg = testConfig({ cooldownS: 0, duplicateWindowS: 0 });
    const ranker = new OnlinePreferenceRanker();
    const rt = new SomaRuntime(cfg, { capturer: new FakeCapturer(EDITOR), ocr: new LinesOcr(["def f():"]),
      model: new FakeModel(cfg.model.dModel), embedder: new CharEmbedder(cfg.model.dModel), ranker });
    let learned = 0;
    rt.onLearned = () => learned++;
    const r = await rt.step();
    const shown = r.suggestion.candidate!;
    const snap = await rt.feedback(r.suggestion.id, FeedbackOutcome.REJECTED);
    expect(snap?.candidateKey).toBe(candidateKey(shown));
    expect(snap?.app).toBe("code");
    expect(snap?.shownIndex).toBe(1);
    expect(snap?.context).toBeInstanceOf(Float32Array);
    expect(JSON.stringify(snap)).not.toContain("def f()");
    expect(await rt.feedback(r.suggestion.id, FeedbackOutcome.ACCEPTED)).toBeNull(); // single use
    expect(ranker.updates).toBe(1);
    expect(learned).toBe(1);
    expect(ranker.candidateBias("code", shown.action, candidateKey(shown))).toBeLessThan(0);
  });
});
