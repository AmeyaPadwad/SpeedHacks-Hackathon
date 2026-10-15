import { beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_CONFIG, type PolicyConfig } from "../config";
import { ActionType, type Candidate, candidateKey, makeCandidate, type OCRElement, type ScreenObservation, SourceType } from "../types";
import { distributionLabels, SafetyGate, toProbs } from "./safetyGate";
import { el, memItem, win } from "./testUtil";

const ERR = "ZeroDivisionError: division by zero";
const BOX: [number, number, number, number] = [0.1, 0.5, 0.6, 0.54];

class FakeClock {
  t = 1000;
  now = () => this.t;
  advance(s: number) { this.t += s; }
}

const policy = (p: Partial<PolicyConfig> = {}): PolicyConfig =>
  ({ ...DEFAULT_CONFIG.policy, relevanceThreshold: 0.55, marginOverAbstain: 0.05, cooldownS: 30,
    duplicateWindowS: 600, quietApps: ["Spotify.exe", "zoom"], ...p });
const obs = (elements: OCRElement[] = [el(ERR, BOX)], app = "code", version = 7): ScreenObservation =>
  ({ window: win(app, "t"), elements, version, timestampMs: 0 });
const explain = (): Candidate => makeCandidate({ action: ActionType.EXPLAIN_ERROR, text: `Explain the error "${ERR}"`, target: el(ERR, BOX), id: "c1" });
const ask = (): Candidate => makeCandidate({ action: ActionType.ASK_CLARIFYING_QUESTION, text: "Which error first?", id: "c2" });
const CONFIDENT = [0.1, 0.8, 0.1];

let clock: FakeClock;
let gate: SafetyGate;
beforeEach(() => {
  clock = new FakeClock();
  gate = new SafetyGate(policy(), clock.now);
});

describe("SafetyGate", () => {
  it("speaks when confident", () => {
    const s = gate.decide(CONFIDENT, [explain(), ask()], obs());
    expect(s.candidate?.action).toBe(ActionType.EXPLAIN_ERROR);
    expect(s.confidence).toBeCloseTo(0.8);
    expect(s.screenVersion).toBe(7);
    expect(Object.keys(s.distribution).sort()).toEqual(["ASK_CLARIFYING_QUESTION", "EXPLAIN_ERROR", "NO_SUGGESTION"]);
    expect(Object.values(s.distribution).reduce((a, b) => a + b)).toBeCloseTo(1);
    expect(s.reason).toContain("EXPLAIN_ERROR");
    expect(s.reason).toContain("threshold");
    expect(s.id.startsWith("sug_")).toBe(true);
  });

  it("accepts batched logits with padding", () => {
    const s = gate.decide([new Float32Array([0, 4, -1, -Infinity])], [explain(), ask()], obs());
    expect(s.candidate).not.toBeNull();
    expect(Object.values(s.distribution).reduce((a, b) => a + b)).toBeCloseTo(1);
  });

  it("is silent below threshold or without margin", () => {
    const s = gate.decide([0.3, 0.4, 0.3], [explain(), ask()], obs());
    expect(s.candidate).toBeNull();
    expect(s.reason).toContain("low relevance");
    expect(s.confidence).toBeCloseTo(0.3);
    const g = new SafetyGate(policy({ relevanceThreshold: 0.4, marginOverAbstain: 0.05 }), clock.now);
    const s2 = g.decide([0.48, 0.52], [explain()], obs());
    expect(s2.candidate).toBeNull();
    expect(s2.reason).toContain("does not beat NO_SUGGESTION");
  });

  it("cooldown", () => {
    expect(gate.decide(CONFIDENT, [explain(), ask()], obs()).candidate).not.toBeNull();
    clock.advance(10);
    const s = gate.decide([0.1, 0.1, 0.8], [explain(), ask()], obs());
    expect(s.candidate).toBeNull();
    expect(s.reason).toContain("cooldown");
    clock.advance(25);
    expect(gate.decide([0.1, 0.1, 0.8], [explain(), ask()], obs()).candidate?.action).toBe(ActionType.ASK_CLARIFYING_QUESTION);
  });

  it("a dismissal suppresses for the duplicate window", () => {
    const s = gate.decide(CONFIDENT, [explain()], obs());
    gate.recordDismissed(candidateKey(s.candidate!));
    clock.advance(60);
    const s2 = gate.decide(CONFIDENT.slice(0, 2), [explain()], obs());
    expect(s2.candidate).toBeNull();
    expect(s2.reason).toContain("dismissed");
    clock.advance(600);
    expect(gate.decide(CONFIDENT.slice(0, 2), [explain()], obs()).candidate).not.toBeNull();
  });

  it("a shown suggestion is not repeated, the next best can speak", () => {
    expect(gate.decide([0.05, 0.9, 0.05], [explain(), ask()], obs()).candidate?.action).toBe(ActionType.EXPLAIN_ERROR);
    clock.advance(60);
    const s = gate.decide([0.1, 0.6, 0.3], [explain(), ask()], obs());
    expect(s.candidate).toBeNull();
    expect(s.reason).toContain("already shown");
    expect(gate.decide([0.05, 0.3, 0.65], [explain(), ask()], obs()).candidate?.action).toBe(ActionType.ASK_CLARIFYING_QUESTION);
  });

  it.each(["Spotify.exe", "spotify", "Zoom.exe"])("quiet app %s", (app) => {
    expect(gate.decide(CONFIDENT, [explain(), ask()], obs(undefined, app)).reason).toContain("quiet");
    expect(gate.decide(CONFIDENT, [explain(), ask()], obs(), { appName: app }).candidate).toBeNull();
  });

  it("the default quiet apps silence chat and mail", () => {
    const g = new SafetyGate({ ...DEFAULT_CONFIG.policy }, clock.now);
    expect(g.decide(CONFIDENT, [explain(), ask()], obs(undefined, "slack")).candidate).toBeNull();
  });

  it("grounding failure is silent, starts no cooldown, falls back to untargeted", () => {
    const movedAway = obs([el("All tests passed", BOX)]);
    const s = gate.decide([0.1, 0.9], [explain()], movedAway);
    expect(s.candidate).toBeNull();
    expect(s.reason).toContain("target no longer on screen");
    expect(gate.inCooldown()).toBe(false);
    expect(gate.decide([0.05, 0.6, 0.35], [explain(), ask()], movedAway).candidate).toBeNull();
    const g2 = new SafetyGate(policy({ relevanceThreshold: 0.3 }), new FakeClock().now);
    const s2 = g2.decide([0.05, 0.55, 0.4], [explain(), ask()], movedAway);
    expect(s2.candidate?.action).toBe(ActionType.ASK_CLARIFYING_QUESTION);
    expect(s2.reason).toContain("skipped");
  });

  it("no candidates is silent; record=false starts nothing", () => {
    const s = gate.decide([5.0], [], obs());
    expect(s.candidate).toBeNull();
    expect(s.distribution).toEqual({ NO_SUGGESTION: 1 });
    expect(gate.decide(CONFIDENT, [explain(), ask()], obs(), { record: false }).candidate).not.toBeNull();
    expect(gate.inCooldown()).toBe(false);
    expect(gate.isDuplicate(candidateKey(explain()))).toBe(false);
  });

  it("blocked candidates are skipped", () => {
    const openFile = makeCandidate({ action: ActionType.OPEN_RELEVANT_FILE, text: "Open secret.py", fileRef: "C:/private/secret.py" });
    const blocked = SafetyGate.blockedMask([openFile, ask()], (p) => p.startsWith("src/"));
    expect(blocked).toEqual([true, false]);
    const s = gate.decide([0.05, 0.9, 0.05], [openFile, ask()], obs(), { blocked });
    expect(s.candidate).toBeNull();
    expect(s.reason).toContain("blocked");
  });

  it("blocked mask rules", () => {
    const item = memItem("m1", SourceType.DOCUMENT, "", { sourceUri: "C:/docs/a.md" });
    const unapproved = memItem("m2", SourceType.DOCUMENT, "", { userApproved: false });
    const mems = new Map([["m1", { item, embedding: new Float32Array(4), score: 0.9 }],
      ["m2", { item: unapproved, embedding: new Float32Array(4), score: 0.9 }]]);
    const cmp = (t: string, ref: string) => makeCandidate({ action: ActionType.COMPARE_WITH_PRIOR_WORK, text: t, memoryRef: ref });
    const f = makeCandidate({ action: ActionType.OPEN_RELEVANT_FILE, text: "o", fileRef: "src/a.py" });
    expect(SafetyGate.blockedMask([cmp("c1", "m1"), cmp("c2", "m2"), cmp("c3", "missing"), f], null, [], mems))
      .toEqual([false, true, true, true]);
    expect(SafetyGate.blockedMask([cmp("c1", "m1"), f], () => true, ["C:/docs/a.md"], mems)).toEqual([true, false]);
    expect(SafetyGate.blockedMask([cmp("c1", "m1")], null, ["m1"])).toEqual([true]);
  });

  it("toProbs and labels", () => {
    expect(toProbs([0.2, 0.8])[1]).toBeCloseTo(0.8);
    const p = toProbs([0, 2]);
    expect(p[0] + p[1]).toBeCloseTo(1);
    expect(p[1]).toBeGreaterThan(p[0]);
    expect(toProbs([-Infinity, -Infinity])).toEqual([1, 0]);
    expect(distributionLabels([explain(), explain(), ask()])).toEqual(
      ["NO_SUGGESTION", "EXPLAIN_ERROR", "EXPLAIN_ERROR#2", "ASK_CLARIFYING_QUESTION"]);
    expect(() => new SafetyGate(policy()).decide([1.0], [explain()], obs())).toThrow();
  });

  it("abstain wins on typical uncertain scores with a strict threshold", () => {
    let s = 1;
    const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const normal = () => Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand());
    let spoke = 0;
    for (let i = 0; i < 50; i++) {
      const logits = Array.from({ length: 4 }, () => normal() * 0.5);
      if (gate.decide(logits, [explain(), ask(), ask()], obs(), { record: false }).candidate) spoke++;
    }
    expect(spoke).toBeLessThanOrEqual(2); // almost never speaks
  });
});
