import { describe, expect, it } from "vitest";
import { ActionType, type Candidate, makeCandidate, type OCRElement, type ScreenObservation, type Suggestion } from "../types";
import {
  type Box,
  findTarget,
  groundBox,
  groundPoint,
  iou,
  isStale,
  normalizeText,
  similarityRatio,
  textMatches,
  verifyTarget,
} from "./grounding";
import { el, win } from "./testUtil";

const ERR = "ZeroDivisionError: division by zero";
const BOX: [number, number, number, number] = [0.1, 0.5, 0.6, 0.54];

const obs = (elements: OCRElement[], windowId = "", version = 2): ScreenObservation =>
  ({ window: win("code", "t", { windowId }), elements, version, timestampMs: 0 });
const cand = (text = ERR, box = BOX, windowId = ""): Candidate =>
  makeCandidate({ action: ActionType.EXPLAIN_ERROR, text: `Explain the error "${text}"`, target: el(text, box, { windowId }) });
const sug = (candidate: Candidate | null, screenVersion: number): Suggestion =>
  ({ candidate, confidence: 0.9, screenVersion, reason: "", distribution: {}, id: "s", createdMs: 0 });

describe("grounding", () => {
  it("iou, normalize and fuzzy text", () => {
    expect(iou(BOX, BOX)).toBeCloseTo(1);
    expect(iou([0, 0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6])).toBe(0);
    expect(iou([0, 0, 0, 0], [0, 0, 0, 0])).toBe(0);
    expect(normalizeText("  ZeroDivisionError:\tdivision  BY zero. ")).toBe("zerodivisionerror: division by zero");
    expect(textMatches(ERR, "ZeroDivisionErr0r: division by zero")).toBe(true);
    expect(textMatches(ERR, "KeyError: 'x'")).toBe(false);
    expect(textMatches("", "")).toBe(false);
    expect(similarityRatio("abcd", "bcde")).toBeCloseTo(0.75); // difflib gives 0.75
  });

  it("verifies same, resized and slightly shifted targets", () => {
    expect(verifyTarget(cand(), obs([el(ERR, BOX)]))).toBe(true);
    const resized: Box = [0.11, 0.505, 0.58, 0.545];
    expect(iou(BOX, resized)).toBeGreaterThanOrEqual(0.5);
    expect(verifyTarget(cand(), obs([el(ERR, [...resized])]))).toBe(true);
    const shifted: Box = [0.1, 0.53, 0.6, 0.57];
    expect(iou(BOX, shifted)).toBeLessThan(0.5);
    expect(verifyTarget(cand(), obs([el(ERR, [...shifted])]))).toBe(true);
  });

  it("fails when moved, missing, changed or in another window", () => {
    expect(verifyTarget(cand(), obs([el(ERR, [0.1, 0.1, 0.6, 0.14])]))).toBe(false);
    expect(verifyTarget(cand(), obs([el("All tests passed", BOX)]))).toBe(false);
    expect(verifyTarget(cand(), obs([]))).toBe(false);
    expect(verifyTarget(cand(), obs([el("KeyError: 'user'", BOX)]))).toBe(false);
    expect(verifyTarget(cand(ERR, BOX, "w1"), obs([el(ERR, BOX)], "w2"))).toBe(false);
    expect(verifyTarget(cand(ERR, BOX, "w1"), obs([el(ERR, BOX)], "w1"))).toBe(true);
    expect(verifyTarget(makeCandidate({ action: ActionType.ASK_CLARIFYING_QUESTION, text: "?" }), obs([]))).toBe(true);
  });

  it("picks the best of duplicates", () => {
    const near = el(ERR, [0.1, 0.51, 0.6, 0.55]);
    const exact = el(ERR, BOX);
    expect(findTarget(cand().target!, obs([near, exact]))).toBe(exact);
  });

  it("ground point maps and clamps", () => {
    expect(groundPoint([0, 0, 1, 1], [1920, 1080])).toEqual([960, 540]);
    expect(groundPoint(el("x", [0, 0, 0, 0]), [100, 50])).toEqual([0, 0]);
    expect(groundPoint([1.2, 1.2, 1.4, 1.4], [100, 50])).toEqual([99, 49]);
    expect(groundBox(BOX, [1000, 100])).toEqual([100, 50, 600, 54]);
    expect(() => groundPoint(BOX, [0, 10])).toThrow();
  });

  it("staleness by version, change predicate and target", () => {
    const s = sug(cand(), 5);
    expect(isStale(s, 5)).toBe(false);
    expect(isStale(s, 6)).toBe(true);
    expect(isStale(s, 6, false)).toBe(false);
    expect(isStale(s, 6, true)).toBe(true);
    expect(isStale(s, 7, (o, n) => n - o > 1)).toBe(true);
    expect(isStale(s, 6, (o, n) => n - o > 1)).toBe(false);
    expect(isStale(s, 4)).toBe(true);
    expect(isStale(s, 6, false, obs([el(ERR, BOX)]))).toBe(false);
    expect(isStale(s, 6, false, obs([el("ok", BOX)]))).toBe(true);
    expect(isStale(sug(null, 5), 5, null, obs([]))).toBe(false);
  });
});
