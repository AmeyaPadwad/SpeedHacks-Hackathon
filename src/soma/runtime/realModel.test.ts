import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import { OnlinePreferenceRanker } from "../learning/preferenceRanker";
import { TextEmbedder } from "../memory/embedder";
import { SomaModel } from "../models/somaModel";
import { ActionType } from "../types";
import { SomaRuntime } from "./pipeline";
import { FakeCapturer, FakeClock, LinesOcr, testConfig } from "./testFakes";

beforeAll(async () => {
  await tf.setBackend("cpu");
});

describe("SomaRuntime with the real (untrained) SomaModel", () => {
  it("the pette profile speaks on an editor and on a traceback, stays silent on an unknown app", async () => {
    const cfg = testConfig({ cooldownS: 0 });
    const model = new SomaModel(cfg.model, 0);
    try {
      const mk = (label: string, lines: string[]) => new SomaRuntime(cfg, {
        capturer: new FakeCapturer(label), ocr: new LinesOcr(lines), model,
        embedder: new TextEmbedder(cfg.model.dModel), ranker: new OnlinePreferenceRanker(), clock: new FakeClock().now,
      });
      const editor = await mk("main.py - Pette - Visual Studio Code", ["def total(items):"]).step();
      expect(editor.skippedReason).toBeNull();
      expect(editor.suggestion.candidate?.action).toBe(ActionType.SUGGEST_NEXT_STEP);
      expect(Object.values(editor.suggestion.distribution).every(Number.isFinite)).toBe(true);

      const tb = await mk("dev@box: ~/proj - Terminal", ["Traceback (most recent call last):",
        "ZeroDivisionError: division by zero"]).step();
      expect(tb.candidates[0].action).toBe(ActionType.EXPLAIN_ERROR);
      expect(tb.suggestion.candidate).not.toBeNull();

      const idle = await mk("", ["Welcome"]).step();
      expect(idle.candidates).toEqual([]);
      expect(idle.suggestion.candidate).toBeNull();
    } finally {
      model.dispose();
    }
  });
});
