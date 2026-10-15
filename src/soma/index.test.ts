import { describe, expect, it } from "vitest";
import { createSoma, skipWhileInFront, type GrantableCapturer } from "./index";
import { FakeDir } from "./memory/fakeFs";
import { MemoryStore } from "./memory/store";
import { memoryStateStore } from "./runtime/stateStore";
import { FakeCapturer, FakeClock, FakeModel, LinesOcr, testConfig } from "./runtime/testFakes";
import { FeedbackOutcome } from "./types";

class GrantCapturer extends FakeCapturer implements GrantableCapturer {
  granted = false;
  onEnded?: () => void;
  constructor(label: string, public allow = true) {
    super(label);
    this.active = false;
  }
  async grant(): Promise<boolean> {
    this.active = this.allow;
    return this.allow;
  }
  revoke(): void {
    this.active = false;
  }
}

let dbN = 0;
const TERMINAL = "dev@box: ~/proj - Terminal";
const SECRET_LINE = "my private notes about the merger";

async function make(opts: { label?: string; lines?: string[]; allow?: boolean } = {}) {
  const cfg = testConfig({ cooldownS: 0, duplicateWindowS: 0 });
  const store = await MemoryStore.open(`soma-test-${++dbN}`);
  const capturer = new GrantCapturer(opts.label ?? "main.py - Pette - Visual Studio Code", opts.allow ?? true);
  const ocr = new LinesOcr(opts.lines ?? ["def total(items):", SECRET_LINE]);
  const state = memoryStateStore();
  let picks = 0;
  const soma = await createSoma({ cfg, deps: {
    store, capturer, state, clock: new FakeClock().now,
    createOcr: async () => ocr,
    createModel: async (c) => new FakeModel(c.model.dModel),
    pickDirectory: async () => {
      picks++;
      return null;
    },
    imageSize: async () => null,
  } });
  const look = async () => {
    expect(await soma.trigger()).toBe(true);
    await soma.scheduler.idle();
  };
  return { soma, store, capturer, ocr, state, look, picks: () => picks };
}

describe("createSoma (SomaService)", () => {
  it("does nothing before the screen is granted", async () => {
    const { soma } = await make();
    const st = await soma.status();
    expect(st).toMatchObject({ capturing: false, ready: false, paused: false, steps: 0, memories: 0 });
    expect(await soma.trigger()).toBe(false);
    expect(await soma.suggestion()).toEqual({ available: false, silent: true });
  });

  it("a declined picker grants nothing", async () => {
    const { soma } = await make({ allow: false });
    expect(await soma.grantScreen()).toBe(false);
    expect((await soma.status()).capturing).toBe(false);
  });

  it("grant -> ready -> look now -> a suggestion that never carries other screen text", async () => {
    const { soma, state } = await make();
    expect(await soma.grantScreen()).toBe(true);
    await soma.whenReady();
    expect((await soma.status()).ready).toBe(true);
    await soma.scheduler.idle();
    expect(await soma.trigger()).toBe(true);
    await soma.scheduler.idle();
    const p = await soma.suggestion();
    expect(p).toMatchObject({ available: true, silent: false, valid: true, app: "code", action: "SUGGEST_NEXT_STEP" });
    expect(p.template).toMatch(/review_changes|write_tests/);
    expect(JSON.stringify(p)).not.toContain("merger");
    expect(JSON.stringify(p)).not.toContain("def total");
    const st = await soma.status();
    expect(st).toMatchObject({ capturing: true, processed: 1, frameVersion: p.frameVersion, lastSkipReason: null });

    await soma.feedback(p.id!, FeedbackOutcome.REJECTED);
    expect((await soma.suggestion()).valid).toBe(false);
    const saved = await state.get<{ updates: number }>("preference-ranker");
    expect(saved?.updates).toBe(1);
    expect(JSON.stringify(saved)).not.toContain("merger");
  });

  it("pause / resume", async () => {
    const { soma, look } = await make();
    await soma.grantScreen();
    await soma.whenReady();
    await look();
    await soma.pause();
    expect((await soma.status()).paused).toBe(true);
    expect(await soma.trigger()).toBe(false);
    expect((await soma.suggestion()).valid).toBe(false);
    await soma.resume();
    expect((await soma.status()).paused).toBe(false);
    await look();
  });

  it("revoking the screen stops capture", async () => {
    const { soma } = await make();
    await soma.grantScreen();
    soma.revokeScreen();
    expect((await soma.status()).capturing).toBe(false);
    expect(await soma.trigger()).toBe(false);
  });

  it("folders: index in the background, offer authorized files, read them, forget them", async () => {
    const lines = ["Traceback (most recent call last):", '  File "C:/work/proj/app/calc.py", line 40, in divide',
      "ZeroDivisionError: division by zero"];
    const { soma, look, picks } = await make({ label: TERMINAL, lines });
    await soma.grantScreen();
    await soma.whenReady();

    expect(await soma.addFolder()).toBeNull(); // picker cancelled
    expect(picks()).toBe(1);

    const dir = new FakeDir("proj", { app: { "calc.py": "def divide(a, b):\n    return a / b\n" }, "notes.md": "# notes" });
    const folder = await soma.addFolder(dir as unknown as FileSystemDirectoryHandle);
    expect(folder).toMatchObject({ name: "proj", memories: 0 });
    expect((await soma.folders()).indexing).toBe(true);
    await soma.whenIndexed();
    const f = await soma.folders();
    expect(f.indexing).toBe(false);
    expect(f.lastReport).toEqual({ files: 2, skipped: 0 });
    expect(f.folders).toEqual([{ id: folder!.id, name: "proj", memories: 2 }]);
    expect((await soma.status()).memories).toBe(2);

    await look();
    const r = soma.runtime.lastResult!;
    const open = r.candidates.find((c) => c.action === "OPEN_RELEVANT_FILE");
    expect(open?.fileRef).toBe(`${folder!.id}/app/calc.py`);

    const file = await soma.readFile(`${folder!.id}/app/calc.py`);
    expect(file?.name).toBe("calc.py");
    expect(file?.text).toContain("def divide");
    expect(await soma.readFile("fld_other/app/calc.py")).toBeNull();
    expect(await soma.readFile(`${folder!.id}/../x.py`)).toBeNull();

    expect(await soma.removeFolder(folder!.id)).toBe(2);
    expect((await soma.folders()).folders).toEqual([]);
    expect(await soma.readFile(`${folder!.id}/app/calc.py`)).toBeNull();
    await look();
    expect(soma.runtime.lastResult!.candidates.some((c) => c.action === "OPEN_RELEVANT_FILE")).toBe(false);
  });

  it("the payload exposes a suggested file only while it is authorized", async () => {
    const lines = ['  File "C:/work/proj/app/calc.py", line 40, in divide', "ZeroDivisionError: division by zero"];
    const { soma, look } = await make({ label: TERMINAL, lines });
    await soma.grantScreen();
    await soma.whenReady();
    const folder = await soma.addFolder(new FakeDir("proj", { app: { "calc.py": "x = 1" } }) as unknown as FileSystemDirectoryHandle);
    await soma.whenIndexed();
    // make the open-file candidate win
    soma.runtime.model = new FakeModel(32, 3, (inp) => [0, ...inp.candidateTypes.map((t) => (t === 2 ? 5 : 1))]);
    await look();
    const p = await soma.suggestion();
    expect(p.action).toBe("OPEN_RELEVANT_FILE");
    expect(p.file).toBe(`${folder!.id}/app/calc.py`);
    expect(p.line).toBe(40);
    soma.runtime.setAllowedFolders([]);
    const q = await soma.suggestion();
    expect(q.file).toBeNull();
    expect(q.line).toBeNull();
  });
});

describe("skipWhileInFront", () => {
  const inner = () => {
    const calls: (boolean | undefined)[] = [];
    return {
      calls,
      cap: {
        active: true, version: 3, lastSkipReason: "rate-limited", reset: () => undefined,
        capture: async (force?: boolean) => { calls.push(force); return null; },
      },
    };
  };

  it("skips scheduled captures while Pette's page is in front, but never look-now", async () => {
    const { calls, cap } = inner();
    let front = true;
    const c = skipWhileInFront(cap, () => front);
    expect(await c.capture()).toBeNull();
    expect(c.lastSkipReason).toBe("pette-in-front");
    expect(calls).toEqual([]);
    await c.capture(true);
    expect(calls).toEqual([true]);
    front = false;
    await c.capture();
    expect(calls).toEqual([true, undefined]);
    expect(c.lastSkipReason).toBe("rate-limited");
    expect(c.version).toBe(3);
  });
});
