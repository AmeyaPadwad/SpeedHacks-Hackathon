import { describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG } from "../config";
import type { OCRElement, ScreenObservation, WindowInfo } from "../types";
import { appBucket, imageToPixels, observationToArrays } from "./featurize";
import { FakeOcr, FallbackOcr, NullOcr, TesseractOcr, type TessWorkerLike } from "./ocr";
import { IMAGENET_MEAN, IMAGENET_STD, changeRatio, dstToSrc, letterboxGeometry, srcToDst } from "./preprocess";
import { extractFileRefs, findButtons, findErrors, findFileRefs, inferRole, iou, looksLikeError, parseScreen } from "./screenParser";
import { PAD_ID, encode, fnv1a, hashBucket, tokenId, tokenize } from "./tokenizer";

const el = (text: string, bbox: [number, number, number, number] = [0, 0, 0.1, 0.1], uiRole: string | null = null): OCRElement => ({
  text, bbox, confidence: 1, source: "ocr", uiRole, windowId: "w", timestampMs: 0,
});
const WIN: WindowInfo = { appName: "windowsterminal", title: "t", windowId: "w", surface: "window" };
const img = (w: number, h: number, fill: number): ImageData =>
  ({ data: new Uint8ClampedArray(w * h * 4).fill(fill), width: w, height: h }) as ImageData;

// ----------------------------------------------------------------- tokenizer

describe("tokenizer", () => {
  it("is deterministic, lower-cased and in range", () => {
    expect(tokenize("Hello, World 42!")).toEqual(["hello", ",", "world", "42", "!"]);
    expect(fnv1a("a")).toBe(0xe40c292c); // reference FNV-1a 32-bit
    const ids = encode("Run tests", 512);
    expect(ids).toEqual(encode("run TESTS", 512));
    expect(ids.every((i) => i >= 2 && i < 512)).toBe(true);
    expect(encode("a b c", 512, 2)).toHaveLength(2);
    expect(tokenId("x", 100)).toBe(tokenId("x", 100));
    expect(hashBucket("Code", 64)).toBe(hashBucket("code", 64));
  });
});

// ----------------------------------------------------------------- OCR

describe("OCR engines", () => {
  it("FakeOcr stamps ids and redacts like real OCR", async () => {
    const ocr = new FakeOcr([{ text: "password=hunter2" }, { text: "   " }, { text: "hello", uiRole: "button" }]);
    const els = await ocr.recognize(img(4, 4, 0), "w1");
    expect(els).toHaveLength(2);
    expect(els[0].text).not.toContain("hunter2");
    expect(els.every((e) => e.windowId === "w1")).toBe(true);
    expect(await new NullOcr().recognize()).toEqual([]);
  });

  it("FallbackOcr swallows backend errors", async () => {
    const broken = { recognize: async () => { throw new Error("wasm crashed"); }, dispose: async () => {} };
    const eng = new FallbackOcr(broken);
    expect(await eng.recognize(img(4, 4, 0), "w")).toEqual([]);
    expect(eng.lastError).toContain("wasm crashed");
  });

  it("TesseractOcr maps lines to normalized, filtered, redacted elements and is lazy", async () => {
    const worker: TessWorkerLike = {
      recognize: vi.fn(async () => ({ data: { blocks: [{ paragraphs: [{ lines: [
        { text: "Traceback (most recent call last):\n", confidence: 91, bbox: { x0: 0, y0: 0, x1: 100, y1: 10 } },
        { text: "token=abc", confidence: 80, bbox: { x0: 0, y0: 20, x1: 50, y1: 30 } },
        { text: "noise", confidence: 20, bbox: { x0: 0, y0: 40, x1: 50, y1: 50 } },
      ] }] }] } })),
      terminate: vi.fn(async () => undefined),
    };
    const create = vi.fn(async () => worker);
    const prepare = vi.fn((image: ImageData, maxSide: number) => {
      const s = Math.min(1, maxSide / Math.max(image.width, image.height));
      return { source: image, width: image.width * s, height: image.height * s };
    });
    const ocr = new TesseractOcr({ createWorker: create, prepare, maxSide: 100 });
    expect(create).not.toHaveBeenCalled();
    const [a, b] = await Promise.all([ocr.recognize(img(200, 100, 0), "w"), ocr.recognize(img(200, 100, 0), "w")]);
    expect(create).toHaveBeenCalledOnce();
    expect(a.map((e) => e.text)).toEqual(b.map((e) => e.text));
    expect(a).toHaveLength(2);
    expect(a[0]).toMatchObject({ text: "Traceback (most recent call last):", source: "ocr", windowId: "w" });
    expect(a[0].bbox).toEqual([0, 0, 1, 0.2]);
    expect(a[0].confidence).toBeCloseTo(0.91);
    expect(a[1].text).not.toContain("abc");
    await ocr.dispose();
    expect(worker.terminate).toHaveBeenCalled();
  });
});

// ----------------------------------------------------------------- parser

describe("screen parser", () => {
  it.each([
    ["Traceback (most recent call last):", true],
    ["Traceback(mostrecentcalllast):", true],
    ["ModuleNotFoundError: No module named 'x'", true],
    ["src/app.ts:4:2 - error TS2322: Type", true],
    ["FAILED tests/test_a.py::test_b - assert 1 == 2", true],
    ["1failed,6passedin0.81s", true],
    ["3 passed in 0.12s", false],
    ["def handle_error(e):", false],
    ["Run tests", false],
  ])("looksLikeError(%s) = %s", (text, expected) => {
    expect(looksLikeError(text)).toBe(expected);
  });

  it("extracts file references", () => {
    let refs = extractFileRefs('File "C:\\dev\\app\\main.py", line 12, in run');
    expect(refs[0]).toMatchObject({ path: "C:\\dev\\app\\main.py", line: 12 });
    expect(refs).toHaveLength(1);
    refs = extractFileRefs("src/utils.py:40:7: E501 line too long");
    expect(refs[0]).toMatchObject({ path: "src/utils.py", line: 40, column: 7 });
    refs = extractFileRefs("Program.cs(12,5): error CS1002");
    expect(refs[0]).toMatchObject({ path: "Program.cs", line: 12, column: 5 });
    expect(extractFileRefs("see https://example.org/guide/index.html")).toEqual([]);
    expect(extractFileRefs("nothing here")).toEqual([]);
    expect(extractFileRefs("FAILED tests/test_a.py::test_b")[0].path).toBe("tests/test_a.py");
  });

  it("infers roles without pixels", () => {
    expect(inferRole(el("Cancel"))).toBe("button");
    expect(inferRole(el("continue"))).toBe("text");
    expect(inferRole(el("https://docs.python.org"))).toBe("link");
    expect(inferRole(el("x.py:3"))).toBe("file_ref");
    expect(inferRole(el("ValueError: bad"))).toBe("error");
  });

  it("parses a terminal traceback in reading order", () => {
    const els = [
      el("FAILED tests/test_app.py::test_add - AssertionError", [0, 0.5, 0.8, 0.52]),
      el("Traceback (most recent call last):", [0, 0.1, 0.5, 0.12]),
      el('  File "C:\\dev\\app\\main.py", line 12, in run', [0, 0.2, 0.6, 0.22]),
      el("   ", [0, 0.3, 0.1, 0.32]),
      el("OK", [0.9, 0.9, 0.95, 0.92]),
      el("hello world", [0.5, 0.6, 0.7, 0.62], "document"),
      el("Settings", [0.9, 0.0, 1.0, 0.05], "menuitem"),
    ];
    const obs = parseScreen(WIN, els, 7, 5);
    expect(obs).toMatchObject({ version: 7, timestampMs: 5, window: WIN });
    expect(obs.elements).toHaveLength(6);
    const ys = obs.elements.map((e) => e.bbox[1]);
    expect(ys).toEqual([...ys].sort((a, b) => a - b));
    const roles = Object.fromEntries(obs.elements.map((e) => [e.text, e.uiRole]));
    expect(roles["Settings"]).toBe("menuitem");
    expect(roles["hello world"]).toBe("document");
    expect(roles["OK"]).toBe("button");
    expect(roles['  File "C:\\dev\\app\\main.py", line 12, in run']).toBe("file_ref");
    const errors = findErrors(obs).map((e) => e.text);
    expect(errors.some((t) => t.startsWith("Traceback"))).toBe(true);
    expect(errors.some((t) => t.startsWith("FAILED"))).toBe(true);
    const refs = findFileRefs(obs);
    expect(refs.some((r) => r.path.startsWith("C:\\dev\\") && r.line === 12)).toBe(true);
    expect(refs.some((r) => r.path.startsWith("tests/test_"))).toBe(true);
    expect(refs.every((r) => r.element !== null)).toBe(true);
    expect(findButtons(obs).map((e) => e.text).sort()).toEqual(["OK", "Settings"]);
  });

  it("iou", () => {
    expect(iou([0, 0, 1, 1], [0, 0, 1, 1])).toBe(1);
    expect(iou([0, 0, 1, 1], [2, 2, 3, 3])).toBe(0);
  });
});

// ----------------------------------------------------------------- preprocess / featurize

describe("preprocess", () => {
  it("letterbox geometry round-trips boxes", () => {
    const tf = letterboxGeometry([500, 300], 64);
    expect(tf.newSize).toEqual([64, 38]);
    const box: [number, number, number, number] = [0.1, 0.2, 0.5, 0.9];
    dstToSrc(tf, srcToDst(tf, box)).forEach((v, i) => expect(v).toBeCloseTo(box[i], 5));
  });

  it("imageToPixels letterboxes, normalizes and pads black", () => {
    const px = imageToPixels(img(500, 300, 128), 64);
    expect(px).toHaveLength(64 * 64 * 3);
    expect(px.every(Number.isFinite)).toBe(true);
    const centre = ((32 * 64) + 32) * 3;
    expect(px[centre]).toBeCloseTo((128 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 4);
    expect(px[0]).toBeCloseTo((0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], 4); // top padding row
    expect(() => imageToPixels({ data: new Uint8ClampedArray(3), width: 4, height: 4 } as ImageData, 8)).toThrow();
  });

  it("changeRatio", () => {
    const a = img(320, 200, 10);
    const b = img(320, 200, 10);
    for (let y = 0; y < 100; y++) for (let x = 0; x < 320; x++) b.data[(y * 320 + x) * 4] = 255;
    expect(changeRatio(a, a)).toBe(0);
    expect(changeRatio(a, b)).toBeGreaterThan(0.01);
    expect(changeRatio(null, a)).toBe(1);
    expect(changeRatio(a, img(10, 10, 0))).toBe(1);
    expect(changeRatio(img(7, 5, 0), img(7, 5, 50))).toBe(1);
  });
});

describe("featurize", () => {
  const cfg = { ...DEFAULT_CONFIG.model, maxOcrTokens: 32, vocabSize: 512 };
  const obs = (elements: OCRElement[]): ScreenObservation => ({ window: WIN, elements, version: 1, timestampMs: 0 });

  it("one entry per word with the element's box; padding is a suffix", () => {
    const { ids, boxes, mask } = observationToArrays(obs([el("Run tests", [0.1, 0.2, 0.3, 0.4])]), cfg);
    expect(ids).toHaveLength(32);
    expect(boxes).toHaveLength(128);
    expect(Array.from(mask.slice(0, 2))).toEqual([0, 0]);
    expect(mask.slice(2).every((m) => m === 1)).toBe(true);
    expect(Array.from(boxes.slice(0, 4))).toEqual(Array.from(new Float32Array([0.1, 0.2, 0.3, 0.4])));
    expect(Array.from(boxes.slice(4, 8))).toEqual(Array.from(boxes.slice(0, 4)));
    expect(ids.slice(2).every((i) => i === PAD_ID)).toBe(true);
    expect(boxes.slice(8).every((v) => v === 0)).toBe(true);
    expect(ids.every((i) => i < 512)).toBe(true);
  });

  it("empty or null observations are all padding", () => {
    expect(observationToArrays(null, cfg).mask.every((m) => m === 1)).toBe(true);
    expect(observationToArrays(obs([]), cfg).ids.every((i) => i === 0)).toBe(true);
  });

  it("truncates and puts errors first", () => {
    const filler = Array.from({ length: 20 }, (_, i) => el(`plain words line ${i}`, [0, i / 40, 1, i / 40 + 0.01], "text"));
    const err = el("ValueError: bad thing happened", [0, 0.9, 1, 0.91], "error");
    const { ids, mask } = observationToArrays(obs([...filler, err]), cfg);
    expect(mask.every((m) => m === 0)).toBe(true);
    expect(Array.from(ids.slice(0, 3))).toEqual(encode("ValueError: bad", 512));
    const plain = observationToArrays(obs([...filler, err]), cfg, false);
    expect(Array.from(plain.ids.slice(0, 3))).toEqual(encode("plain words line", 512));
  });

  it("app buckets are stable and case-insensitive", () => {
    expect(appBucket("Code", 64)).toBe(appBucket("code", 64));
    expect(appBucket("", 64)).toBeGreaterThanOrEqual(0);
    expect(appBucket("chrome", 64)).toBeLessThan(64);
  });
});
