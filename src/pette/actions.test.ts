// Port of tests/test_actions.py: "Do it" never runs anything; it opens a granted file in Pette's viewer or copies text.
import { ActionError, baseName, copyText, downloadBlob, fileKind, openFile } from "./actions";
import type { OpenedFile } from "./controller";
import { FakeSoma } from "./fakeSoma";

it("opens a text file in the viewer at the line", async () => {
  const soma = new FakeSoma();
  const shown: OpenedFile[] = [];
  const msg = await openFile(soma, (f) => shown.push(f), "sandbox/calc.py", 7);
  expect(msg).toBe("Opened calc.py at line 7.");
  expect(shown).toHaveLength(1);
  expect(shown[0].name).toBe("calc.py");
  expect(shown[0].line).toBe(7);
  expect(shown[0].text).toContain("def total");
  expect(fileKind(shown[0])).toBe("text");
});

it("ignores a line that is not a positive integer", async () => {
  const shown: OpenedFile[] = [];
  await openFile(new FakeSoma(), (f) => shown.push(f), "sandbox/calc.py", 0);
  expect(shown[0].line).toBeNull();
});

it("shows images and offers anything else as a download only", async () => {
  const soma = new FakeSoma();
  const png = new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: "image/png" });
  const exe = new Blob([new Uint8Array([0x4d, 0x5a])]);
  soma.readFile = async (ref) =>
    ref.endsWith(".png") ? { name: "shot.png", text: null, blob: png } : { name: "tool.exe", text: null, blob: exe };
  const shown: OpenedFile[] = [];
  expect(await openFile(soma, (f) => shown.push(f), "f/shot.png", 3)).toBe("Opened shot.png.");
  expect(fileKind(shown[0])).toBe("image");
  expect(shown[0].line).toBeNull();
  expect(await openFile(soma, (f) => shown.push(f), "f/tool.exe")).toContain("download");
  expect(fileKind(shown[1])).toBe("binary");
  expect(downloadBlob(exe).type).toBe("application/octet-stream"); // never rendered by the browser
});

it("reports a file that is gone or not granted", async () => {
  const soma = new FakeSoma();
  await expect(openFile(soma, () => undefined, "sandbox/gone.py")).rejects.toThrow(ActionError);
  await expect(openFile(soma, () => undefined, "sandbox/gone.py")).rejects.toThrow("gone.py isn't there any more.");
  soma.readFile = async () => {
    throw new Error("boom");
  };
  await expect(openFile(soma, () => undefined, "x/y.md")).rejects.toThrow("isn't there");
});

it("copies text, and reports a blocked clipboard", async () => {
  const got: string[] = [];
  expect(await copyText("hello", async (t) => void got.push(t))).toContain("Copied");
  expect(got).toEqual(["hello"]);
  await expect(copyText("x", async () => {
    throw new Error("denied");
  })).rejects.toThrow(ActionError);
});

it("baseName", () => {
  expect(baseName("folder_1/a/b/calc.py")).toBe("calc.py");
  expect(baseName("a\\b.md")).toBe("b.md");
});
