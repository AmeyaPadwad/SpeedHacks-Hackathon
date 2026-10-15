import { describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG, type CaptureConfig } from "../config";
import type { OCRElement, ScreenObservation, WindowInfo } from "../types";
import { containsSecret, luhnValid, redactElements, redactObservation, redactText, sensitiveBoxes } from "./redaction";
import { ScreenCapturer } from "./screen";
import { hostBrowser, inferWindow, isExcluded } from "./window";

const el = (text: string, bbox: [number, number, number, number] = [0, 0.5, 0.5, 0.52]): OCRElement => ({
  text, bbox, confidence: 1, source: "ocr", uiRole: null, windowId: "w", timestampMs: 0,
});
const win = (appName: string, title = ""): WindowInfo => ({ appName, title, windowId: "w", surface: "window" });
const NO_USER: CaptureConfig = { ...DEFAULT_CONFIG.capture, excludedApps: [], excludedTitlePatterns: [] };

// ----------------------------------------------------------------- exclusion

describe("isExcluded", () => {
  it.each([
    ["1Password.exe", "Vault", "excluded-app"],
    ["KeePassXC.exe", "db.kdbx", "excluded-app"],
    ["bitwarden", "My vault", "excluded-app"],
    ["CredentialUIBroker.exe", "Windows Security", "excluded-app"],
    ["msedge", "New tab - [InPrivate] - Microsoft Edge", "private-browsing"],
    ["chrome", "New Tab - Google Chrome (Incognito)", "private-browsing"],
    ["firefox", "Mozilla Firefox Private Browsing", "private-browsing"],
    ["chrome", "Chase Online Banking - Google Chrome", "banking"],
    ["chrome", "PayPal: Summary", "banking"],
    ["chrome", "Sign in to GitHub", "auth-prompt"],
    ["chrome", "2-Step Verification", "auth-prompt"],
  ])("excludes %s / %s", (app, title, prefix) => {
    expect(isExcluded(win(app, title), NO_USER)?.startsWith(prefix)).toBe(true);
  });

  it("applies user lists case-insensitively", () => {
    const cfg = { ...NO_USER, excludedApps: ["SLACK"], excludedTitlePatterns: ["Payroll"] };
    expect(isExcluded(win("slack", "general"), cfg)).toBe("user-excluded-app:slack");
    expect(isExcluded(win("excel", "PAYROLL 2026.xlsx"), cfg)).toBe("user-excluded-title:payroll");
  });

  it("lets regular apps through", () => {
    for (const [a, t] of [["code", "main.py - SOMA - Visual Studio Code"], ["windowsterminal", "pytest"],
      ["chrome", "PyTorch documentation - Google Chrome"], ["", ""]]) {
      expect(isExcluded(win(a, t), DEFAULT_CONFIG.capture)).toBeNull();
    }
  });
});

// ----------------------------------------------------------------- inference

describe("inferWindow", () => {
  it("reads a window title from the track label", () => {
    const w = inferWindow("window", "main.py - Pette - Visual Studio Code", []);
    expect(w).toMatchObject({ appName: "code", title: "main.py - Pette - Visual Studio Code", surface: "window" });
    expect(inferWindow("window", "Docs - Google Chrome", []).appName).toBe("chrome");
    expect(inferWindow("window", "Windows PowerShell", []).appName).toBe("powershell");
    expect(w.windowId).not.toContain("main.py");
  });

  it("finds a title line at the top of a monitor share", () => {
    const w = inferWindow("monitor", "screen:0:0", [el("app.ts - web - Visual Studio Code", [0.3, 0.0, 0.7, 0.02]), el("const x = 1")]);
    expect(w.appName).toBe("code");
    expect(w.title).toBe("app.ts - web - Visual Studio Code");
  });

  it("recognises VS Code by its menu and status bar", () => {
    const els = [el("File Edit Selection View Go Run Terminal Help", [0, 0, 0.4, 0.02]), el("Ln 12, Col 5", [0.8, 0.98, 0.9, 1])];
    expect(inferWindow("monitor", "", els).appName).toBe("code");
  });

  it("recognises shells", () => {
    for (const prompt of ["PS C:\\Users\\me\\proj> pytest", "C:\\proj>dir", "me@box:~/proj$ ls", "$ npm test", ">>> import os"]) {
      expect(inferWindow("monitor", "", [el(prompt)]).appName).toBe("windowsterminal");
    }
  });

  it("recognises an address bar and browser tabs", () => {
    expect(inferWindow("monitor", "", [el("https://example.org/docs/intro", [0.1, 0.05, 0.5, 0.07])]).appName).toBe("chrome");
    expect(inferWindow("browser", "web-contents-media-stream://1:2", [], "Mozilla/5.0 Edg/120").appName).toBe("msedge");
    expect(hostBrowser("Mozilla/5.0 Firefox/120")).toBe("firefox");
  });

  it("unknown screens get an empty app name", () => {
    expect(inferWindow("monitor", "", [el("hello world")])).toMatchObject({ appName: "", title: "" });
  });

  it("surfaces password managers and private windows so they are excluded", () => {
    const pm = inferWindow("monitor", "", [el("Bitwarden", [0, 0, 0.2, 0.02])]);
    expect(isExcluded(pm, NO_USER)).toBe("excluded-app:bitwarden");
    const priv = inferWindow("monitor", "", [el("InPrivate", [0, 0, 0.2, 0.02])]);
    expect(isExcluded(priv, NO_USER)).toMatch(/^private-browsing/);
  });
});

// ----------------------------------------------------------------- redaction

const SECRETS: Record<string, string> = {
  openai: "sk-proj-abcdefghijklmnopqrstuvwx1234",
  openaiPlain: "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
  aws: "AKIAIOSFODNN7EXAMPLE",
  github: "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
  githubPat: "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz",
  jwt: "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
  slack: "xoxb-1234567890-abcdefghij",
  email: "jane.doe@example.com",
  card: "4111 1111 1111 1111",
  cardDashes: "5500-0000-0000-0004",
  hex: "a3f5c9e1b2d4f6a8c0e2b4d6f8a0c2e4",
  b64: "Zm9vYmFyQmF6UXV4MTIzNDU2Nzg5MGFiY2RlZmdoaWprbG1ub3A",
};

describe("redactText", () => {
  it.each(Object.entries(SECRETS))("masks %s", (_k, secret) => {
    const text = `value: ${secret} end`;
    const out = redactText(text);
    expect(out).not.toContain(secret);
    expect(out).toContain("[REDACTED:");
    expect(out.startsWith("value") && out.endsWith("end")).toBe(true);
    expect(containsSecret(text)).toBe(true);
  });

  it.each([
    ["password=hunter2", "hunter2"],
    ["PASSWORD: 'correct horse battery'", "correct horse battery"],
    ['api_key = "abc123"', "abc123"],
    ["export ACCESS_TOKEN=xyz987&next=1", "xyz987"],
    ['{"client_secret": "s3cr3t"}', "s3cr3t"],
  ])("masks key/value pair %s", (text, secret) => {
    const out = redactText(text);
    expect(out).not.toContain(secret);
    expect(out).toContain("[REDACTED:secret]");
  });

  it("keeps non-Luhn numbers and ordinary text", () => {
    expect(luhnValid("1234567890123456")).toBe(false);
    expect(redactText("order 1234 5678 9012 3456 shipped")).toBe("order 1234 5678 9012 3456 shipped");
    for (const t of ["Traceback (most recent call last):", 'File "app.py", line 12, in <module>',
      "FAILED tests/test_x.py::test_add - AssertionError", "Run tests", "def password_strength(pw):", ""]) {
      expect(redactText(t)).toBe(t);
    }
  });

  it("redacts elements and observations without touching the originals", () => {
    const els = [el("hello", [0, 0, 0.1, 0.1]), el("token=abc", [0.2, 0.2, 0.4, 0.3])];
    const out = redactElements(els);
    expect(out[0].text).toBe("hello");
    expect(out[1].text).not.toContain("abc");
    expect(els[1].text).toBe("token=abc");
    expect(sensitiveBoxes(els)).toEqual([[0.2, 0.2, 0.4, 0.3]]);
    expect(sensitiveBoxes(out)).toEqual([[0.2, 0.2, 0.4, 0.3]]);
    const obs: ScreenObservation = { window: win("code", "mail jane.doe@example.com"), elements: els, version: 1, timestampMs: 0 };
    const r = redactObservation(obs);
    expect(r.window.title).not.toContain("jane");
    expect(r.elements[1].text).not.toContain("abc");
    expect(obs.window.title).toContain("jane");
  });
});

// ----------------------------------------------------------------- capturer

function fakeImage(w: number, h: number, fill = 0): ImageData {
  return { data: new Uint8ClampedArray(w * h * 4).fill(fill), width: w, height: h, colorSpace: "srgb" } as ImageData;
}

function fakeStream(label = "main.py - Pette - Visual Studio Code", surface = "window") {
  const listeners: (() => void)[] = [];
  const track = {
    label, readyState: "live" as MediaStreamTrackState,
    getSettings: () => ({ displaySurface: surface }),
    stop: vi.fn(function (this: { readyState: string }) { track.readyState = "ended"; }),
    addEventListener: (_: string, fn: () => void) => listeners.push(fn),
  };
  const stream = { getVideoTracks: () => [track], getTracks: () => [track] } as unknown as MediaStream;
  const end = () => { track.readyState = "ended"; listeners.forEach((f) => f()); };
  return { stream, track, end };
}

function setup(images: ImageData[], opts: { decline?: boolean } = {}) {
  const s = fakeStream();
  let t = 0;
  const getDisplayMedia = vi.fn(async (o?: DisplayMediaStreamOptions) => {
    if (opts.decline) throw new DOMException("denied", "NotAllowedError");
    expect(o).toMatchObject({ audio: false, selfBrowserSurface: "exclude" });
    return s.stream;
  });
  let i = 0;
  const grabFrame = vi.fn(async () => images[Math.min(i++, images.length - 1)]);
  const cap = new ScreenCapturer(DEFAULT_CONFIG.capture, { mediaDevices: { getDisplayMedia }, now: () => t, grabFrame });
  return { cap, s, grabFrame, getDisplayMedia, tick: (ms: number) => { t += ms; } };
}

describe("ScreenCapturer", () => {
  it("never grabs without a granted stream", async () => {
    const { cap, grabFrame } = setup([fakeImage(8, 8)], { decline: true });
    expect(await cap.capture(true)).toBeNull();
    expect(cap.lastSkipReason).toBe("no-consent");
    expect(await cap.grant()).toBe(false);
    expect(cap.active).toBe(false);
    expect(await cap.capture(true)).toBeNull();
    expect(grabFrame).not.toHaveBeenCalled();
  });

  it("rate-limits, detects change and versions only meaningful frames", async () => {
    const a = fakeImage(64, 64, 10);
    const b = fakeImage(64, 64, 200);
    const { cap, tick } = setup([a, a, b, b]);
    expect(await cap.grant()).toBe(true);
    const f1 = await cap.capture();
    expect(f1).toMatchObject({ version: 1, changed: true, surface: "window", trackLabel: "main.py - Pette - Visual Studio Code" });
    tick(500);
    expect(await cap.capture()).toBeNull();
    expect(cap.lastSkipReason).toBe("rate-limited");
    tick(2000);
    expect(await cap.capture()).toMatchObject({ version: 1, changed: false });
    tick(2000);
    expect(await cap.capture()).toMatchObject({ version: 2, changed: true });
    expect(await cap.capture(true)).toMatchObject({ version: 3, changed: true }); // force: no rate limit, counts as change
  });

  it("tiny changes stay below the threshold", async () => {
    const a = fakeImage(256, 256, 50);
    const b = fakeImage(256, 256, 50);
    for (let i = 0; i < 4 * 4; i++) b.data[i] = 255; // a few pixels
    const { cap, tick } = setup([a, b]);
    await cap.grant();
    expect((await cap.capture())?.version).toBe(1);
    tick(5000);
    expect(await cap.capture()).toMatchObject({ version: 1, changed: false });
    expect(cap.lastChangeRatio).toBeLessThan(DEFAULT_CONFIG.capture.changeThreshold);
  });

  it("revoke releases tracks; the user stopping the share fires onEnded", async () => {
    const { cap, s } = setup([fakeImage(8, 8)]);
    await cap.grant();
    cap.revoke();
    expect(s.track.stop).toHaveBeenCalled();
    expect(cap.active).toBe(false);

    const t2 = setup([fakeImage(8, 8)]);
    const ended = vi.fn();
    t2.cap.onEnded = ended;
    await t2.cap.grant();
    t2.s.end();
    expect(ended).toHaveBeenCalledOnce();
    expect(t2.cap.active).toBe(false);
    expect(await t2.cap.capture(true)).toBeNull();
  });

  it("a grab error is a skip, not a crash", async () => {
    const s = fakeStream();
    const cap = new ScreenCapturer(DEFAULT_CONFIG.capture, {
      mediaDevices: { getDisplayMedia: async () => s.stream },
      grabFrame: async () => { throw new TypeError("boom"); },
    });
    await cap.grant();
    expect(await cap.capture(true)).toBeNull();
    expect(cap.lastSkipReason).toBe("grab-error:TypeError");
  });
});
