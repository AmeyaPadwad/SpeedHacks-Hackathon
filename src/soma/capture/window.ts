/**
 * Which app is being shared, and may it be captured at all (port of soma/capture/window.py and the
 * exclusion half of soma/capture/redaction.py).
 *
 * A web page cannot see the foreground window. `inferWindow` guesses a canonical, process-like app name
 * that `contextCandidates.appKind` recognises ("code", "windowsterminal", "chrome", ... or "") from the
 * `getDisplayMedia` surface, the track label (Chrome labels a shared window with its title) and cheap
 * on-screen text cues (VS Code's menu and status bar, shell prompts, an address bar). Nothing here keeps
 * pixels or text beyond the returned title line, which callers redact before use.
 */
import type { CaptureConfig } from "../config";
import type { OCRElement, WindowInfo } from "../types";
import { fnv1a } from "../perception/tokenizer";

// ----------------------------------------------------------------- exclusion

/** Process-name fragments that are always excluded (case-insensitive substring). */
export const BUILTIN_EXCLUDED_APPS: readonly string[] = [
  // password managers
  "1password", "keepass", "keepassxc", "bitwarden", "lastpass", "dashlane",
  "enpass", "keeper", "roboform", "nordpass", "proton pass",
  // authentication prompts / secure desktop
  "credentialuibroker", "consent.exe", "logonui", "lockapp", "credwiz",
];

/** Window-title fragments that are always excluded, by reason (case-insensitive substring). */
export const BUILTIN_EXCLUDED_TITLES: Readonly<Record<string, readonly string[]>> = {
  "private-browsing": ["inprivate", "incognito", "private browsing", "private window"],
  "password-manager": ["1password", "bitwarden", "keepass", "lastpass", "dashlane", "password manager"],
  banking: ["online banking", "bank of america", "chase.com", "wells fargo", "paypal", "barclays", "hsbc",
    "credit union"],
  "auth-prompt": ["windows security", "user account control", "sign in", "log in", "login", "two-factor",
    "2-step verification", "verification code", "enter your password", "authenticator"],
};

/**
 * The reason `win` must never be captured (`"excluded-app:bitwarden"`, `"private-browsing:incognito"`,
 * `"user-excluded-title:payroll"`, ...), or null when it may be. Built-in rules come first, then
 * `cfg.excludedApps` / `cfg.excludedTitlePatterns`.
 */
export function isExcluded(win: WindowInfo, cfg: CaptureConfig): string | null {
  const app = (win.appName ?? "").toLowerCase();
  const title = (win.title ?? "").toLowerCase();
  for (const frag of BUILTIN_EXCLUDED_APPS) if (frag && app.includes(frag)) return `excluded-app:${frag}`;
  for (const raw of cfg.excludedApps) {
    const f = (raw ?? "").trim().toLowerCase();
    if (f && app.includes(f)) return `user-excluded-app:${f}`;
  }
  for (const [reason, frags] of Object.entries(BUILTIN_EXCLUDED_TITLES)) {
    for (const frag of frags) if (title.includes(frag)) return `${reason}:${frag}`;
  }
  for (const raw of cfg.excludedTitlePatterns) {
    const f = (raw ?? "").trim().toLowerCase();
    if (f && title.includes(f)) return `user-excluded-title:${f}`;
  }
  return null;
}

// ----------------------------------------------------------------- app inference

/** Window-title suffixes -> canonical app names (the ones `appKind` knows). */
const TITLE_APPS: [RegExp, string][] = [
  [/visual studio code(?:\s*-\s*insiders)?\s*$/i, "code"],
  [/(?:^|\s[-—]\s)cursor\s*$/i, "cursor"],
  [/(?:^|\s[-—]\s)windsurf\s*$/i, "windsurf"],
  [/\bpycharm\b/i, "pycharm64"],
  [/\bintellij idea\b/i, "idea64"],
  [/\bwebstorm\b/i, "webstorm64"],
  [/\bsublime text\s*$/i, "sublime_text"],
  [/notepad\+\+\s*$/i, "notepad++"],
  [/google chrome\s*$/i, "chrome"],
  [/microsoft​?\s*edge\s*$/i, "msedge"],
  [/mozilla firefox\s*$/i, "firefox"],
  [/(?:^|\s[-—]\s)brave\s*$/i, "brave"],
  [/(?:^|\s[-—]\s)opera\s*$/i, "opera"],
  [/(?:^|\s[-—]\s)vivaldi\s*$/i, "vivaldi"],
  [/\bpowershell\b|\bpwsh\b/i, "powershell"],
  [/command prompt|\bcmd\.exe\s*$/i, "cmd"],
  [/^[\w.-]+@[\w.-]+:|\bterminal\s*$/i, "windowsterminal"],
];

/** Labels that are not a window title ("screen:0:0", "web-contents-media-stream://...", "Entire screen"). */
const NON_TITLE_LABEL = /^(?:screen|window|web-contents-media-stream)[:/]|^(?:entire screen|screen\s*\d*|monitor\s*\d*)$/i;

const MENU_WORDS = new Set(["file", "edit", "selection", "view", "go", "run", "terminal", "help"]);
const EDITOR_CUES: RegExp[] = [/\bLn \d+,\s*Col \d+/, /^EXPLORER\b/, /\bPROBLEMS\b.*\bOUTPUT\b|\bDEBUG CONSOLE\b/,
  /\bSpaces: \d+\b|\bUTF-8\b|\bCRLF\b/];
const PROMPTS: RegExp[] = [
  /^PS [A-Za-z]:\\/, /^[A-Za-z]:\\[^<>|]*>/, /^[\w.-]+@[\w.-]+[:\s].*[$#]\s?/, /^\$\s/, /^>>>\s?/,
  /^Microsoft Windows \[Version/, /^Windows PowerShell\b/, /^Copyright \(C\) Microsoft Corporation/,
];
const ADDRESS = /^(?:https?:\/\/|www\.)\S+|^[\w-]+(?:\.[\w-]+)+\/\S*/i;

/** Canonical name of the browser Pette itself runs in, from its user agent. */
export function hostBrowser(userAgent: string = globalThis.navigator?.userAgent ?? ""): string {
  if (/\bEdg\//.test(userAgent)) return "msedge";
  if (/\bOPR\//.test(userAgent)) return "opera";
  if (/\bFirefox\//.test(userAgent)) return "firefox";
  return "chrome";
}

/** App name for a window title, or "" when no suffix is recognised. */
export function appFromTitle(title: string): string {
  for (const [re, app] of TITLE_APPS) if (re.test(title)) return app;
  return "";
}

const isTop = (e: OCRElement, limit: number): boolean => e.bbox[1] < limit;

/**
 * Guess the shared window's app and title.
 *
 * Order: a track label that is a window title (Chrome window shares), then a title line read at the top
 * of the screen, then on-screen cues (editor beats terminal beats address bar, since VS Code shows an
 * integrated terminal), then the surface (`"browser"` means a tab of Pette's own browser). A top line that
 * names a password manager or private browsing becomes the title, so `isExcluded` fires.
 */
export function inferWindow(surface: string, trackLabel: string, elements: OCRElement[], userAgent?: string): WindowInfo {
  const label = (trackLabel ?? "").trim();
  const windowId = `${surface || "screen"}:${fnv1a(label).toString(16)}`;
  const make = (appName: string, title: string): WindowInfo => ({ appName, title, windowId, surface: surface ?? "" });

  const labelTitle = label && !NON_TITLE_LABEL.test(label) ? label : "";
  const top = elements.filter((e) => isTop(e, 0.08));

  // Sensitive windows first: never let a weaker cue hide them.
  for (const e of top) {
    const t = e.text.toLowerCase();
    const app = BUILTIN_EXCLUDED_APPS.find((f) => t.includes(f));
    if (app) return make(app, labelTitle || e.text.trim());
    if (BUILTIN_EXCLUDED_TITLES["private-browsing"].some((f) => t.includes(f))) {
      return make(appFromTitle(labelTitle), labelTitle ? `${labelTitle} ${e.text.trim()}` : e.text.trim());
    }
  }

  if (labelTitle) {
    const app = appFromTitle(labelTitle);
    if (app) return make(app, labelTitle);
  }
  for (const e of top) {
    const app = appFromTitle(e.text.trim());
    if (app) return make(app, e.text.trim());
  }

  let editor = 0;
  let terminal = 0;
  let browser = 0;
  const menu = new Set<string>();
  for (const e of top) for (const w of e.text.split(/\s+/)) if (MENU_WORDS.has(w.toLowerCase())) menu.add(w.toLowerCase());
  if (menu.size >= 4) editor += 3;
  for (const e of elements) {
    const t = e.text.trim();
    if (EDITOR_CUES.some((re) => re.test(t))) editor += 1;
    if (PROMPTS.some((re) => re.test(t))) terminal += 2;
    if (isTop(e, 0.15) && ADDRESS.test(t)) browser += 3;
  }
  if (editor >= 3) return make("code", labelTitle);
  if (terminal >= 2 && terminal >= browser) return make("windowsterminal", labelTitle);
  if (browser >= 3 || surface === "browser") return make(hostBrowser(userAgent), labelTitle);
  return make("", labelTitle);
}
