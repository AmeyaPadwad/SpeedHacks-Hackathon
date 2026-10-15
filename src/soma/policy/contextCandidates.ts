/**
 * App-context next steps, so ordinary screens have something worth ranking (port of
 * SOMA/soma/policy/context_candidates.py).
 *
 * `generateCandidates` only proposes actions for visible errors, authorized file references and retrieved
 * memories, so on most real screens (an editor with clean code, a terminal, a web page) it proposes nothing.
 * With `PolicyConfig.contextTemplates` the runtime appends these fixed templates, chosen from the shared
 * app's kind and a few cheap screen signals. The model still ranks them against NO_SUGGESTION and the gate
 * still applies.
 *
 * Same safety rules as the generator: the text is a fixed template, screen text appears only as a short
 * sanitized quote, injection-like titles are dropped, nothing here executes. `Candidate.template` names the
 * template so Pette and the feedback log can tell them apart.
 */
import { ActionType, type Candidate, candidateKey, makeCandidate, type ScreenObservation } from "../types";
import { capText, detectRole, isSuspicious, quote, ROLE_ERROR, sanitizeSnippet } from "./candidateGenerator";

export const WRITE_TESTS = "write_tests";
export const REVIEW_CHANGES = "review_changes";
export const COMMIT_WORK = "commit_work";
export const TAKE_NOTES = "take_notes";
export const TEMPLATES = [WRITE_TESTS, REVIEW_CHANGES, COMMIT_WORK, TAKE_NOTES] as const;

export type AppKind = "editor" | "terminal" | "browser" | "";

export const EDITORS = new Set(["code", "code - insiders", "cursor", "windsurf", "pycharm64", "idea64", "webstorm64",
  "devenv", "sublime_text", "notepad++", "zed", "rider64", "clion64"]);
export const TERMINALS = new Set(["windowsterminal", "openconsole", "cmd", "powershell", "pwsh", "wezterm-gui",
  "alacritty", "mintty", "conhost", "tabby", "hyper", "warp"]);
export const BROWSERS = new Set(["chrome", "msedge", "firefox", "brave", "arc", "opera", "vivaldi", "comet",
  "chromium", "thorium", "zen", "librewolf", "waterfox", "floorp"]);

const FILE_RE = /[\w.-]{1,80}\.[A-Za-z][A-Za-z0-9]{0,7}/;
const TESTS_PASS_RE = /\b\d+\s+passed\b|\btests?\s+passed\b|\ball tests pass/i;
const BROWSER_SUFFIX_RE =
  /\s+[-—|]\s+(google chrome|microsoft​? edge|mozilla firefox|brave|arc|opera|vivaldi|comet|chromium|zen).*$/i;

/** "editor" | "terminal" | "browser" | "" from a canonical app name (`"code"`, `"Code.exe"` also works). */
export function appKind(appName: string | null | undefined): AppKind {
  let name = (appName ?? "").trim().toLowerCase();
  if (name.endsWith(".exe")) name = name.slice(0, -4);
  if (EDITORS.has(name)) return "editor";
  if (TERMINALS.has(name)) return "terminal";
  if (BROWSERS.has(name)) return "browser";
  return "";
}

/** The file name an editor title shows first (`"● main.py - Pette - Visual Studio Code"` -> `main.py`). */
export function editorFile(title: string | null | undefined): string | null {
  const t = (title ?? "").trim();
  const m0 = /\s+[-—]\s+/.exec(t);
  let head = m0 ? t.slice(0, m0.index) : t;
  head = head.replace(/^[●*• ]+/, "");
  const m = FILE_RE.exec(head);
  return m ? m[0] : null;
}

/** A browser tab title without the browser's name, if it is safe to quote. */
export function pageTopic(title: string | null | undefined): string | null {
  const raw = (title ?? "").trim().replace(BROWSER_SUFFIX_RE, "");
  const topic = sanitizeSnippet(raw, 60);
  if (topic.length < 3 || isSuspicious(topic) || ["new tab", "[redacted]"].includes(topic.toLowerCase())) return null;
  return topic;
}

/** Template candidates for the shared app (may be empty). */
export function contextCandidates(obs: ScreenObservation): Candidate[] {
  const kind = appKind(obs.window.appName);
  if (!kind) return [];
  const texts = obs.elements.map((e) => e.text).filter((t) => !isSuspicious(t));
  const hasError = obs.elements.some((e) => detectRole(e) === ROLE_ERROR);
  const out: Candidate[] = [];

  if (kind === "editor") {
    const f = editorFile(obs.window.title);
    const label = f ? quote(f, 60) : "this file";
    out.push(makeCandidate({
      action: ActionType.SUGGEST_NEXT_STEP,
      text: capText(`Review your changes to ${label} before you commit`),
      evidence: [`editing ${label}`], template: REVIEW_CHANGES,
    }));
    if (!hasError) {
      out.push(makeCandidate({
        action: ActionType.SUGGEST_NEXT_STEP,
        text: capText(`Write a test for the code in ${label}`),
        evidence: [`editing ${label}`], template: WRITE_TESTS,
      }));
    }
  }

  if (kind === "terminal" && !hasError && texts.some((t) => TESTS_PASS_RE.test(t))) {
    out.push(makeCandidate({
      action: ActionType.SUGGEST_NEXT_STEP,
      text: "The tests pass: commit your work",
      evidence: ["test run passed"], template: COMMIT_WORK,
    }));
  }

  if (kind === "browser") {
    const topic = pageTopic(obs.window.title);
    if (topic) {
      out.push(makeCandidate({
        action: ActionType.SUGGEST_NEXT_STEP,
        text: capText(`Save what you learn from ${quote(topic, 60)} to your notes`),
        evidence: [topic], template: TAKE_NOTES,
      }));
    }
  }
  return out;
}

/** `candidates` plus context templates, de-duplicated by key and bounded. */
export function addContextCandidates(obs: ScreenObservation, candidates: readonly Candidate[],
  maxCandidates: number): Candidate[] {
  const out = candidates.slice(0, Math.max(0, maxCandidates));
  const keys = new Set(out.map(candidateKey));
  for (const c of contextCandidates(obs)) {
    if (out.length >= maxCandidates) break;
    const k = candidateKey(c);
    if (!keys.has(k)) {
      keys.add(k);
      out.push(c);
    }
  }
  return out;
}
