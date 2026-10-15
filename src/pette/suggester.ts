/**
 * Pette's hub (port of pet/soma.py's Suggester): SOMA's suggestions in, cards and "Do it" out.
 *
 *   * polls the latest suggestion every second and, while Pette is free, shows it as a glass card:
 *     Open or Copy / Not now, with a bar for the 15 s it stays;
 *   * click Pette / the hotkey -> SOMA looks at the screen now; settings -> add or remove folders;
 *   * "Do it" (or a pinch) -> open the file SOMA pointed at in Pette's viewer, or copy the text. Nothing else;
 *   * accepted / Not now / ignored go back to SOMA (its online ranker) and into Pette's fact log (kind +
 *     outcome, never the text), and kinds you keep putting off are muted (`WorldModel.mutedKinds`).
 *
 * Framework-free: it drives a `PetteController` and reports through `events`. Uses the global timers, so
 * tests run it under `vi.useFakeTimers()`.
 */
import type { FolderInfo, FoldersPayload, SomaService, SuggestionPayload } from "@/soma/service";
import { FeedbackOutcome } from "@/soma/types";
import * as actions from "./actions";
import type { Outcome, PetteStatus, Proposal } from "./contracts";
import type { OpenedFile, PetteController } from "./controller";
import { Emitter } from "./emitter";
import type { OfferLog } from "./world/store";

/** SOMA steps every >= 2 s, so a 1 s poll sees every shown suggestion. */
export const POLL_S = 1;
/** An unanswered card goes away (and counts as "ignored", never as "Not now"). */
export const SHOW_S = 15;
/** After "look now", give up waiting for a suggestion. */
export const LOOK_TIMEOUT_S = 12;
/** Consecutive failed polls (~20 s) before saying SOMA is offline. */
export const REPORT_AFTER = 20;
export const TICK_MS = 250;
export const INDEX_POLL_MS = 1500;

export const LABELS: Record<string, string> = {
  EXPLAIN_ERROR: "Explain error", OPEN_RELEVANT_FILE: "Open file", COMPARE_WITH_PRIOR_WORK: "Your files",
  SUGGEST_NEXT_STEP: "Next step", ASK_CLARIFYING_QUESTION: "Question",
};

/** Fact-log kind for a suggestion: its template, else its action (never its text). */
export function kindOf(s: SuggestionPayload): string {
  return "soma:" + (s.template || s.action || "suggestion").toLowerCase();
}

/** The file "Do it" opens: the on-screen file SOMA matched (with its line), or the indexed file behind a memory. */
export function opener(s: SuggestionPayload): [ref: string, line: number | null] | null {
  if (s.file) {
    const line = s.line;
    return [s.file, typeof line === "number" && Number.isInteger(line) && line > 0 ? line : null];
  }
  if (s.memorySource) return [s.memorySource, null];
  return null;
}

let nextId = 0;
const proposalId = () => `card_${Date.now().toString(36)}_${(++nextId).toString(36)}`;
const message = (e: unknown) => (e instanceof Error && e.message ? e.message : "SOMA is unreachable.");

export interface SuggesterEvents {
  status: [state: PetteStatus, detail: string];
  foldersChanged: [folders: FolderInfo[]];
  enabledChanged: [on: boolean];
}

export interface SuggesterOptions {
  /** Pette's fact log (facts only). */
  memory?: OfferLog | null;
  /** The world model's muting rule. */
  world?: { mutedKinds(): Set<string> } | null;
  pollS?: number;
  showS?: number;
  /** "Do it" for a file: defaults to `actions.openFile` into the controller's viewer. */
  openFile?: (ref: string, line: number | null) => Promise<string>;
  /** "Do it" for text: defaults to the clipboard. */
  copyText?: (text: string) => Promise<string>;
}

export class Suggester {
  readonly events = new Emitter<SuggesterEvents>();
  enabled = true;
  state: PetteStatus = "idle";
  folders: FolderInfo[] = [];
  /** Proposal id -> SOMA suggestion. */
  readonly shown = new Map<string, SuggestionPayload>();

  private readonly memory: OfferLog | null;
  private readonly world: { mutedKinds(): Set<string> } | null;
  private readonly pollMs: number;
  private readonly showS: number;
  private readonly openFileFn: (ref: string, line: number | null) => Promise<string>;
  private readonly copyFn: (text: string) => Promise<string>;
  /** SOMA suggestion ids already handled. */
  private readonly seen = new Set<string>();
  private card: string | null = null;
  private looking = false;
  private failures = 0;
  /** Wall-clock end of the card: hidden tabs throttle timers, so counting ticks drifts. */
  private deadline = 0;
  private running = false;
  private generation = 0;
  private polling = false;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private tick: ReturnType<typeof setInterval> | null = null;
  private lookTimeout: ReturnType<typeof setTimeout> | null = null;
  private indexing: ReturnType<typeof setInterval> | null = null;
  private indexingName = "";
  private readonly unsubscribe: (() => void)[];

  constructor(private readonly ui: PetteController, private readonly soma: SomaService, opts: SuggesterOptions = {}) {
    this.memory = opts.memory ?? null;
    this.world = opts.world ?? null;
    this.pollMs = (opts.pollS ?? POLL_S) * 1000;
    this.showS = opts.showS ?? SHOW_S;
    this.openFileFn = opts.openFile
      ?? ((ref, line) => actions.openFile(soma, (f: OpenedFile) => ui.showFile(f), ref, line));
    this.copyFn = opts.copyText ?? ((text) => actions.copyText(text));
    const ev = ui.events;
    this.unsubscribe = [
      ev.on("lookRequested", () => this.lookNow()),
      ev.on("approved", (pid) => void this.accepted(pid)),
      ev.on("cancelled", (pid) => this.dismissed(pid)),
      ev.on("pinchOnProposal", (pid) => this.pinched(pid)),
      ev.on("pauseToggled", (paused) => this.paused(paused)),
    ];
  }

  // --- helpers --------------------------------------------------------------
  private set(state: PetteStatus, detail = "") {
    this.state = state;
    this.ui.setStatus(state, detail);
    this.events.emit("status", state, detail);
  }

  /** Fire-and-forget SOMA call; errors go to `onErr` (or are dropped). */
  private call<T>(work: () => Promise<T>, onOk?: (r: T) => void, onErr?: (msg: string) => void) {
    let p: Promise<T>;
    try {
      p = work();
    } catch (e) {
      p = Promise.reject(e);
    }
    p.then((r) => onOk?.(r), (e) => onErr?.(message(e)));
  }

  // --- lifecycle --------------------------------------------------------------
  start(): void {
    if (this.running) return;
    this.running = true;
    const generation = ++this.generation; // a stop/start while a poll is in flight must not leave two loops
    const live = () => this.running && this.generation === generation;
    const loop = async () => {
      if (!live()) return;
      await this.pollOnce();
      if (live()) this.pollTimer = setTimeout(loop, this.pollMs);
    };
    void loop();
    this.refreshFolders();
  }

  stop(): void {
    this.running = false;
    for (const t of [this.pollTimer, this.lookTimeout]) if (t !== null) clearTimeout(t);
    for (const t of [this.tick, this.indexing]) if (t !== null) clearInterval(t);
    this.pollTimer = this.lookTimeout = this.tick = this.indexing = null;
  }

  /** Stop and detach from the controller. */
  dispose(): void {
    this.stop();
    for (const off of this.unsubscribe) off();
  }

  setEnabled(on: boolean): void {
    if (on === this.enabled) return;
    this.enabled = on;
    if (!on) {
      this.clearCard("ignored");
      this.stopLooking();
      this.set("idle");
    }
    this.call(() => (on ? this.soma.resume() : this.soma.pause()));
    this.events.emit("enabledChanged", on);
    this.ui.say(on ? "Suggestions on." : "Suggestions off. SOMA stopped looking at the screen.", { title: "SOMA", seconds: 4 });
  }

  toggle(): void {
    this.setEnabled(!this.enabled);
  }

  // --- polling ----------------------------------------------------------------
  /** One poll: show a fresh, valid, non-silent suggestion. */
  async pollOnce(): Promise<void> {
    if (!this.enabled || this.polling) return;
    this.polling = true;
    let s: SuggestionPayload;
    try {
      s = await this.soma.suggestion();
    } catch (e) {
      this.failures++;
      if (this.failures === REPORT_AFTER) this.onOffline(message(e));
      return;
    } finally {
      this.polling = false;
    }
    this.failures = 0;
    if (s.available && !s.silent && s.valid && s.text && s.id && !this.seen.has(s.id)) this.onSuggestion(s);
  }

  // --- look now ---------------------------------------------------------------
  private free(): boolean {
    const o = this.ui;
    return this.enabled && !o.paused && o.contextAllowed && o.bubbleEnabled && o.proposal === null;
  }

  /** Click / hotkey: SOMA looks at the screen now; the next suggestion shows, or "nothing" after a while. */
  lookNow(): void {
    if (this.looking || !this.free()) return;
    this.looking = true;
    this.set("looking");
    this.ui.say("Looking at your screen…", { title: "SOMA", seconds: null });
    this.lookTimeout = setTimeout(() => this.nothingFound(), LOOK_TIMEOUT_S * 1000);
    this.call(() => this.soma.trigger(), (ran) => {
      if (!ran) this.lookFailed("SOMA is paused or no screen is shared.");
    }, (msg) => this.lookFailed(msg));
  }

  private stopLooking() {
    this.looking = false;
    if (this.lookTimeout !== null) clearTimeout(this.lookTimeout);
    this.lookTimeout = null;
  }

  private lookFailed(msg: string) {
    if (!this.looking) return;
    this.stopLooking();
    this.set("failed", "SOMA offline");
    this.ui.say(msg, { title: "Couldn't look", seconds: 6 });
  }

  /** The look timed out. */
  nothingFound(): void {
    if (!this.looking) return;
    this.stopLooking();
    this.set("idle");
    this.ui.say("Nothing worth suggesting right now.", { title: "SOMA", seconds: 4 });
  }

  // --- cards ------------------------------------------------------------------
  private onSuggestion(s: SuggestionPayload) {
    const sid = s.id as string;
    if (this.seen.has(sid)) return;
    const asked = this.looking;
    if (asked) this.ui.dismiss(); // the "Looking…" bubble gives way to the card
    if (!this.free()) return; // busy: SOMA's cooldown decides when to try again
    this.seen.add(sid);
    this.stopLooking();
    if (this.world && !asked && this.world.mutedKinds().has(kindOf(s))) {
      this.feedback(s, "ignored", false); // you keep saying Not now to this kind
      return;
    }
    const label = LABELS[s.action ?? ""] ?? "Suggestion";
    const p: Proposal = {
      id: proposalId(), title: `SOMA suggests · ${label}`, text: s.text as string,
      actionLabel: opener(s) ? "Open" : "Copy", cancelLabel: "Not now", glow: true,
    };
    this.shown.set(p.id, s);
    this.card = p.id;
    this.set("suggesting");
    this.ui.propose(p);
    this.deadline = Date.now() + this.showS * 1000;
    this.ui.setProgress(1);
    this.stopTick();
    this.tick = setInterval(() => this.countDown(), TICK_MS);
    this.log(s, "shown");
  }

  private countDown() {
    const left = (this.deadline - Date.now()) / 1000;
    if (left > 1e-9) {
      this.ui.setProgress(left / this.showS);
      return;
    }
    this.clearCard("ignored");
    this.set("idle");
  }

  private stopTick() {
    if (this.tick !== null) clearInterval(this.tick);
    this.tick = null;
  }

  /** Pinch on a card = Do it (single-use, like the button). */
  private pinched(pid: string) {
    if (this.shown.has(pid) && this.ui.consume() === pid) void this.accepted(pid);
  }

  private async accepted(pid: string): Promise<void> {
    const s = this.shown.get(pid);
    if (!s) return;
    this.shown.delete(pid);
    this.card = null;
    this.stopTick();
    this.feedback(s, "accepted");
    const target = opener(s);
    let msg: string;
    try {
      msg = target ? await this.openFileFn(target[0], target[1]) : await this.copyFn(s.text ?? "");
    } catch (e) {
      const text = e instanceof actions.ActionError ? e.message : "That didn't work.";
      this.set("failed", target ? "Couldn't open" : "Couldn't copy");
      this.ui.say(text, { title: target ? "Couldn't open" : "Couldn't copy", seconds: 6 });
      return;
    }
    this.set("done");
    this.ui.say(msg, { title: "Done", seconds: 4 });
  }

  private dismissed(pid: string) {
    const s = this.shown.get(pid);
    if (!s) return;
    this.shown.delete(pid);
    this.card = null;
    this.stopTick();
    this.feedback(s, "rejected");
    this.ui.dismiss();
    this.set("dismissed");
  }

  private clearCard(outcome: Outcome) {
    const pid = this.card;
    this.card = null;
    this.stopTick();
    const s = pid ? this.shown.get(pid) : undefined;
    if (!pid || !s) return;
    this.shown.delete(pid);
    const up = this.ui.getState().bubble;
    if (this.ui.proposal?.id === pid || (up?.kind === "card" && up.proposal.id === pid)) this.ui.dismiss();
    this.feedback(s, outcome);
  }

  private paused(paused: boolean) {
    if (paused) {
      this.clearCard("ignored");
      this.stopLooking();
      this.set("idle");
    }
    if (this.enabled) this.call(() => (paused ? this.soma.pause() : this.soma.resume()));
  }

  private feedback(s: SuggestionPayload, outcome: Outcome, log = true) {
    if (log) this.log(s, outcome);
    const fb = { accepted: FeedbackOutcome.ACCEPTED, rejected: FeedbackOutcome.REJECTED, ignored: FeedbackOutcome.IGNORED }[outcome];
    this.call(() => this.soma.feedback(s.id as string, fb));
  }

  private log(s: SuggestionPayload, outcome: Outcome | "shown") {
    this.memory?.logOffer(kindOf(s), outcome); // facts only: which kind of suggestion and what you did
  }

  private onOffline(msg: string) {
    this.set("failed", "SOMA offline");
    this.ui.notify("SOMA", `${msg} Suggestions resume when it's back.`);
  }

  // --- folders ----------------------------------------------------------------
  refreshFolders(): void {
    this.call(() => this.soma.folders(), (d) => this.gotFolders(d));
  }

  private gotFolders(data: FoldersPayload) {
    this.folders = [...(data.folders ?? [])];
    this.events.emit("foldersChanged", this.folders);
    if (data.indexing) {
      if (this.indexing === null) this.indexing = setInterval(() => this.refreshFolders(), INDEX_POLL_MS);
      return;
    }
    if (this.indexing !== null) { // just finished
      clearInterval(this.indexing);
      this.indexing = null;
      const r = data.lastReport ?? {};
      const name = this.indexingName || "the folder";
      if (r.error) {
        this.ui.say(`Couldn't index ${name} (${r.error}).`, { title: "SOMA memory", seconds: 8 });
      } else {
        this.ui.say(`Indexed ${r.files ?? 0} file(s) from ${name}. SOMA uses them in suggestions now.`,
          { title: "SOMA memory", seconds: 6 });
      }
      if (this.state === "indexing") this.set("idle");
    }
  }

  /**
   * Settings -> "Add folder": SOMA shows the folder picker (call this from a click), records the grant and
   * indexes in the background. Cancelling the picker does nothing.
   */
  addFolder(handle?: FileSystemDirectoryHandle): Promise<void> {
    return new Promise((resolve) => {
      this.call(() => this.soma.addFolder(handle), (info) => {
        if (info) {
          this.indexingName = info.name;
          this.set("indexing", info.name);
          this.ui.say(`Indexing ${info.name}…`, { title: "SOMA memory", seconds: null });
          if (this.indexing === null) this.indexing = setInterval(() => this.refreshFolders(), INDEX_POLL_MS);
          this.refreshFolders();
        }
        resolve();
      }, (msg) => {
        this.set("failed", "Not indexed");
        this.ui.say(msg, { title: "Couldn't add folder", seconds: 8 });
        resolve();
      });
    });
  }

  /** Forget a folder and everything SOMA learned from it. */
  removeFolder(folderId: string): Promise<void> {
    const name = this.folders.find((f) => f.id === folderId)?.name ?? "the folder";
    return new Promise((resolve) => {
      this.call(() => this.soma.removeFolder(folderId), (n) => {
        this.ui.say(`Removed ${name}; SOMA forgot ${n} file(s) from it.`, { title: "SOMA memory", seconds: 6 });
        this.refreshFolders();
        resolve();
      }, (msg) => {
        this.ui.say(msg, { title: "Couldn't remove folder", seconds: 6 });
        resolve();
      });
    });
  }
}
