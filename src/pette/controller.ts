/**
 * Pette's on-screen state (port of pet/ui.py's Overlay + Bubble, without the widgets): the orb's status,
 * mood and camera picture, and the speech bubble's message or card. React reads it through
 * `useSyncExternalStore(controller.subscribe, controller.getState)`; the hub (`Suggester`) and the webcam
 * (`WatcherTarget`) drive it through the methods below.
 *
 * Change bubble content only through `say`, `propose`, `dismiss` and `setProgress`. A card's decision is
 * single-use: `consume()` returns the card's id once, and the buttons and a pinch both go through it.
 *
 * Suggestion text is untrusted: it is stored here as plain strings and rendered as text, never as HTML.
 */
import type { Mood, PetteStatus, PreviewFrame, Proposal, WatcherTarget } from "./contracts";
import { Emitter } from "./emitter";

/** Read-only suggestions kept for paging with point left / right. */
export const HISTORY_SIZE = 5;
/** A history page goes back to the live view after this long. */
export const HISTORY_S = 8;
/** A notice (the old tray balloon) stays this long. */
export const NOTICE_S = 5;

export interface Proposed {
  id: string;
  title: string;
  text: string;
}

/** What the speech bubble shows. `null` = hidden. */
export type BubbleView =
  | { kind: "message"; title: string; text: string }
  /** `live` is false once the decision was consumed (buttons disabled). */
  | { kind: "card"; proposal: Proposal; title: string; live: boolean }
  | { kind: "history"; title: string; text: string; index: number; total: number };

/** A file "Do it" opened in Pette's viewer. Kept in memory only while the viewer is open. */
export interface OpenedFile {
  name: string;
  /** Text content (text files), or null for binary files. */
  text: string | null;
  blob: Blob;
  line: number | null;
}

export interface Notice { id: number; title: string; text: string }

export interface PetteState {
  status: PetteStatus;
  detail: string;
  mood: Mood;
  paused: boolean;
  /** Ctrl+Alt+B: the speech bubble (and SOMA's cards) on or off. */
  bubbleEnabled: boolean;
  /** Kept for parity with the desktop focus gate; always true on the web unless an embedder hides Pette. */
  contextAllowed: boolean;
  away: boolean;
  eyesClosed: boolean;
  cameraOn: boolean;
  /** Ctrl+Alt+C: hide the camera picture, keep Pette (ring, moods, clicks). */
  feedVisible: boolean;
  /** Webcam gestures (pinch, point, pinch-and-drag) on or off; presence still works when off. */
  gesturesEnabled: boolean;
  gesture: string;
  worldVisible: boolean;
  bubble: BubbleView | null;
  /** Time left on a card (1 -> 0), drawn as a draining bar; null hides it. */
  progress: number | null;
  notice: Notice | null;
  file: OpenedFile | null;
}

export interface ControllerEvents {
  approved: [pid: string];
  cancelled: [pid: string];
  /** Click Pette / hotkey / settings: ask SOMA to look at the screen now. */
  lookRequested: [];
  pauseToggled: [paused: boolean];
  /** Pinch on a card = "Do it". */
  pinchOnProposal: [pid: string];
  bubbleToggled: [on: boolean];
  /** Pinch-and-drag: image heights moved, + = hand moved down. */
  scroll: [dy: number];
}

type Message = { title: string; text: string };

export class PetteController implements WatcherTarget {
  readonly events = new Emitter<ControllerEvents>();
  private state: PetteState = {
    status: "idle", detail: "", mood: "idle", paused: false, bubbleEnabled: true, contextAllowed: true,
    away: false, eyesClosed: false, cameraOn: false, feedVisible: true, gesturesEnabled: true, gesture: "", worldVisible: true,
    bubble: null, progress: null, notice: null, file: null,
  };
  private listeners = new Set<() => void>();
  private frameListeners = new Set<(f: PreviewFrame | null) => void>();
  private lastFrame: PreviewFrame | null = null;

  /** The card that can still be decided (Bubble.proposal); null once consumed or dismissed. */
  private live: Proposal | null = null;
  /** The card or message the bubble shows in its live view. */
  private shownCard: { proposal: Proposal; title: string } | null = null;
  private message: Message | null = null;
  private history: Proposed[] = [];
  private view: number | null = null;
  private hideTimer: ReturnType<typeof setTimeout> | null = null;
  private historyTimer: ReturnType<typeof setTimeout> | null = null;
  private noticeTimer: ReturnType<typeof setTimeout> | null = null;
  private noticeId = 0;

  // --- store --------------------------------------------------------------
  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  getState = (): PetteState => this.state;

  private set(patch: Partial<PetteState>) {
    this.state = { ...this.state, ...patch };
    for (const fn of [...this.listeners]) fn();
  }

  /** Recompute the bubble from the live card / message / history page. */
  private render(extra: Partial<PetteState> = {}) {
    let bubble: BubbleView | null = null;
    if (this.view !== null && this.history[this.view]) {
      const h = this.history[this.view];
      bubble = { kind: "history", title: h.title, text: h.text, index: this.view + 1, total: this.history.length };
    } else if (this.shownCard) {
      bubble = { kind: "card", proposal: this.shownCard.proposal, title: this.shownCard.title,
        live: this.live?.id === this.shownCard.proposal.id };
    } else if (this.message) {
      bubble = { kind: "message", ...this.message };
    }
    this.set({ bubble, ...extra });
  }

  get paused(): boolean { return this.state.paused; }
  get bubbleEnabled(): boolean { return this.state.bubbleEnabled; }
  get contextAllowed(): boolean { return this.state.contextAllowed; }
  /** The card that is up and still undecided (Bubble.proposal). */
  get proposal(): Proposal | null { return this.live; }

  // --- bubble -------------------------------------------------------------
  /** A plain message; `seconds` null keeps it until something replaces it. Drops an open card. */
  say(text: string, opts: { title?: string; seconds?: number | null } = {}): void {
    if (this.state.paused || !this.state.bubbleEnabled) return;
    this.live = null;
    this.shownCard = null;
    this.message = { title: opts.title ?? "", text };
    this.view = null;
    this.render({ progress: null }); // a plain message never carries a card's time-left bar
    this.armHide(opts.seconds === undefined ? 6 : opts.seconds);
  }

  /** Show a card (title, text, Open/Copy + Not now). */
  propose(p: Proposal): void {
    if (this.state.paused || !this.state.bubbleEnabled) return;
    this.live = p;
    this.shownCard = { proposal: p, title: p.title };
    this.message = null;
    this.view = null;
    this.history = [...this.history.filter((h) => h.id !== p.id), { id: p.id, title: p.title, text: p.text }]
      .slice(-HISTORY_SIZE);
    this.render();
    this.armHide(null);
  }

  /** Hide the bubble; an open card can't be accepted afterwards. */
  dismiss(): void {
    this.clearTimer("hideTimer");
    this.clearTimer("historyTimer");
    this.live = null;
    this.shownCard = null;
    this.message = null;
    this.view = null;
    this.render({ progress: null });
  }

  setProgress(fraction: number | null): void {
    this.set({ progress: fraction === null ? null : Math.max(0, Math.min(1, fraction)) });
  }

  /** Single-use decision bound to the displayed card: its id once, then null. */
  consume(): string | null {
    const p = this.live;
    this.live = null;
    if (p) this.render();
    return p ? p.id : null;
  }

  /** The card's primary button (Open / Copy). */
  approve(): void {
    const pid = this.consume();
    if (pid !== null) this.events.emit("approved", pid);
  }

  /** The card's "Not now". */
  cancel(): void {
    const pid = this.consume();
    if (pid !== null) this.events.emit("cancelled", pid);
  }

  // --- WatcherTarget ------------------------------------------------------
  /** Pinch on a card = "Do it". Otherwise ignored (also while paging through history). */
  pinch(): void {
    if (!this.state.gesturesEnabled || this.state.paused || !this.state.contextAllowed || this.view !== null) return;
    if (this.live) this.events.emit("pinchOnProposal", this.live.id);
  }

  /** Back from being away: an open card is re-titled "While you were away · …". */
  nudge(): void {
    const p = this.live;
    if (this.state.paused || !p || !this.shownCard || this.state.bubble === null) return;
    if (!this.shownCard.title.startsWith("While you were away")) {
      this.shownCard = { ...this.shownCard, title: `While you were away · ${p.title}` };
      this.render();
    }
  }

  /** Webcam reactions: Pette dozes while you're away and shuts her eyes with yours. */
  setPresence(away: boolean, eyesClosed: boolean): void {
    this.set({ away, eyesClosed });
    this.updateMood();
  }

  /** A camera preview frame: drawn on the orb only, never stored. */
  setFrame(frame: PreviewFrame): void {
    if (this.state.paused) return;
    this.lastFrame = frame;
    for (const fn of [...this.frameListeners]) fn(frame);
    if (frame.gesture !== this.state.gesture) this.set({ gesture: frame.gesture });
  }

  /** Point left / right: page through the recent suggestions (read-only, "2 / 5"). */
  step(direction: -1 | 1): void {
    if (this.state.gesturesEnabled) this.page(direction);
  }

  /** Page through the recent suggestions (the bubble's arrows and the pointing gesture). */
  page(direction: -1 | 1): void {
    if (this.state.paused || !this.state.bubbleEnabled || !this.history.length) return;
    const n = this.history.length;
    const liveIndex = this.live ? this.history.findIndex((h) => h.id === this.live!.id) : -1;
    const current = this.view ?? (liveIndex >= 0 ? liveIndex : n);
    const next = Math.max(0, Math.min(n, current + direction));
    this.clearTimer("historyTimer");
    if (next === n || next === liveIndex) {
      this.view = null; // back to the live view (the open card, a message, or nothing)
    } else {
      this.view = next;
      this.historyTimer = setTimeout(() => {
        this.historyTimer = null;
        this.view = null;
        this.render();
      }, HISTORY_S * 1000);
    }
    this.render();
  }

  /** Pinch-and-drag: scrolls Pette's world panel touch-style. */
  scroll(dy: number): void {
    if (!this.state.gesturesEnabled || this.state.paused) return;
    this.events.emit("scroll", dy);
  }

  // --- orb ----------------------------------------------------------------
  setStatus(state: PetteStatus, detail = ""): void {
    this.set({ status: state, detail });
  }

  /** Click on the orb: look now (never while paused). */
  requestLook(): void {
    if (this.state.paused) return;
    this.events.emit("lookRequested");
  }

  setCameraOn(on: boolean): void {
    if (!on) {
      this.lastFrame = null;
      for (const fn of [...this.frameListeners]) fn(null);
    }
    this.set({ cameraOn: on, ...(on ? {} : { gesture: "", away: false, eyesClosed: false }) });
    this.updateMood();
  }

  /** Hide or show the camera picture inside the orb; the camera itself keeps running. */
  setCameraVisible(visible: boolean): void {
    if (visible !== this.state.feedVisible) this.set({ feedVisible: visible });
  }

  setGesturesEnabled(on: boolean): void {
    this.set({ gesturesEnabled: on });
  }

  toggleCameraView(): void {
    this.setCameraVisible(!this.state.feedVisible);
  }

  /** The orb's canvas subscribes to frames here (not through React state: 30 fps). */
  onFrame(fn: (f: PreviewFrame | null) => void): () => void {
    this.frameListeners.add(fn);
    fn(this.lastFrame);
    return () => this.frameListeners.delete(fn);
  }

  // --- toggles ------------------------------------------------------------
  setPaused(paused: boolean): void {
    if (paused === this.state.paused) return;
    if (paused) this.dismiss(); // also drops an open card so it can't be accepted later
    this.set({ paused });
    this.updateMood();
    this.events.emit("pauseToggled", paused);
  }

  /** Speech bubble on/off. Off: nothing shows and SOMA's cards wait. */
  setBubbleEnabled(on: boolean): void {
    if (on === this.state.bubbleEnabled) return;
    if (!on) this.dismiss();
    this.set({ bubbleEnabled: on });
    this.events.emit("bubbleToggled", on);
    this.notify("Pette", on ? "Speech bubble on." : "Speech bubble off: suggestions wait until it's back.");
  }

  toggleBubble(): void {
    this.setBubbleEnabled(!this.state.bubbleEnabled);
  }

  setContextAllowed(allowed: boolean): void {
    this.set({ contextAllowed: allowed });
  }

  setWorldVisible(visible: boolean): void {
    this.set({ worldVisible: visible });
  }

  toggleWorld(): void {
    this.setWorldVisible(!this.state.worldVisible);
  }

  /** A short notice (the desktop tray balloon), shown even while the bubble is off. */
  notify(title: string, text: string): void {
    this.clearTimer("noticeTimer");
    this.set({ notice: { id: ++this.noticeId, title, text } });
    this.noticeTimer = setTimeout(() => {
      this.noticeTimer = null;
      this.set({ notice: null });
    }, NOTICE_S * 1000);
  }

  // --- file viewer --------------------------------------------------------
  showFile(file: OpenedFile): void {
    this.set({ file });
  }

  closeFile(): void {
    this.set({ file: null });
  }

  // --- internals ----------------------------------------------------------
  private updateMood() {
    const s = this.state;
    const mood: Mood = s.paused ? "sleeping" : s.away ? "away" : s.eyesClosed ? "blink" : "idle";
    if (mood !== s.mood) this.set({ mood });
  }

  private armHide(seconds: number | null) {
    this.clearTimer("hideTimer");
    if (seconds) {
      this.hideTimer = setTimeout(() => {
        this.hideTimer = null;
        this.dismiss();
      }, seconds * 1000);
    }
  }

  private clearTimer(name: "hideTimer" | "historyTimer" | "noticeTimer") {
    const t = this[name];
    if (t !== null) clearTimeout(t);
    this[name] = null;
  }

  /** Stop every timer (unmount). */
  dispose(): void {
    this.clearTimer("hideTimer");
    this.clearTimer("historyTimer");
    this.clearTimer("noticeTimer");
    this.listeners.clear();
    this.frameListeners.clear();
  }
}
