/**
 * Contracts shared by Pette's modules (UI, hub, webcam). Kept free of React and tfjs imports.
 */

/** Debounced webcam events (port of pet/vision.py's event names). */
export type VisionEvent =
  | "face_away" | "face_back"
  | "eyes_closed" | "eyes_open"
  | "pinch"
  | "point_left" | "point_right"
  | "drag_start" | "drag_end";

/** One hand landmark in the mirrored preview's coordinates (0..1, x grows to the user's right). */
export interface PreviewPoint { x: number; y: number }

/** What the orb draws for one camera frame. Never stored, logged or sent anywhere. */
export interface PreviewFrame {
  /** Mirrored square crop of the webcam, drawn only on the orb's canvas. */
  bitmap: ImageBitmap | null;
  hands: PreviewPoint[][];
  /** Gesture name shown on the orb ("pinch", "point left", ""). */
  gesture: string;
}

/** Hub status (Suggester.status): drives the orb's ring colour and label. */
export type PetteStatus = "idle" | "looking" | "suggesting" | "done" | "dismissed" | "indexing" | "failed";

/** Orb moods from webcam presence / pause. */
export type Mood = "idle" | "away" | "blink" | "sleeping";

/** A card in the speech bubble. Only Pette's hub creates these. */
export interface Proposal {
  id: string;
  title: string;
  text: string;
  actionLabel: string;
  cancelLabel: string;
  glow: boolean;
}

/** Fact-log outcomes for a card. "ignored" (expired, paused, toggled off) is never a dismissal. */
export type Outcome = "accepted" | "rejected" | "ignored";

/**
 * Pette's fact log (port of db/store.py): facts only, never suggestion text, file names or screen content.
 * Implemented over IndexedDB in `pette/world/store.ts`; `null`-able everywhere it is injected.
 */
export interface FactLog {
  logEvent(kind: string, duration?: number | null): void;
  logOffer(kind: string, outcome: Outcome): void;
  logBlinkBaseline(mean: number, p95: number, samples: number): void;
}

/**
 * What the webcam watcher drives (implemented by Pette's controller). A web page cannot switch the
 * user's browser tabs or scroll other apps, so the gestures act inside Pette:
 *   point left / right -> `step(-1 | +1)`: previous / next of the recent suggestions in the bubble;
 *   pinch-and-drag     -> `scroll(dy)`: scrolls Pette's world panel touch-style;
 *   pinch (held still) -> `pinch()`: "Do it" on the card that is up, otherwise ignored.
 */
export interface WatcherTarget {
  readonly paused: boolean;
  pinch(): void;
  /** Away -> back: an open card is re-titled "While you were away · …". */
  nudge(): void;
  setPresence(away: boolean, eyesClosed: boolean): void;
  setFrame(frame: PreviewFrame): void;
  step(direction: -1 | 1): void;
  /** Image heights moved since the last call, + = hand moved down. */
  scroll(dy: number): void;
}
