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
