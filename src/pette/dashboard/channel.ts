/**
 * The link between the running Pette tab and the `/debug` dashboard tab: a same-origin BroadcastChannel.
 * Nothing leaves the browser. Pette publishes a `DashboardSnapshot` every second (and when asked with
 * "hello"); the dashboard only listens, except for "hello". Pinch-and-drag scroll amounts are forwarded so
 * the gesture scrolls the dashboard.
 *
 * Facts and numbers only, like the world panel was: status, counts, timings, kinds. Never screen text, file
 * names or suggestion text.
 */
import type { FolderInfo, SomaDiagnostics, SomaStatus } from "@/soma/service";
import type { Mood, PetteStatus } from "../contracts";
import type { WorldSnapshot } from "../world/worldModel";

export const CHANNEL = "pette-dashboard";
export const PUBLISH_MS = 1000;

export interface PetteLive {
  status: PetteStatus;
  /** The orb's pill text. */
  label: string;
  mood: Mood;
  camera: boolean;
  gesture: string;
  present: boolean;
  eyesClosed: boolean;
  paused: boolean;
  suggestions: boolean;
  bubble: boolean;
  gestures: boolean;
  screenShared: boolean;
  floating: boolean;
  cardUp: boolean;
}

export interface DashboardSnapshot {
  at: number;
  pette: PetteLive;
  world: WorldSnapshot;
  folders: FolderInfo[];
  soma: SomaStatus | null;
  diagnostics: SomaDiagnostics | null;
  sandbox: boolean;
}

export type DashboardMessage =
  | { type: "snapshot"; data: DashboardSnapshot }
  | { type: "hello" }
  | { type: "scroll"; dy: number };

/** A channel that never throws (BroadcastChannel is missing in some test environments). */
export function openChannel(onMessage?: (m: DashboardMessage) => void): { post(m: DashboardMessage): void; close(): void } {
  if (typeof BroadcastChannel === "undefined") return { post: () => undefined, close: () => undefined };
  const ch = new BroadcastChannel(CHANNEL);
  if (onMessage) ch.onmessage = (e: MessageEvent<DashboardMessage>) => onMessage(e.data);
  return {
    post: (m) => {
      try {
        ch.postMessage(m);
      } catch {
        // a snapshot that can't be cloned is dropped, never fatal
      }
    },
    close: () => ch.close(),
  };
}
