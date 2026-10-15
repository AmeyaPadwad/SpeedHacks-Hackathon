/**
 * Status -> ring colour and label (port of pet/ui.py's STATUS), plus the bubble's glass alphas.
 */
import type { PetteStatus } from "./contracts";

/** status -> [ring colour, label]; "{d}" is replaced by the status detail ("SOMA offline"). */
export const STATUS: Record<PetteStatus, readonly [string, string]> = {
  idle: ["#a1a1aa", ""],
  looking: ["#3b82f6", "Looking"],
  suggesting: ["#f59e0b", "Suggestion"],
  done: ["#22c55e", "Done ✓"],
  dismissed: ["#71717a", "Not now"],
  indexing: ["#8b5cf6", "Indexing"],
  failed: ["#ef4444", "{d}"],
};

/** The pill under the orb: "", "Looking", "Done ✓", "SOMA offline", ... */
export function statusLabel(state: PetteStatus, detail = ""): string {
  const text = (STATUS[state] ?? STATUS.idle)[1];
  return text.includes("{d}") ? text.replace("{d}", detail) : text;
}

export const statusColor = (state: PetteStatus): string => (STATUS[state] ?? STATUS.idle)[0];

/** Frosted-glass bubble alphas (0..1): fill top/bottom, sheen, inner edge, hairline (GLASS_* / 255). */
export const GLASS = { top: 182 / 255, bottom: 142 / 255, sheen: 110 / 255, edge: 190 / 255, hairline: 34 / 255 };

/** Camera orb diameter in CSS pixels. */
export const ORB = 116;
