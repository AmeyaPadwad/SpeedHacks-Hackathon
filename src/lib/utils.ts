import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** shadcn's class merger: clsx + tailwind-merge. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** "12s ago" / "3m ago" / "2h ago" / "1d ago", both times in milliseconds. */
export function ago(tsMs: number, nowMs: number) {
  const s = Math.max(0, Math.round((nowMs - tsMs) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
