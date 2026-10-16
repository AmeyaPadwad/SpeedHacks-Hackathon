/**
 * Pette's keyboard shortcuts (port of pet/hotkeys.py). A web page can't register global hotkeys, so these
 * work while Pette's tab (or her floating Picture-in-Picture window) has focus. Same combos as the desktop app.
 */
import { useEffect, useRef } from "react";

export type HotkeyAction = "look" | "suggestions" | "bubble" | "camera" | "world";

export const HOTKEYS: Record<HotkeyAction, string> = {
  look: "ctrl+shift+space", // same as clicking Pette: look now
  suggestions: "ctrl+alt+s", // SOMA suggestions on / off
  bubble: "ctrl+alt+b", // speech bubble on / off
  camera: "ctrl+alt+c", // hide / show the camera picture
  world: "ctrl+alt+w", // open the dashboard (/debug)
};

export interface Combo { ctrl: boolean; alt: boolean; shift: boolean; meta: boolean; key: string }

/** "ctrl+shift+space" -> { ctrl, shift, key: "space" }. Keys are lower-case names ("s", "space", "f1"). */
export function parseCombo(combo: string): Combo {
  const parts = combo.toLowerCase().split("+").map((p) => p.trim()).filter(Boolean);
  const c: Combo = { ctrl: false, alt: false, shift: false, meta: false, key: "" };
  for (const p of parts) {
    if (p === "ctrl" || p === "control") c.ctrl = true;
    else if (p === "alt" || p === "option") c.alt = true;
    else if (p === "shift") c.shift = true;
    else if (p === "meta" || p === "cmd" || p === "win") c.meta = true;
    else c.key = p;
  }
  return c;
}

/** The pressed key's name, layout-independent for letters, digits and space (uses `code`). */
export function keyName(e: Pick<KeyboardEvent, "key" | "code">): string {
  const code = e.code ?? "";
  if (/^Key[A-Z]$/.test(code)) return code.slice(3).toLowerCase();
  if (/^Digit\d$/.test(code)) return code.slice(5);
  if (code === "Space" || e.key === " ") return "space";
  return (e.key ?? "").toLowerCase();
}

type KeyLike = Pick<KeyboardEvent, "key" | "code" | "ctrlKey" | "altKey" | "shiftKey" | "metaKey">;

export function matches(e: KeyLike, combo: string | Combo): boolean {
  const c = typeof combo === "string" ? parseCombo(combo) : combo;
  return e.ctrlKey === c.ctrl && e.altKey === c.alt && e.shiftKey === c.shift && e.metaKey === c.meta
    && keyName(e) === c.key;
}

/** Which action a key event triggers, or null. */
export function matchHotkey<A extends string>(e: KeyLike, bindings: Record<A, string>): A | null {
  for (const action of Object.keys(bindings) as A[]) if (matches(e, bindings[action])) return action;
  return null;
}

/** Title-cased combo for menus: "Ctrl+Alt+S". */
export const comboLabel = (combo: string): string =>
  combo.split("+").map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join("+");

/**
 * Listen for Pette's hotkeys on `targets` (her window, and the PiP window when open). Handlers may change
 * between renders without re-subscribing.
 */
export function useHotkeys(
  handlers: Partial<Record<HotkeyAction, () => void>>,
  targets: (Window | null | undefined)[] = [typeof window === "undefined" ? null : window],
  bindings: Record<HotkeyAction, string> = HOTKEYS,
): void {
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;
      const action = matchHotkey(e, bindings);
      const fn = action ? ref.current[action] : undefined;
      if (fn) {
        e.preventDefault();
        fn();
      }
    };
    const live = targets.filter((t): t is Window => !!t);
    for (const t of live) t.addEventListener("keydown", onKey);
    return () => {
      for (const t of live) t.removeEventListener("keydown", onKey);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bindings, ...targets]);
}
