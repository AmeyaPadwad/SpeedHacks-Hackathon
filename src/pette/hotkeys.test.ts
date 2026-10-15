// Port of tests/test_hotkeys.py: the same combos, matched exactly; the hook fires handlers and cleans up.
import { renderHook } from "@testing-library/react";
import { HOTKEYS, comboLabel, matchHotkey, matches, parseCombo, useHotkeys } from "./hotkeys";

const key = (code: string, mods: Partial<Record<"ctrlKey" | "altKey" | "shiftKey" | "metaKey", boolean>> = {}, k = "") =>
  ({ code, key: k, ctrlKey: false, altKey: false, shiftKey: false, metaKey: false, ...mods });

it("keeps the desktop combos", () => {
  expect(HOTKEYS).toEqual({ look: "ctrl+shift+space", suggestions: "ctrl+alt+s", bubble: "ctrl+alt+b",
    camera: "ctrl+alt+c", world: "ctrl+alt+w" });
  expect(parseCombo("Ctrl+Shift+Space")).toEqual({ ctrl: true, alt: false, shift: true, meta: false, key: "space" });
  expect(comboLabel(HOTKEYS.suggestions)).toBe("Ctrl+Alt+S");
});

it("matches exactly, by physical key", () => {
  expect(matchHotkey(key("Space", { ctrlKey: true, shiftKey: true }, " "), HOTKEYS)).toBe("look");
  expect(matchHotkey(key("KeyS", { ctrlKey: true, altKey: true }, "ß"), HOTKEYS)).toBe("suggestions"); // AltGr layouts
  expect(matchHotkey(key("KeyB", { ctrlKey: true, altKey: true }), HOTKEYS)).toBe("bubble");
  expect(matchHotkey(key("KeyC", { ctrlKey: true, altKey: true }), HOTKEYS)).toBe("camera");
  expect(matchHotkey(key("KeyW", { ctrlKey: true, altKey: true }), HOTKEYS)).toBe("world");
  expect(matchHotkey(key("KeyS", { ctrlKey: true }), HOTKEYS)).toBeNull(); // Ctrl+S is the page's
  expect(matchHotkey(key("KeyS", { ctrlKey: true, altKey: true, shiftKey: true }), HOTKEYS)).toBeNull();
  expect(matches(key("F2", {}, "F2"), "f2")).toBe(true);
});

it("the hook fires the handler and stops after unmount", () => {
  const looks: number[] = [];
  const { unmount } = renderHook(() => useHotkeys({ look: () => looks.push(1) }));
  const press = () => window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space", key: " ", ctrlKey: true, shiftKey: true }));
  press();
  expect(looks).toHaveLength(1);
  window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space", key: " ", ctrlKey: true, shiftKey: true, repeat: true }));
  expect(looks).toHaveLength(1); // held keys don't repeat
  unmount();
  press();
  expect(looks).toHaveLength(1);
});
