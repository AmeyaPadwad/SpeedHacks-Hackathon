/**
 * "Float Pette": Document Picture-in-Picture (Chromium 116+) opens a small always-on-top window that can sit
 * over other apps, like the desktop overlay did. The orb and bubble render into it through a React portal,
 * so they keep sharing the same controller and state. Feature-detected: elsewhere the menu item is disabled.
 */
import { useCallback, useEffect, useState } from "react";

interface DocumentPictureInPicture extends EventTarget {
  requestWindow(options?: { width?: number; height?: number; disallowReturnToOpener?: boolean }): Promise<Window>;
  readonly window: Window | null;
}

declare global {
  interface Window {
    documentPictureInPicture?: DocumentPictureInPicture;
  }
}

export const PIP_SIZE = { width: 520, height: 320 };

export const pipSupported = (w: Window = window): boolean => typeof w.documentPictureInPicture?.requestWindow === "function";

/** Copy the page's styles (Tailwind's sheet, fonts) into the PiP document. */
export function copyStyles(from: Document, to: Document): void {
  for (const sheet of Array.from(from.styleSheets)) {
    try {
      const css = Array.from(sheet.cssRules).map((r) => r.cssText).join("\n");
      const style = to.createElement("style");
      style.textContent = css;
      to.head.appendChild(style);
    } catch {
      // cross-origin sheet: link it instead
      if (sheet.href) {
        const link = to.createElement("link");
        link.rel = "stylesheet";
        link.href = sheet.href;
        to.head.appendChild(link);
      }
    }
  }
}

/** Open the floating window (call from a click). Resolves to the container to portal into, or null. */
export async function openPip(size = PIP_SIZE, w: Window = window): Promise<{ win: Window; root: HTMLElement } | null> {
  if (!pipSupported(w)) return null;
  let win: Window;
  try {
    win = await w.documentPictureInPicture!.requestWindow(size);
  } catch {
    return null; // no user gesture, or the user/browser refused
  }
  copyStyles(w.document, win.document);
  win.document.title = "Pette";
  win.document.body.className = "pette-pip";
  const root = win.document.createElement("div");
  root.id = "pette-pip-root";
  root.style.height = "100%";
  win.document.body.appendChild(root);
  return { win, root };
}

/** React state for the floating window: `root` to portal into while open. */
export function usePip() {
  const [pip, setPip] = useState<{ win: Window; root: HTMLElement } | null>(null);
  const supported = pipSupported();

  const open = useCallback(async () => {
    if (pip) return true;
    const p = await openPip();
    if (!p) return false;
    p.win.addEventListener("pagehide", () => setPip(null), { once: true });
    setPip(p);
    return true;
  }, [pip]);

  const close = useCallback(() => {
    pip?.win.close();
    setPip(null);
  }, [pip]);

  useEffect(() => () => pip?.win.close(), [pip]);

  return { supported, window: pip?.win ?? null, root: pip?.root ?? null, open, close };
}
