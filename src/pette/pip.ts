/**
 * "Float Pette": Document Picture-in-Picture (Chromium 116+) opens a small always-on-top window that can sit
 * over other apps, like the desktop overlay did. The orb and bubble render into it through a React portal,
 * so they keep sharing the same controller and state. Feature-detected: elsewhere the menu item is disabled.
 */
import { useCallback, useEffect, useRef, useState } from "react";

interface DocumentPictureInPicture extends EventTarget {
  requestWindow(options?: {
    width?: number; height?: number; disallowReturnToOpener?: boolean; preferInitialWindowPlacement?: boolean;
  }): Promise<Window>;
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

/** Shrink a floating window that came up larger than asked (browsers may ignore the requested size). */
export function fitWindow(win: Window, size = PIP_SIZE): void {
  if (win.innerWidth <= size.width * 1.25 && win.innerHeight <= size.height * 1.25) return;
  try {
    win.resizeTo(size.width, size.height + (win.outerHeight - win.innerHeight));
  } catch {
    // not allowed here: the user can still resize it
  }
}

/** Open the floating window (call from a click). Resolves to the container to portal into, or null. */
export async function openPip(size = PIP_SIZE, w: Window = window): Promise<{ win: Window; root: HTMLElement } | null> {
  if (!pipSupported(w)) return null;
  let win: Window;
  try {
    // Chromium remembers a site's last PiP size; Comet also starts it large. Ask for the initial size.
    win = await w.documentPictureInPicture!.requestWindow({ ...size, preferInitialWindowPlacement: true });
  } catch {
    return null; // no user gesture, or the user/browser refused
  }
  // Resizing needs a user gesture, and opening the window used up the click's. Comet opens it far larger than
  // asked (mostly off-screen), so shrink it on the first click or key inside it.
  fitWindow(win, size);
  const fit = () => fitWindow(win, size);
  win.document.addEventListener("pointerdown", fit, { capture: true });
  win.document.addEventListener("keydown", fit, { capture: true });
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

  // The open window lives in a ref so `open` / `close` stay stable across renders (callers register them
  // as media-session handlers; re-registering on every render could close the window).
  const current = useRef<{ win: Window; root: HTMLElement } | null>(null);

  const open = useCallback(async () => {
    if (current.current) return true;
    const p = await openPip();
    if (!p) return false;
    current.current = p;
    p.win.addEventListener("pagehide", () => {
      if (current.current === p) current.current = null;
      setPip((now) => (now === p ? null : now));
    }, { once: true });
    setPip(p);
    return true;
  }, []);

  const close = useCallback(() => {
    const p = current.current;
    current.current = null;
    p?.win.close();
    setPip(null);
  }, []);

  // Close the floating window only when Pette itself unmounts.
  useEffect(() => () => current.current?.win.close(), []);

  return { supported, window: pip?.win ?? null, root: pip?.root ?? null, open, close };
}
