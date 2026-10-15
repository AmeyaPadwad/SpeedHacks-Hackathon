/**
 * What "Do it" may do (port of pet/actions.py): open the file SOMA pointed at in Pette's own file viewer,
 * or copy the suggestion text. Nothing else.
 *
 * Opening never runs anything: the file is read through `SomaService.readFile` (which only serves files inside
 * the folders you granted) and shown as text, as an image, or offered as a download of its bytes. Nothing
 * from SOMA is ever evaluated, navigated to or passed to `window.open`.
 */
import type { SomaService } from "@/soma/service";
import type { OpenedFile } from "./controller";

export const IMAGE_EXT = new Set([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]);

/** The file is gone or can't be opened / copied; shown in the bubble. */
export class ActionError extends Error {
  override name = "ActionError";
}

/** Last path segment of a `folderId/relative/path` ref. */
export const baseName = (ref: string): string => ref.split(/[\\/]/).filter(Boolean).pop() ?? ref;

export const extOf = (name: string): string => {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i).toLowerCase() : "";
};

/** How the viewer shows a file: text with line numbers, an image, or a download. */
export function fileKind(f: Pick<OpenedFile, "name" | "text">): "image" | "text" | "binary" {
  if (IMAGE_EXT.has(extOf(f.name))) return "image";
  return f.text !== null ? "text" : "binary";
}

/** Open `ref` (at `line` when known) in Pette's viewer. Returns a short description of what happened. */
export async function openFile(
  soma: Pick<SomaService, "readFile">,
  show: (file: OpenedFile) => void,
  ref: string,
  line: number | null = null,
): Promise<string> {
  let r: Awaited<ReturnType<SomaService["readFile"]>>;
  try {
    r = await soma.readFile(ref);
  } catch {
    r = null;
  }
  if (!r) throw new ActionError(`${baseName(ref) || ref} isn't there any more.`);
  const ln = typeof line === "number" && Number.isInteger(line) && line > 0 ? line : null;
  const file: OpenedFile = { name: r.name, text: r.text, blob: r.blob, line: fileKind(r) === "text" ? ln : null };
  show(file);
  if (fileKind(file) === "binary") return `${r.name} can't be shown here; download it from the viewer.`;
  return `Opened ${r.name}` + (file.line ? ` at line ${file.line}` : "") + ".";
}

/** Copy the suggestion text (never anything else). */
export async function copyText(
  text: string,
  write: (text: string) => Promise<void> = (t) => navigator.clipboard.writeText(t),
): Promise<string> {
  try {
    await write(text);
  } catch {
    throw new ActionError("The browser blocked the clipboard. Select the text in the bubble and copy it.");
  }
  return "Copied. Paste it to your coding agent or notes.";
}

/** A download-only copy of a file's bytes (never rendered by the browser). */
export const downloadBlob = (blob: Blob): Blob => new Blob([blob], { type: "application/octet-stream" });
