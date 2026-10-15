/**
 * Access to files in the user's picked folders (the web counterpart of `runtime.authorized_file` and
 * `ConsentManager.path_allowed`).
 *
 * A reference is `folderId/relative/path`. It is authorized only when it is syntactically safe (no `..`,
 * `.`, empty or absolute parts, no drive letters, no control characters) and its folder is one the user
 * picked and has not removed. Resolution walks the folder's directory handle, which cannot escape the
 * folder anyway. Screen text is untrusted: `matchScreenPath` only maps a path seen on screen onto an
 * already-indexed reference; it never builds a new one.
 */
import type { MemoryStore } from "./store";

export interface ParsedRef {
  folderId: string;
  /** Path segments inside the folder (at least one). */
  parts: string[];
}

/** Split a reference into folder id + safe segments, or null when it is malformed or escapes. */
export function parseRef(ref: string): ParsedRef | null {
  if (typeof ref !== "string" || !ref || ref.length > 4096) return null;
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(ref)) return null;
  if (ref.startsWith("/") || ref.startsWith("\\") || /^[A-Za-z]:/.test(ref) || ref.includes("\\") || ref.includes("://")) return null;
  const [folderId, ...parts] = ref.split("/");
  if (!folderId || !parts.length) return null;
  if (parts.some((p) => !p || p === "." || p === ".." || p.includes(":"))) return null;
  return { folderId, parts };
}

/** Sync check for policy code: a safe reference inside one of `folderIds`. */
export function isAuthorizedRef(ref: string, folderIds: ReadonlySet<string> | Iterable<string>): boolean {
  const parsed = parseRef(ref);
  if (!parsed) return false;
  const ids = folderIds instanceof Set ? folderIds : new Set(folderIds);
  return ids.has(parsed.folderId);
}

/** Async check backed by the store's folder list. */
export async function authorizedFile(store: MemoryStore, ref: string): Promise<boolean> {
  const parsed = parseRef(ref);
  return !!parsed && (await store.getFolder(parsed.folderId)) !== null;
}

/** Current permission on a handle, without prompting ("granted" when the API is absent, e.g. in tests). */
export async function queryReadPermission(handle: FileSystemHandle): Promise<PermissionState> {
  const h = handle as FileSystemHandle & { queryPermission?: (d: { mode: "read" }) => Promise<PermissionState> };
  if (typeof h.queryPermission !== "function") return "granted";
  try {
    return await h.queryPermission({ mode: "read" });
  } catch {
    return "denied";
  }
}

/**
 * Make sure we may read `handle`: query first, then ask (needs a user gesture after a reload).
 * Resolves true only when read access is granted.
 */
export async function requestPermission(handle: FileSystemHandle): Promise<boolean> {
  if ((await queryReadPermission(handle)) === "granted") return true;
  const h = handle as FileSystemHandle & { requestPermission?: (d: { mode: "read" }) => Promise<PermissionState> };
  if (typeof h.requestPermission !== "function") return false;
  try {
    return (await h.requestPermission({ mode: "read" })) === "granted";
  } catch {
    return false;
  }
}

/**
 * The file handle behind an authorized reference, or null (unknown folder, bad reference, missing file,
 * or read permission not granted). Never prompts; call `requestPermission` from a click first.
 */
export async function resolveFile(store: MemoryStore, ref: string): Promise<FileSystemFileHandle | null> {
  const parsed = parseRef(ref);
  if (!parsed) return null;
  const folder = await store.getFolder(parsed.folderId);
  if (!folder?.handle) return null;
  if ((await queryReadPermission(folder.handle)) !== "granted") return null;
  try {
    let dir: FileSystemDirectoryHandle = folder.handle;
    for (const part of parsed.parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part);
    return await dir.getFileHandle(parsed.parts[parsed.parts.length - 1]);
  } catch {
    return null;
  }
}

/**
 * The indexed reference a path seen on screen points to, by the longest unique suffix match on path
 * segments (case-insensitive): `C:\dev\app\src\main.py` or `src/main.py` -> `fld_1/src/main.py`.
 * Null when nothing matches or the best match is ambiguous.
 */
export function matchScreenPath(sourceUris: Iterable<string>, screenPath: string): string | null {
  const want = (screenPath ?? "").replace(/\\/g, "/").split("/").filter((p) => p && p !== "." && !/^[A-Za-z]:$/.test(p)).map((p) => p.toLowerCase());
  if (!want.length || want.includes("..")) return null;
  let best: string | null = null;
  let bestLen = 0;
  let tie = false;
  for (const uri of sourceUris) {
    const parsed = parseRef(uri);
    if (!parsed) continue;
    const have = parsed.parts.map((p) => p.toLowerCase());
    let n = 0;
    while (n < want.length && n < have.length && want[want.length - 1 - n] === have[have.length - 1 - n]) n++;
    if (n === 0) continue;
    if (n > bestLen) {
      best = uri;
      bestLen = n;
      tie = false;
    } else if (n === bestLen) {
      tie = true;
    }
  }
  return tie ? null : best;
}
