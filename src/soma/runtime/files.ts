/**
 * Map file paths seen on screen to the user's authorized files.
 *
 * In the desktop build an on-screen path was authorized when it resolved inside an allowed folder. A web
 * page has no absolute paths: the user's folders are directory handles and every indexed file is a ref
 * `folderId/relative/path`. So a path read on screen (`C:\proj\app\calc.py`, `app/calc.py`, `calc.py`) is
 * authorized only when it matches exactly one indexed file of an allowed folder by its trailing path
 * components; the candidate then carries that ref, which `readFile` can open. Ambiguous matches resolve to
 * nothing (never guess which file the user meant).
 */

const norm = (p: string): string =>
  p.replace(/\\/g, "/").replace(/\/{2,}/g, "/").replace(/^(\.\/)+/, "").trim().toLowerCase();

const basename = (p: string): string => p.slice(p.lastIndexOf("/") + 1);

/** Split `folderId/relative/path` -> `[folderId, relative/path]` (null when malformed). */
export function splitRef(ref: string): [string, string] | null {
  const i = ref.indexOf("/");
  if (i <= 0 || i === ref.length - 1) return null;
  const rel = ref.slice(i + 1);
  if (rel.split("/").some((part) => part === ".." || part === ".")) return null;
  return [ref.slice(0, i), rel];
}

export class FileRefIndex {
  private readonly byName = new Map<string, { ref: string; rel: string }[]>();

  constructor(refs: Iterable<string> = []) {
    for (const ref of refs) {
      const parts = splitRef(ref);
      if (!parts) continue;
      const rel = norm(parts[1]);
      const name = basename(rel);
      const list = this.byName.get(name) ?? [];
      list.push({ ref, rel });
      this.byName.set(name, list);
    }
  }

  get size(): number {
    let n = 0;
    for (const l of this.byName.values()) n += l.length;
    return n;
  }

  /** The unique indexed ref an on-screen path points at, or null. */
  resolve(screenPath: string): string | null {
    const p = norm(screenPath);
    if (!p || p.split("/").includes("..")) return null;
    const list = this.byName.get(basename(p));
    if (!list) return null;
    let best: string | null = null;
    let bestScore = 0;
    let tie = false;
    for (const { ref, rel } of list) {
      let score = 0;
      if (p === rel) score = rel.length + 1;
      else if (p.endsWith("/" + rel)) score = rel.length;
      else if (rel.endsWith("/" + p)) score = p.length;
      if (score === 0) continue;
      if (score > bestScore) {
        best = ref;
        bestScore = score;
        tie = false;
      } else if (score === bestScore && ref !== best) {
        tie = true;
      }
    }
    return tie ? null : best;
  }
}
