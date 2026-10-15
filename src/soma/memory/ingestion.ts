/**
 * Index a user-picked folder into memory (port of soma/memory/ingestion.py over the File System Access API).
 *
 * Rules enforced here:
 * - The walk never leaves the picked directory handle (a handle cannot reach its parent). Hidden entries
 *   (".git", ".venv", dotfiles) and tool folders (node_modules, __pycache__, ...) are skipped.
 * - Only `cfg.allowedExtensions` under `cfg.maxFileBytes` are read, at most `cfg.maxFilesPerFolder` files.
 * - Raw contents stay in memory: only a short redacted summary (file name, code symbols, head of the
 *   text; or an image's name, size and dimensions), an embedding and provenance are stored.
 * - Re-indexing an unchanged file (same size and lastModified) is a no-op; files that disappeared from
 *   the folder are forgotten when the walk completes.
 */
import type { MemoryConfig } from "../config";
import { redactText } from "../capture/redaction";
import { fnv1a } from "../perception/tokenizer";
import { SourceType, type MemoryItem } from "../types";
import type { TextEmbedder } from "./embedder";
import { utcNowIso, type MemoryStore } from "./store";
import type { VectorIndex } from "./vectorIndex";

export const IMAGE_EXTENSIONS: ReadonlySet<string> = new Set([".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]);
export const SKIP_DIR_NAMES: ReadonlySet<string> = new Set(["__pycache__", "node_modules", ".git", ".venv", "venv",
  "$recycle.bin", "system volume information", ".svn", ".hg", ".idea", ".vscode"]);
/** Redacted text used for the embedding; never stored. */
export const EMBED_CHARS = 20_000;
export const SUMMARY_CHARS = 400;
const MAX_SYMBOLS = 12;

// ---------------------------------------------------------------- redaction

const FALLBACK_PATTERNS: RegExp[] = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
  /\b(?:sk|pk|rk|ghp|gho|ghs|xox[abprs]|AKIA|AIza)[-_A-Za-z0-9]{12,}\b/g,
  /\b(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[:=]\s*\S+/gi,
  /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g,
  /\b(?:\d[ -]?){13,19}\b/g,
];

/** `redactText` plus a stricter local fallback (file text is redacted before it is summarized or embedded). */
export function redactForMemory(text: string): string {
  let out = redactText(text);
  for (const p of FALLBACK_PATTERNS) out = out.replace(p, "[REDACTED]");
  return out;
}

// ---------------------------------------------------------------- extraction

const SYMBOL_RES: RegExp[] = [
  /^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)/gm,
  /^\s*class\s+([A-Za-z_]\w*)/gm,
  /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)/gm,
  /^\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/gm,
  /^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)/gm,
  /^\s*(?:pub\s+)?(?:fn|struct|trait|impl)\s+([A-Za-z_]\w*)/gm,
  /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)/gm,
  /^#{1,3}\s+(.{1,60})$/gm,
];
const CODE_OR_DOC = /\.(?:py|ts|tsx|js|jsx|mjs|java|c|h|cpp|cs|go|rs|rb|sh|md|rst)$/i;

/** Up to `max` declared names (functions, classes, types, markdown headings), in order of appearance. */
export function codeSymbols(text: string, max = MAX_SYMBOLS): string[] {
  const found: [number, string][] = [];
  for (const re of SYMBOL_RES) for (const m of text.matchAll(re)) found.push([m.index ?? 0, m[1].trim()]);
  found.sort((a, b) => a[0] - b[0]);
  const out: string[] = [];
  for (const [, name] of found) {
    if (!out.includes(name)) out.push(name);
    if (out.length >= max) break;
  }
  return out;
}

const collapse = (text: string, n: number): string => text.replace(/\s+/g, " ").trim().slice(0, n);
const extOf = (name: string): string => {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i).toLowerCase() : "";
};
const stemOf = (name: string): string => {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(0, i) : name;
};

/** Stable memory id for a source URI (same file -> same id across runs). */
export function itemIdFor(sourceUri: string): string {
  const a = fnv1a(sourceUri).toString(16).padStart(8, "0");
  const b = fnv1a(sourceUri, 0x01234567).toString(16).padStart(8, "0");
  const c = fnv1a(`${sourceUri}#`, 0x89abcdef).toString(16).padStart(8, "0");
  return `mem_${a}${b}${c}`.slice(0, 24);
}

/** Image width/height without keeping the decoded pixels. */
async function defaultImageSize(file: Blob): Promise<{ width: number; height: number } | null> {
  if (typeof createImageBitmap !== "function") return null;
  try {
    const bmp = await createImageBitmap(file);
    const size = { width: bmp.width, height: bmp.height };
    bmp.close();
    return size;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- walk

export interface IndexDeps {
  store: MemoryStore;
  embedder: TextEmbedder;
  index: VectorIndex;
  cfg: MemoryConfig;
  /** Injectable for tests; defaults to createImageBitmap. */
  imageSize?: (file: Blob) => Promise<{ width: number; height: number } | null>;
}

export interface IndexProgress {
  files: number;
  skipped: number;
}

export interface IndexReport extends IndexProgress {
  added: number;
  updated: number;
  unchanged: number;
  removed: number;
  /** True when the per-folder cap stopped the walk early. */
  capped: boolean;
}

/** Skip hidden entries and tool folders. */
export const isHiddenOrSkipped = (name: string): boolean => name.startsWith(".") || SKIP_DIR_NAMES.has(name.toLowerCase());

async function* walk(dir: FileSystemDirectoryHandle, prefix: string, onSkip: () => void): AsyncGenerator<[string, FileSystemFileHandle]> {
  const entries: [string, FileSystemHandle][] = [];
  for await (const entry of dir.entries()) entries.push(entry);
  entries.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  for (const [name, handle] of entries) {
    if (!name || name.includes("/") || name.includes("\\") || name === "..") continue;
    const rel = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === "directory") {
      if (isHiddenOrSkipped(name)) continue;
      yield* walk(handle as FileSystemDirectoryHandle, rel, onSkip);
    } else if (isHiddenOrSkipped(name)) {
      onSkip();
    } else {
      yield [rel, handle as FileSystemFileHandle];
    }
  }
}

/**
 * Walk `handle` recursively and index allowed files as memories of `folderId`.
 * One bad file never stops the walk (it counts as skipped). `files` counts files now in memory.
 */
export async function indexFolder(
  handle: FileSystemDirectoryHandle,
  folderId: string,
  deps: IndexDeps,
  onProgress?: (p: IndexProgress) => void,
): Promise<IndexReport> {
  const { store, index, cfg } = deps;
  const imageSize = deps.imageSize ?? defaultImageSize;
  const allowed = new Set(cfg.allowedExtensions.map((e) => e.toLowerCase()));
  const report: IndexReport = { files: 0, skipped: 0, added: 0, updated: 0, unchanged: 0, removed: 0, capped: false };
  const seen = new Set<string>();
  const progress = () => onProgress?.({ files: report.files, skipped: report.skipped });

  for await (const [rel, fileHandle] of walk(handle, "", () => { report.skipped += 1; })) {
    const name = rel.slice(rel.lastIndexOf("/") + 1);
    const ext = extOf(name);
    if (!allowed.has(ext)) {
      report.skipped += 1;
      continue;
    }
    if (report.files >= cfg.maxFilesPerFolder) {
      report.capped = true;
      break;
    }
    try {
      const outcome = await indexFile(fileHandle, rel, name, ext, folderId, deps, imageSize);
      if (outcome === "skipped") {
        report.skipped += 1;
      } else {
        report.files += 1;
        report[outcome] += 1;
        seen.add(itemIdFor(`${folderId}/${rel}`));
      }
    } catch {
      report.skipped += 1;
    }
    if ((report.files + report.skipped) % 25 === 0) progress();
  }

  if (!report.capped) {
    const stale = (await store.itemsForFolder(folderId)).map((i) => i.id).filter((id) => !seen.has(id));
    if (stale.length) {
      report.removed = await store.deleteItems(stale);
      stale.forEach((id) => index.remove(id));
    }
  }
  progress();
  return report;
}

async function indexFile(
  fileHandle: FileSystemFileHandle, rel: string, name: string, ext: string, folderId: string,
  deps: IndexDeps, imageSize: NonNullable<IndexDeps["imageSize"]>,
): Promise<"added" | "updated" | "unchanged" | "skipped"> {
  const { store, embedder, index, cfg } = deps;
  const file = await fileHandle.getFile();
  if (file.size > cfg.maxFileBytes) return "skipped";
  const sourceUri = `${folderId}/${rel}`;
  const id = itemIdFor(sourceUri);
  const fingerprint = `${file.size}:${file.lastModified}`;
  const prior = await store.getItem(id);
  if (prior && prior.provenance.fingerprint === fingerprint && index.has(id)) return "unchanged";

  let summary: string;
  let vec: Float32Array;
  let sourceType: SourceType;
  if (IMAGE_EXTENSIONS.has(ext)) {
    const dims = await imageSize(file);
    const kb = Math.max(1, Math.floor(file.size / 1024));
    summary = dims ? `Image ${name} (${dims.width}x${dims.height}, ${kb} KB)` : `Image ${name} (${kb} KB)`;
    vec = embedder.embed(`${stemOf(name)} image ${rel.replace(/[/_.-]+/g, " ")}`);
    sourceType = SourceType.IMAGE;
  } else {
    let text = await file.text();
    if (text.includes("\u0000")) return "skipped"; // binary despite the extension
    let clean = redactForMemory(text.slice(0, EMBED_CHARS));
    text = "";
    const symbols = CODE_OR_DOC.test(name) ? codeSymbols(clean) : [];
    const symbolPart = symbols.length ? ` [${symbols.join(", ")}]` : "";
    summary = `${name}${symbolPart}: ${collapse(clean, SUMMARY_CHARS)}`;
    vec = embedder.embed(`${stemOf(name)} ${symbols.join(" ")} ${clean}`);
    clean = "";
    sourceType = SourceType.DOCUMENT;
  }

  const item: MemoryItem = {
    id, sourceType, userApproved: true, textSummary: summary, sourceUri,
    tags: [ext.replace(/^\./, "")],
    provenance: { folderId, path: rel, size: file.size, lastModified: file.lastModified, fingerprint, ingestedAt: utcNowIso() },
    createdAt: prior?.createdAt ?? "", updatedAt: "", expiresAt: null,
  };
  await store.putItem(item, vec);
  index.add(id, vec);
  return prior ? "updated" : "added";
}
