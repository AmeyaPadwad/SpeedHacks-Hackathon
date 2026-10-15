/**
 * The latest suggestion as plain data (port of `suggestion_payload` in SOMA/soma/api/local_api.py).
 *
 * No OCR text beyond the suggestion itself leaves here. `file` / `line` / `memorySource` are refs inside the
 * user's allowed folders only, so Pette may offer to open them; nothing is opened or executed here.
 */
import type { SuggestionPayload } from "../service";
import type { MemoryItem } from "../types";
import type { SomaRuntime } from "./pipeline";

const round4 = (x: number): number => Math.round(x * 1e4) / 1e4;

/** The file behind a memory, only while it is still an authorized file. */
async function memorySource(runtime: SomaRuntime, getItem: ((id: string) => Promise<MemoryItem | null>) | null,
  memoryRef: string | null): Promise<string | null> {
  if (!memoryRef || !getItem) return null;
  try {
    const uri = (await getItem(memoryRef))?.sourceUri ?? null;
    return uri && runtime.authorizedFile(uri) ? uri : null;
  } catch {
    return null;
  }
}

export async function suggestionPayload(runtime: SomaRuntime,
  getItem: ((id: string) => Promise<MemoryItem | null>) | null = null): Promise<SuggestionPayload> {
  const r = runtime.lastResult;
  if (r === null) return { available: false, silent: true };
  const s = r.suggestion;
  const distribution: Record<string, number> = {};
  for (const [k, v] of Object.entries(s.distribution)) distribution[k] = round4(v);
  const out: SuggestionPayload = {
    available: true, id: s.id, silent: s.candidate === null, confidence: round4(s.confidence),
    screenVersion: s.screenVersion, frameVersion: r.frameVersion, distribution,
    valid: runtime.currentSuggestionValid(r.frameVersion), app: r.window.appName,
  };
  const c = s.candidate;
  if (c !== null) {
    const file = c.fileRef !== null && runtime.authorizedFile(c.fileRef) ? c.fileRef : null;
    Object.assign(out, {
      action: c.action, text: c.text, evidence: [...c.evidence],
      targetBox: c.target ? [...c.target.bbox] as [number, number, number, number] : null,
      memoryRef: c.memoryRef, template: c.template, file, line: file ? c.line : null,
      memorySource: await memorySource(runtime, getItem, c.memoryRef),
    });
  }
  return out;
}
