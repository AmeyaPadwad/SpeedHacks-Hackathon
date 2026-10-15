/** Test helpers for the policy and runtime suites (hand-built elements, observations and memories). */
import {
  type MemoryItem,
  type OCRElement,
  type RetrievedMemory,
  type ScreenObservation,
  SourceType,
  type WindowInfo,
} from "../types";

export function el(text: string, bbox: [number, number, number, number] = [0.05, 0.1, 0.65, 0.13],
  extra: Partial<OCRElement> = {}): OCRElement {
  return { text, bbox, confidence: 1, source: "ocr", uiRole: null, windowId: "", timestampMs: 0, ...extra };
}

export function win(appName = "code", title = "app.py", extra: Partial<WindowInfo> = {}): WindowInfo {
  return { appName, title, windowId: "", surface: "window", ...extra };
}

/** Elements stacked down the screen, one per text. */
export function obsOf(texts: string[], opts: { app?: string; title?: string; version?: number } = {}): ScreenObservation {
  return {
    window: win(opts.app ?? "code", opts.title ?? "app.py"),
    elements: texts.map((t, i) => el(t, [0.05, 0.05 + 0.04 * i, 0.65, 0.08 + 0.04 * i])),
    version: opts.version ?? 1,
    timestampMs: 0,
  };
}

export function memItem(id: string, sourceType: SourceType, textSummary: string,
  extra: Partial<MemoryItem> = {}): MemoryItem {
  return {
    id, sourceType, userApproved: true, textSummary, sourceUri: null, tags: [], provenance: {},
    createdAt: "", updatedAt: "", expiresAt: null, ...extra,
  };
}

export function mem(id: string, kind: SourceType, summary: string, score = 0.8, approved = true,
  extra: Partial<MemoryItem> = {}): RetrievedMemory {
  return { item: memItem(id, kind, summary, { userApproved: approved, ...extra }), embedding: new Float32Array(8).fill(1), score };
}
