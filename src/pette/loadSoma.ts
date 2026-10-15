/**
 * Loads the real SOMA (`src/soma/index.ts`, `createSoma()`) lazily, so tfjs and the OCR engine stay out of
 * the first paint and the `?sandbox` mode never loads them.
 */
import type { SomaService } from "@/soma/service";

export async function loadSoma(): Promise<SomaService> {
  const mod = await import("@/soma");
  return await mod.createSoma();
}
