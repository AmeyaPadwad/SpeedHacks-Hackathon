/**
 * Loads the real SOMA (`src/soma/index.ts`, `createSoma()`) lazily, so tfjs and the OCR engine stay out of
 * the first paint and the `?sandbox` mode never loads them.
 */
import type { SomaService } from "@/soma/service";

export async function loadSoma(): Promise<SomaService> {
  // @ts-ignore -- src/soma/index.ts is written by the SOMA port; the integrator removes this line.
  const mod: { createSoma: () => SomaService | Promise<SomaService> } = await import("@/soma");
  return await mod.createSoma();
}
