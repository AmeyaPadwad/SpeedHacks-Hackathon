/**
 * The contract between SOMA and Pette. It replaces SOMA's localhost HTTP API (SOMA/soma/api/local_api.py):
 * same verbs, now in-process calls. Pette depends only on this interface, so tests pass a fake.
 *
 * Nothing behind this interface executes anything: it only returns suggestions and records feedback.
 */
import type { FeedbackOutcome } from "./types";

/** Latest suggestion as plain data (`suggestion_payload` in the Python API). */
export interface SuggestionPayload {
  available: boolean;
  silent: boolean;
  id?: string;
  confidence?: number;
  screenVersion?: number;
  frameVersion?: number;
  distribution?: Record<string, number>;
  /** False once the screen changed enough that the suggestion is stale. */
  valid?: boolean;
  app?: string;
  // present only when not silent:
  action?: string;
  text?: string;
  evidence?: string[];
  targetBox?: [number, number, number, number] | null;
  memoryRef?: string | null;
  template?: string;
  /** Authorized file (`folderId/relative/path`) seen on screen, with an optional line. */
  file?: string | null;
  line?: number | null;
  /** Authorized file behind a retrieved memory. */
  memorySource?: string | null;
}

export interface FolderInfo {
  id: string;
  name: string;
  memories: number;
}

export interface FoldersPayload {
  folders: FolderInfo[];
  indexing: boolean;
  lastReport: { files?: number; skipped?: number; error?: string };
}

export interface SomaStatus {
  paused: boolean;
  /** True while a screen stream is granted and live. */
  capturing: boolean;
  steps: number;
  processed: number;
  frameVersion: number;
  lastSkipReason: string | null;
  memories: number;
  /** "loading" while tfjs / OCR warm up. */
  ready: boolean;
}

export interface SomaService {
  /** Ask the user to share a screen or window (getDisplayMedia). Resolves false if they decline. */
  grantScreen(): Promise<boolean>;
  /** Stop capturing and release the stream. */
  revokeScreen(): void;
  /** Start the step loop (every `capture.minIntervalS`). */
  start(): void;
  stop(): void;

  suggestion(): Promise<SuggestionPayload>;
  status(): Promise<SomaStatus>;
  feedback(suggestionId: string, outcome: FeedbackOutcome): Promise<void>;
  /** "Look now": run one step immediately; true if it ran (not paused, screen granted). */
  trigger(): Promise<boolean>;
  pause(): Promise<void>;
  resume(): Promise<void>;

  folders(): Promise<FoldersPayload>;
  /** Prompt with showDirectoryPicker (or take a handle) and index it in the background. */
  addFolder(handle?: FileSystemDirectoryHandle): Promise<FolderInfo | null>;
  /** Forget the folder and every memory from it. Returns the number of memories forgotten. */
  removeFolder(folderId: string): Promise<number>;
  /** Read an authorized file (`folderId/relative/path`) for Pette's file viewer; null when not authorized. */
  readFile(ref: string): Promise<{ name: string; text: string | null; blob: Blob } | null>;
}

/**
 * Pipeline internals for the debug dashboard. Facts and numbers only: no screen text, no file names; the
 * candidates carry their action and template, never their text.
 */
export interface SomaDiagnostics {
  /** Inferred foreground app ("code", "chrome", "" when unknown) and the shared surface. */
  app: string;
  surface: string;
  /** OCR lines read on the last processed frame. */
  ocrLines: number;
  candidates: { action: string; template: string }[];
  /** Label -> probability from the last decision (NO_SUGGESTION included). */
  distribution: Record<string, number>;
  /** Per-stage milliseconds of the last processed step. */
  timingsMs: Record<string, number>;
  /** How many steps ended for each skip reason ("unchanged", "pette-in-front", ...). */
  skips: Record<string, number>;
  /** Suggestions the gate let through since start. */
  spoke: number;
  /** tfjs backend and live tensors / bytes; null before the model loads. */
  tf: { backend: string; tensors: number; bytes: number } | null;
}

export interface SomaServiceWithDiagnostics extends SomaService {
  diagnostics(): Promise<SomaDiagnostics>;
}

export const hasDiagnostics = (s: SomaService): s is SomaServiceWithDiagnostics =>
  typeof (s as Partial<SomaServiceWithDiagnostics>).diagnostics === "function";
