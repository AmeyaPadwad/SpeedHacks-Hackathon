/**
 * SOMA's typed configuration (port of SOMA/soma/config.py with configs/pette.yaml applied).
 *
 * The web build has one config: the "pette" profile. An untrained model spreads its probability over
 * NO_SUGGESTION and the candidates (about 0.3 each), so the gate lets the best-ranked candidate through
 * (low threshold, abstain does not veto) and relies on the cooldown and duplicate window to keep
 * interruptions rare. Feedback trains the online ranker.
 */

export interface ModelConfig {
  dModel: number;
  nHeads: number;
  dropout: number;
  /** Screenshots are resized to imageSize x imageSize before the visual encoder. */
  imageSize: number;
  patchSize: number;
  visionLayers: number;
  /** Hashed OCR token vocabulary. */
  vocabSize: number;
  maxOcrTokens: number;
  textLayers: number;
  fusionLayers: number;
  temporalKind: "gru" | "transformer";
  historyLen: number;
  nApps: number;
  maxMemories: number;
  maxCandidates: number;
  nActionTypes: number;
}

export interface CaptureConfig {
  minIntervalS: number;
  /** Fraction of changed pixels to count as a meaningful change. */
  changeThreshold: number;
  excludedApps: string[];
  excludedTitlePatterns: string[];
}

export interface MemoryConfig {
  topK: number;
  recencyHalfLifeDays: number;
  allowedExtensions: string[];
  maxFileBytes: number;
  /** Per-folder cap on indexed files. */
  maxFilesPerFolder: number;
}

export interface PolicyConfig {
  relevanceThreshold: number;
  marginOverAbstain: number;
  cooldownS: number;
  duplicateWindowS: number;
  quietApps: string[];
  contextTemplates: boolean;
}

export interface SomaConfig {
  seed: number;
  model: ModelConfig;
  capture: CaptureConfig;
  memory: MemoryConfig;
  policy: PolicyConfig;
}

export const DEFAULT_CONFIG: SomaConfig = {
  seed: 0,
  model: {
    dModel: 256,
    nHeads: 8,
    dropout: 0.1,
    imageSize: 224,
    patchSize: 32,
    visionLayers: 2,
    vocabSize: 8192,
    maxOcrTokens: 256,
    textLayers: 2,
    fusionLayers: 2,
    temporalKind: "gru",
    historyLen: 8,
    nApps: 64,
    maxMemories: 8,
    maxCandidates: 8,
    nActionTypes: 6,
  },
  capture: {
    minIntervalS: 2.0,
    changeThreshold: 0.01,
    excludedApps: ["1password", "keepass", "bitwarden", "lastpass", "dashlane"],
    excludedTitlePatterns: ["password", "private browsing", "incognito", "inprivate", "sign in", "log in"],
  },
  memory: {
    topK: 8,
    recencyHalfLifeDays: 30,
    allowedExtensions: [
      ".txt", ".md", ".rst", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml", ".toml",
      ".html", ".css", ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".rb", ".sh", ".csv",
      ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ],
    maxFileBytes: 2_000_000,
    maxFilesPerFolder: 5_000,
  },
  policy: {
    relevanceThreshold: 0.1,
    marginOverAbstain: -1.0,
    cooldownS: 45,
    duplicateWindowS: 300,
    contextTemplates: true,
    // chat and mail apps are where you'd act on a suggestion; never interrupt there
    quietApps: ["slack", "discord", "ms-teams", "teams", "outlook", "olk"],
  },
};
