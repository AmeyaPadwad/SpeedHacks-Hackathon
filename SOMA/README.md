# SOMA: Screen-Observing Memory and Action

SOMA is a local-first desktop companion. It looks at the current screen (only
with your consent), combines what it sees with personal memory you have
approved, and **suggests** a next step. Every learnable part is a PyTorch
`nn.Module` in this repo, and no remote LLM stands in for the model. SOMA
never clicks, types, or runs anything for you.

The full specification is in [SOMA_BUILD_PLAN.md](SOMA_BUILD_PLAN.md).
Design choices are logged in [docs/DECISIONS.md](docs/DECISIONS.md).

## Quick start (Windows, from this folder)

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e ".[capture,ui,ocr,dev]"

.venv\Scripts\python.exe -m scripts.run_desktop --synthetic   # consent-first UI on synthetic screens
.venv\Scripts\python.exe -m scripts.run_desktop               # real screen, after you grant capture
.venv\Scripts\python.exe -m pytest                            # 462 tests, about 60 s on CPU
```

`--smoke` builds the window offscreen, runs one synthetic step, prints the
score distribution and exits.

## Commands

| Command | What it does |
|---|---|
| `python -m scripts.run_desktop [--synthetic] [--checkpoint p]` | Consent dialog, live OCR/app view, PyTorch score distribution (includes `NO_SUGGESTION`), suggestion card, memory viewer, debug overlay |
| `python -m scripts.build_memory --folder <dir> --grant` | Index one user-selected folder (`--list`, `--forget <id>`, `--export <path>`) |
| `python -m scripts.train_ranker [--small]` | Seeded offline training on synthetic data; saves a state-dict checkpoint |
| `python -m scripts.evaluate --checkpoint p` | Compares the full model with the no-memory and scripted baselines |
| `python -m scripts.ablations` | M5 ablations, one variable at a time |
| `python -m scripts.benchmark` | P50/P95 latency per stage on this machine |

## Architecture

```
capture (consent, exclusions, change detection) ─► perception (OCR + a11y + parser + redaction)
   ─► SomaModel: VisualEncoder + TextLayoutEncoder ─► ScreenFusion ─► TemporalEncoder
                 ─► MemoryFusion (gated, NO_MEMORY null slot) ─► ActionPolicy [B, A+1]
memory (SQLite + vector index, permission prefilter) ───────────┘
policy: template candidates ─► CandidateEncoder ─► scores ─► SafetyGate (threshold, cooldown,
        duplicates, quiet apps, grounding) ─► Suggestion or silence ─► feedback ─► learning
```

| Package | Contents |
|---|---|
| `soma/models` | All neural modules, the `SomaBatch` contract, checkpoints (`weights_only=True`) |
| `soma/capture` | Consent-gated `ScreenCapturer`, foreground window, redaction, accessibility |
| `soma/perception` | RapidOCR / synthetic OCR adapters, screen parser, featurization, synthetic GUI renderer |
| `soma/memory` | `MemoryStore`, `VectorIndex`, PyTorch embedders, `Ingestor`, `Retriever`, deletion, export |
| `soma/policy` | Candidate generator, encoder, grounding, safety gate, `ApprovalLedger` (no executors) |
| `soma/learning` | Feedback, replay buffer, datasets, losses, trainer, evaluator, online ranker, controlled rollout |
| `soma/runtime` | `SomaRuntime`, `InferenceScheduler`, rolling history, telemetry, profiling |
| `soma/api` | Authenticated localhost-only HTTP API (status, suggestion, feedback, pause, memories) |
| `soma/ui/desktop` | PySide6 consent dialog, main window, tray, memory viewer, debug overlay |

## Privacy and safety

- Every scope is opt-in and off by default, and each can be toggled on its
  own: capture, app monitoring, folder indexing, feedback storage, training,
  remote inference. Remote inference is not available in v0.1.
- Pause is instant. Password managers, private windows, banking and sign-in
  prompts are excluded before any pixels are grabbed.
- Screenshots stay in memory and are never written to disk by default. Secrets
  are redacted from OCR text and summaries.
- Memory is checked against permissions **before** vector scoring. Deleting a
  memory removes its row, its vector, any thumbnail and any feedback
  references, and the deletion is verified.
- Screen and memory text are treated as untrusted. Candidate text comes only
  from templates, and injection-like content is dropped.
- Weight updates go through `ControlledRollout`, which needs training consent,
  evaluates on a held-out set and a forgetting check, and rolls back bitwise on
  failure.

## Results so far (synthetic only; not real-world quality)

See [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

- Holdout top-1: full model 0.994, no-memory baseline 0.854, scripted type rule 0.130.
- Ablations: OCR+visual 0.998 vs visual-only 0.921; with memory 0.998 vs without 0.937.
- The temporal ablation shows no difference yet, because the synthetic history
  carries no signal.
- Latency on a Ryzen 9 8940HX CPU: model forward 14.5 ms P50. RapidOCR on a
  1280x800 frame is the bottleneck at about 580 ms P50.

## Status vs milestones

| Milestone | State |
|---|---|
| M0 PyTorch modules, shapes, masks, checkpoints | Done |
| M1 Consent-gated capture, OCR + boxes, debug overlay | Done |
| M2 Folder indexing, retrieval with permissions, deletion | Done |
| M3 Candidates, temporal + memory fusion, scorer, gate, UI, feedback | Done (scripted scenarios) |
| M4 Feedback, replay, offline train/eval, controlled rollout | Done (synthetic) |
| M5 Ablations | Partly done: multi-scale vision and history-dependent synthetic scenarios are open |

Known gaps: the online ranker's default bias is too small to override a
confident base model; retrieval has no minimum-relevance floor (D14); and real
consented GUI sessions with labels are still needed before any quality claim.
