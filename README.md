# Pette - the on-screen face of SOMA

**Pette** is a glowing camera orb. **SOMA** is the brain behind her: it looks at a screen or window you share (with your consent), searches the folders you let it index, and ranks what you might want to do next with its own neural model, running in your browser on [TensorFlow.js](https://www.tensorflow.org/js). Pette shows each suggestion in a glass speech bubble. Nothing happens until you press **Open** / **Copy** (or pinch).

The whole thing is a static web app in TypeScript. There is no server: screen frames, OCR, the model, your folder index and Pette's fact log all live in the browser tab. Pette sends no messages, calls no cloud AI and records no audio.

## Quick start

```bash
npm ci
npm run dev        # http://localhost:5173  (?sandbox: scripted fake SOMA, ?debug: window.__soma)
npm run build      # static site in dist/ (any static host; GitHub Pages via .github/workflows/web.yml)
npm test           # vitest (jsdom, fake IndexedDB; never the real screen or camera)
npm run typecheck
```

Use a recent Chrome or Edge (Firefox works without folders and floating). Screen capture, the webcam and folder access need a secure context: `https://` or `localhost`.

1. **Share a screen** on the first-run page. The browser's picker is your opt-in; SOMA sees only what you share and stops when you stop sharing or pause.
2. **Float Pette** (settings) to keep the orb and bubble on top of your other apps (Document Picture-in-Picture).
3. **Add a folder** (settings): your notes, a repo or a project. SOMA indexes it (text, code, image names; size-capped) and starts pointing at those files.

On first use the browser downloads the model weights it needs (tesseract.js OCR data and the TF.js hand and face models); your data never goes the other way.

## How it works

```
shared screen ─► SOMA: capture (change detection, exclusions) → OCR (tesseract.js) → redaction
your folders ──┘      → screen + memory fusion (TF.js) → candidates → online ranker → safety gate
                                                                                      │
Pette: speech bubble ("SOMA suggests · Open file")  ◄── SomaService (in-page) ◄──────┘
       ── Open / Copy / Not now / pinch ──► feedback back to SOMA (trains the ranker)
```

| SOMA sees | Card | Open / Copy does |
|---|---|---|
| A `file.py:40` reference or traceback, in a folder you indexed | *Open calc.py at line 40* | Shows the file in Pette's viewer at that line |
| Something related in your indexed files | *Compare with your earlier document "…"* | Shows that file |
| An error | *Explain the error "…"* | Copies the text for your coding agent |
| A file in your editor | *Review your changes to "app.py" before you commit*, *Write a test for the code in "app.py"* | Copies the text |
| A passing test run | *The tests pass: commit your work* | Copies the text |
| A web page | *Save what you learn from "…" to your notes* | Copies the text |

- A web page can't see which app is in front, so SOMA infers it (editor, terminal, browser) from the shared window's title and on-screen text.
- **Click Pette** (or `Ctrl+Shift+Space`) to have SOMA look now; otherwise it checks every 2 s and stays quiet most of the time (45 s between cards, no repeats for 5 minutes).
- A card stays 15 s (a bar drains under it). Ignoring it is not a *Not now*.
- **Learning:** every answer goes to SOMA's online ranker (kept in IndexedDB). Pette also mutes a kind of suggestion you answer *Not now* to at least 80 % of the time (after 4 answers). The world-model panel shows why.
- **Opening never runs anything:** files open read-only in Pette's own viewer (text with the line highlighted, images inline, anything else as a download).

### Pette, the camera orb
The orb shows **you** (a mirrored webcam preview, drawn locally only), your hand skeleton, the gesture it sees, and a status ring: blue *Looking*, amber *Suggestion*, green *Done ✓*, grey *Not now*, violet *Indexing*, red *SOMA offline*. Click it to look now; drag it to move her. She dozes (*zZ*) while you're away or paused, and a lid comes down when your eyes close. The camera is optional and off until you turn it on.

### Gestures (TF.js hand and face landmarks)
| Gesture | Effect |
|---|---|
| **Pinch** (thumb to index), held still ~0.5 s, a card is up | Same as **Open** / **Copy** |
| **Point left / right** | Previous / next of your recent suggestions in the bubble |
| **Pinch and drag** up / down | Scrolls the world-model panel, touch-style |
| **Away → back** | An open card is re-titled *"While you were away · …"* |
| **Eyes closed** | Pette shuts her eyes too |

A web page can't switch your browser tabs or scroll other apps, so gestures act inside Pette. Frames are never stored, logged or uploaded; the fact log gets gesture names and timings only.

### Hotkeys (while Pette's page or floating window has focus)
| Keys | Action |
|---|---|
| Ctrl+Shift+Space | Look at my screen now (same as clicking Pette) |
| Ctrl+Alt+S | SOMA suggestions on / off |
| Ctrl+Alt+B | Speech bubble on / off |
| Ctrl+Alt+C | Hide / show the camera picture in the orb (gestures still work) |
| Ctrl+Alt+W | Hide / show the world-model panel |

### The world-model panel
Live status, suggestions shown / done / not now per kind (and which are muted), the folders SOMA indexed, the explainable rules in effect, presence over the last 24 hours, and recent kinds + outcomes. Facts only: never the suggestion text, file names or screen content.

## Architecture

| Part | Path | What it does |
|---|---|---|
| Contracts | `src/soma/service.ts`, `src/soma/types.ts`, `src/pette/contracts.ts` | `SomaService` (the old localhost API's verbs, now in-page), shared types |
| SOMA capture | `src/soma/capture/` | `getDisplayMedia` (Pette's own tab excluded), change detection, app inference, exclusions, redaction |
| SOMA perception | `src/soma/perception/` | tesseract.js OCR, screen parsing (errors, file refs, buttons), hashed tokenizer, featurization |
| SOMA memory | `src/soma/memory/` | File System Access folder walk, summaries + hashed embeddings in IndexedDB, retrieval, forget-folder |
| SOMA model | `src/soma/models/` | TF.js: ViT visual encoder, text-layout encoder, screen fusion, GRU temporal encoder, memory fusion, action policy (index 0 = stay silent) |
| SOMA learning | `src/soma/learning/` | Online preference ranker from accept / not-now feedback, feedback snapshots (no pixels) |
| SOMA policy | `src/soma/policy/` | Candidate generator, app-context templates, candidate encoder, safety gate (threshold, cooldown, duplicates, quiet apps) |
| SOMA runtime | `src/soma/runtime/`, `src/soma/index.ts` | One step per tick, scheduler, `createSoma()` |
| Hub | `src/pette/suggester.ts` | Polls SOMA, shows cards when Pette is free, look-now, folders, feedback, muting |
| Do it | `src/pette/actions.ts` | Show a file from your folders or copy text. Nothing else. |
| UI | `src/pette/ui/`, `src/pette/controller.ts`, `src/components/` | Orb, speech bubble, file viewer, settings, world panel, floating window |
| Webcam | `src/pette/vision/` | TF.js hand-pose and face-landmarks, debounce policy, watcher |
| World model | `src/pette/world/` | IndexedDB fact log → explainable rules |
| Sandbox | `src/pette/fakeSoma.ts`, `src/pette/Sandbox.tsx` | `?sandbox`: the real UI around a scripted fake SOMA, with a manual test checklist |

### Safety rules
- **Nothing runs by itself.** SOMA only suggests; Pette acts only on Open / Copy / pinch, and opening only displays a file.
- **Screen capture is opt-in** (the browser's share picker), pausable, skips password managers, private windows and sign-in prompts, and keeps frames in memory only.
- **Your files:** SOMA reads only folders you pick; removing a folder forgets everything from it. Suggestions only point at files inside those folders.
- **Untrusted text:** screen text, file contents and suggestion text are displayed as text or copied, never executed and never turned into commands or URLs.
- **Pette's fact log stores facts only:** event names, timings, and which kind of suggestion you answered how.

## Status
- SOMA's model is untrained in the browser (seeded weights); the ranker learns from your answers. Suggestion quality on real screens needs training on consented real sessions.
- The Python / PySide6 / PyTorch version lives in git history before this branch.
