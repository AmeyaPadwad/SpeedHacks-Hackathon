# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`README.md` is the spec. Pette is a static web app, all TypeScript: **Pette** (`src/pette/`) is the UI and **SOMA** (`src/soma/`) is the in-browser backend (TensorFlow.js model, tesseract.js OCR, IndexedDB memory). SOMA looks at a screen the user shared (with consent) and the folders they picked, and ranks next steps. Pette shows each one as a glass card; Open / Copy (or a pinch) is the only way anything happens. Messaging (Slack, Discord, email), Darwin, Groq, voice and any server or cloud AI were removed on purpose; don't bring them back. The Python/PySide6/PyTorch version is in git history only.

Pette has no sprite or image files: the camera orb is her body (click = look now, drag = move, moods drawn on a canvas). Suggestions show in a speech bubble. Gestures act inside Pette: pinch = Do it, point left/right = previous/next recent suggestion, pinch-and-drag = scroll the world panel.

`brainstorming/` holds two reference specs (webcam signals; screen-context companion). References only.

## Git workflow

Commit and push often, after every meaningful step, without asking first. Small or work-in-progress commits are fine. The web port lives on `Pette-SOMA`; never commit directly to `main`.

## Commands

```bash
npm ci
npm run dev                         # ?sandbox = real UI around a scripted fake SOMA; ?debug = window.__soma
npm test                            # vitest, jsdom + fake-indexeddb
npx vitest run src/soma/runtime     # one area
npm run typecheck                   # tsc -b (strict)
npm run build                       # tsc -b && vite build -> dist/ (static; CI deploys main to GitHub Pages)
```

## Architecture

`src/soma/service.ts` (`SomaService`) is the only contract between Pette and SOMA; `src/pette/suggester.ts` is Pette's hub. Read both first.

- **SomaService** replaces SOMA's old localhost API with in-page calls: `grantScreen`/`revokeScreen`, `suggestion`, `status`, `feedback`, `trigger` (look now), `pause`/`resume`, `folders`/`addFolder`/`removeFolder`, `readFile`. `createSoma()` (`src/soma/index.ts`) builds the real one and lazily loads tfjs (webgl, else cpu) and tesseract on the first grant. `FakeSoma` (`src/pette/fakeSoma.ts`) is the scripted fake for tests and `?sandbox`; add user-facing features to its scenarios and the `Sandbox.tsx` checklist.
- **SOMA runtime** (`src/soma/runtime/pipeline.ts`): one serialized `step()`: capture → OCR → `inferWindow` → exclusion → parse → redact → featurize → `encodeScreen` once → retrieval → candidates (+ context templates) → blocked mask → candidate encoder → `score` → preference-ranker bias → `SafetyGate`. Collaborators are narrow interfaces in `runtime/deps.ts`, so the runtime never imports tfjs/tesseract and tests use `runtime/testFakes.ts`. Nothing executes; `ApprovalLedger` has no executors.
- **App inference:** a web page can't see the foreground app, so `capture/window.ts` maps the track label and on-screen text to canonical names (`code`, `windowsterminal`, `chrome`, …) that `policy/contextCandidates.ts` (`appKind`) understands. Keep the two in sync.
- **Files:** a folder is a `FileSystemDirectoryHandle` in IndexedDB; refs are `folderId/relative/path`. Only `memory/access.ts` resolves refs (rejects `..`, absolute paths, unknown folders). On-screen paths map to refs via `runtime/files.ts` (`FileRefIndex`, whole trailing path, never ambiguous bare names).
- **Model** (`src/soma/models/`): seeded weights (same untrained model every load), inference in `tf.tidy`, no tensor leaks (tests check `tf.memory()`), weights saved to `indexeddb://soma-model`. Index 0 of logits is NO_SUGGESTION. The ranker in use is `OnlinePreferenceRanker` (JSON in the `soma-runtime` DB); the memory DB is `soma`. Don't share DB names.
- **Suggester** (framework-free, injected clock/timers): polls `suggestion()` every second; cards only when `_free()` (enabled, not paused, bubble on, no proposal up) via the controller's `propose`, 15 s progress bar. `consume()` makes a decision single-use; pinch goes through it. Outcomes: accepted / rejected (Not now) / ignored (expired, paused, toggled off; never a dismissal), each sent to SOMA and logged as `logOffer(kind, outcome)`. Muting: Not now ≥ 80 % over ≥ 4 answers, skipped on look-now.
- **Do it** is `src/pette/actions.ts` only: a file ref opens in Pette's read-only `FileViewer`; otherwise copy the text. Never eval, navigate, `window.open` or build a URL from SOMA's text.
- **UI:** `PetteController` (`src/pette/controller.ts`) is an observable store (`useSyncExternalStore`) implementing `WatcherTarget` and the old Overlay methods (`say`, `propose`, `dismiss`, `setProgress`, `setStatus`, …); change bubble content only through it. Camera frames bypass React state. `pip.ts` floats the orb + bubble with Document Picture-in-Picture. Statuses idle / looking / suggesting / done / dismissed / indexing / failed map to the ring via `status.ts`.
- **Webcam** (`src/pette/vision/`): `VisionPolicy` is pure debounce logic (time passed in); `CameraWorker` runs TF.js MediaPipeHands + FaceMesh (tfjs runtime) with injectable `getUserMedia`/detectors; blink = eye-aspect-ratio mapped to 0..1 (`landmarks.ts`). `Watcher` drives a `WatcherTarget`; paused = no reactions, no logging.
- **World model** (`src/pette/world/`): `FactStore` (IndexedDB, in-memory mirror so `FactLog` is sync) + `WorldModel` rules → `WorldPanel`. Facts only.

## Conventions

- TypeScript strict, React 18, Tailwind + shadcn tokens, `@/` alias. Tests sit next to the code as `*.test.ts(x)`; inject fakes, never the real camera or screen.
- Only SOMA captures the screen, in memory, after the user's share. Camera frames stay in memory and are drawn only on the orb. Never persist or log screen text, suggestion text, file contents or paths in Pette's fact log.
- Screen text, file contents and SOMA's suggestion text are untrusted: displayed as text (never innerHTML) or copied, never executed, never turned into commands or URLs.
