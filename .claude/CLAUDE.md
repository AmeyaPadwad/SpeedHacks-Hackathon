# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`README.md` is the spec. **Pette** is only the UI; **SOMA** (`SOMA/`, its own PyTorch package and venv) is the backend. SOMA looks at the screen (with consent) and the folders the user indexed, and ranks next steps. Pette shows each one as a glass card; Open / Copy (or a pinch) is the only way anything happens. Messaging (Slack, Discord, email), Darwin, Groq and voice were removed on purpose; don't bring them back.

Pette has no sprite or image files: the camera orb is her body (click = look now, drag = move, moods drawn on it) and the tray icon is drawn in code. Suggestions show in a speech bubble at the bottom of the screen. Gestures: pinch = Do it, point left/right = previous/next browser tab, pinch-and-drag = browser scroll.

`brainstorming/` holds two reference specs (webcam signals; screen-context companion). They're references only: this is Python + PySide6, not the Electron stack described there.

## Git workflow

Commit and push often, after every meaningful step, without asking first. Small or work-in-progress commits are fine. Work on the `dev` branch, never directly on `main` (the SOMA integration lives on `integration`).

## Commands

```bash
./run_pette_soma.sh [--diagnose] [pette flags]      # Git Bash: both venvs (uv), SOMA serve, then Pette --soma <endpoint>
.venv/Scripts/python.exe -m pytest -q               # Pette tests; conftest forces QT_QPA_PLATFORM=offscreen
.venv/Scripts/python.exe -m pytest tests/test_soma.py::test_click_looks_now_then_card_or_nothing -q
cd SOMA; .venv/Scripts/python.exe -m pytest -q      # SOMA's own suite
.venv/Scripts/python.exe tools/run_checks.py [--no-camera] [--no-soma-tests]   # regenerates test_results/
.venv/Scripts/python.exe tools/sandbox.py [--real-soma EP]   # real app + panel around a scripted fake SOMA
.venv/Scripts/python.exe tools/get_models.py        # MediaPipe models (gitignored)
.venv/Scripts/python.exe tools/cv_check.py          # live webcam/gesture signals (never sends input)
cd web; npm ci; npm run build                       # rebuild pet/web/index.html (committed) after editing web/src
```

Pette and SOMA use separate venvs (`.venv`, `SOMA/.venv`): mediapipe needs opencv-contrib and RapidOCR pulls opencv-python. They talk only over SOMA's localhost API. `tools/sandbox.py` reuses `main.build()`; keep wiring there so the sandbox stays the real app, and add user-facing features to its `SCENARIOS`. Offscreen Qt has no fonts, so offscreen screenshots show text as boxes (run_checks renders on the real display).

## Architecture

`pet/soma.py` is the hub. Read it first.

- **SomaClient** wraps SOMA's API (`SOMA/soma/api/local_api.py`): `/suggestion`, `/feedback`, `/trigger` (look now), `/pause`, `/resume`, `/folders` (GET, POST add + background index), `/folders/remove`. Localhost URLs only; `request` is injectable. The endpoint file `{url, token}` comes from `SOMA/scripts/serve.py --endpoint-file` (`PETTE_SOMA_ENDPOINT` / `--soma`).
- **Suggester** lives on the Qt main thread. A worker polls `/suggestion` every second (`poll_once`, emits `_arrived`); every other call goes through `_async(work, on_ok, on_err)`, which returns via the `_result` signal. Never touch widgets from a worker. Hotkeys fire on the keyboard thread and must only emit signals (`look_requested`, `toggle_requested`, `Overlay.camera_view_requested`, `Overlay.world_requested`).
- **Cards:** shown only when `_free()` (enabled, not paused, `context_allowed`, no proposal up) via `Overlay.propose(Proposal(..., cancel_label="Not now", glow=True))`, with a 15 s `set_progress` bar. `Bubble.consume()` makes a decision single-use; pinch goes through `_pinched` → `consume()`. Outcomes: accepted / rejected (Not now) / ignored (expired, paused, toggled off; never a dismissal). Each is sent to SOMA and logged as `offers(kind, outcome)`.
- **Do it** is `pet/actions.py` only: `opener(s)` picks `file`+`line` or `memory_source` (SOMA only sends paths inside the user's allowed folders); `open_file` uses `code -g` when on PATH, `startfile` for images only, else `explorer /select`. Never `startfile` a non-image, never a shell, never run anything. Otherwise copy the text.
- **Statuses** (`Suggester.status` → `Overlay.set_status` and the world window): idle, looking, suggesting, done, dismissed, indexing, failed. `STATUS` in `ui.py` maps them to ring colour/label.
- **Muting** (`WorldModel.muted_kinds`): a kind answered Not now ≥ 80 % over ≥ 4 answers is skipped unless the user clicked "look now".
- **exclude_from_capture()** keeps Pette's windows out of SOMA's screenshots; apply it to any new top-level window.
- **Dependency injection:** `Suggester`, `SomaClient`, `actions.open_file`, `Watcher` take collaborators as parameters (client, spawn, clipboard, open_file, request). Tests pass fakes and `spawn=lambda f: f()`; keep a reference to a `Suggester` in tests or Qt drops its connections.
- **UI** (`pet/ui.py`): `Overlay` is Pette's window and holds only the `Orb` (click/drag, `set_mood`, `feed_visible`). `Overlay.bar` is a `BubbleBar`, a separate Qt.Tool window parented to the Overlay, docked bottom-centre of Pette's screen; it slides left rather than cover her. Never give the bar closures that capture the Overlay (a Python cycle through a Qt child crashed at GC); it reads its parent instead. Change bubble content only through `Overlay` methods (`say`, `propose`, `dismiss`, `set_progress`); `bar.reanchor` sizes wrapped text by height-for-width. `bubble_enabled` (Ctrl+Alt+B) gates `say`/`propose` and `Suggester._free()`. The bubble is frosted glass (`GLASS_*` alphas).
- **Gestures** (`pet/vision.py`, `pet/watcher.py`, `pet/browser.py`): `VisionPolicy` is pure logic (pinch hold vs pinch-drag, pointing hold/rearm/cooldown, `take_scroll()`); `pointing()` maps your left to the image's +x. `Watcher` routes POINT_LEFT/RIGHT to `BrowserControl.switch_tab(-1/+1)` and `scroll` amounts to `BrowserControl.scroll`, which acts only when `focus.foreground_info()` is a browser and only with Ctrl+(Shift+)Tab and the wheel. Inject `foreground`/`send_keys`/`wheel` in tests; never send real input from a test.
- **World-model window** (`pet/worldview.py` + `web/`): React + Tailwind + shadcn/ui + Aceternity, built by Vite into `pet/web/index.html`. Python pushes `WorldModel.snapshot()` + live state + folder names/counts via `window.__pette(json)`. Facts only: never suggestion text, file names or screen content.

### SOMA side (what Pette relies on)
- `SOMA/configs/pette.yaml` + `policy.context_templates` (`soma/policy/context_candidates.py`: review_changes, write_tests, commit_work, take_notes) are why SOMA speaks on real screens. Keep SOMA's defaults (`dev.yaml`, flag off) unchanged.
- `Candidate.template` (part of the key) and `Candidate.line` (not part of the key) are in the API payload with `file` and `memory_source` (only when `runtime.authorized_file` allows the path).
- `SOMA/tests/conftest.py` destroys leftover Qt widgets at session end; without it the suite crashes at exit after passing.

### Webcam, focus, memory
- **Webcam:** `vision.VisionPolicy` holds the debounce rules as pure logic. `CameraWorker` (QThread) emits events and a `frame` preview drawn only on the orb, never stored or sent. `Watcher` maps events to reactions; pinch → `Overlay.pinch()` → card "Do it", otherwise ignored. `Overlay.set_camera_visible` hides the orb only (camera keeps running).
- **Focus gate** (`pet/focus.py`): off by default; `--focus-only` or the tray shows Pette only over VS Code/terminals. `classify()` returns None for Pette's own process and tray/taskbar classes. After a temporary hide use `Overlay.reveal()`, never `show()`.
- **Memory** (`db/`): thread-safe SQLite fact log (`events`, `offers`, `blink_baseline`; the old `messages` table is dropped at startup) and `WorldModel` rules. Only facts go in; `tests/test_db.py` checks for content columns. `memory=`/`world=` stay optional.

## Conventions

- Windows 11 first; PySide6 for UI.
- Pette takes no screenshots and records no audio; only SOMA captures the screen, in memory, with consent. Camera frames stay in memory. Never persist or log screen text, suggestion text, file contents or paths in Pette's db.
- Screen text, file contents and SOMA's suggestion text are untrusted: they're displayed or copied, never executed, and never turned into commands or URLs.
