# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`README.md` is the spec being built: **FlowState**, click Pette, speak, and she sends a Slack, Discord or email message with the code and a summary of what changed. It is shown on screen as the **Pette** sprite. All five README modules and the VS Code extension are implemented.

The computer-vision layer draws on the two `brainstorming/` specs:

- `BUILD.md`: webcam face, eye and pinch signals.
- `VISUAL_CODING_COMPANION.md`: screen context to a coding-agent harness, plus a proactive policy.

Both specs are references only. This project is Python and PySide6, not the Electron stack in `VISUAL_CODING_COMPANION.md`.

## Git workflow

Commit and push often, after every meaningful step, without asking first. Small or work-in-progress commits are fine. Work on the `dev` branch, never directly on `main`.

## Commands

```powershell
py -m pip install -r requirements.txt
py main.py [--no-voice] [--demo] [--no-camera] [--no-world]   # needs GROQ_API_KEY unless --demo (see .env.example)
py tools/get_models.py                  # MediaPipe models for the webcam features
py -m pytest -q                         # all tests; conftest forces QT_QPA_PLATFORM=offscreen
py -m pytest tests/test_pipeline.py::test_pause_drops_late_results -q   # single test
py tools/prep_sprites.py                # rebuild assets/pette_sheet.png (needs Pillow + numpy)
py tools/sandbox.py [--demo] [--live-webhooks] [--real-ide]   # manual test panel around the real app
```

`tools/sandbox.py` reuses `main.build()`; keep wiring changes in `build()` so the sandbox stays the real app. It sets env vars before importing `main`/`pet` (they read settings at import time). Add new user-facing features to its `SCENARIOS` checklist.

Offscreen Qt has no fonts, so grabbed screenshots show text as boxes.

## Architecture

`pet/pipeline.py` is the hub. Read it first: it connects `Overlay` signals to the flow and owns all state.

- **Threading:** `Pipeline` lives on the Qt main thread. Slow work (listen/transcribe, capture + draft, dispatch) runs through `_run()` in daemon threads, and results come back through the `_deliver(gen, fn)` signal, which Qt queues onto the main thread. Never touch widgets from a worker. The `keyboard` hotkey fires on its own thread, so it must emit `pipeline.hotkey_pressed`, never call `trigger()` directly.
- **Generation guard:** pause increments `gen`. `_on_deliver` drops any result whose `gen` is stale. Any new async step must go through `_run()` to keep this guarantee.
- **One workflow** (`pet/pipeline.py`): click Pette / hotkey / idle pinch → `trigger()` → `listen(on_level, register_cancel)` → `_heard` (transcript shown in the bubble) → `capture()` (file + `git diff`) + `brain.draft` → `_drafted` → countdown proposal → `_approved` dispatches. States: idle, listening, thinking, pending, sending, sent, cancelled, failed, emitted on `Pipeline.status` for the orb and the world window. There is no other flow: no investigate, coding agents or proactive offers (all removed on purpose).
- **Sending:** the user asked for auto-dispatch, so a draft sends itself when the countdown (`delay_s()`: `WorldModel.send_delay_s()`, 2.5–6 s) ends. The safety net is the countdown: Cancel, `pinch_on_proposal` and pause must always stop it. `Bubble.consume()` makes a decision single-use for `Proposal.id`, and dispatch happens only in `_approved(pid)` with a `pending` entry. Never dispatch straight from a brain result, and never let a gesture send.
- **Errors:** each module raises one recoverable exception type (`VoiceError`, `BrainError`, `DispatchError`) with a short user-facing message, shown in the bubble with status `failed`. There are no automatic retries. A `VoiceError` (except "Listening cancelled.") falls back to `Overlay.ask_text()`.
- **LLM provider:** Groq via the `openai` SDK with a Groq `base_url`, all in `pet/llm.py` (key, models, error text). Prompts are text-only (no screenshots). Never call `OpenAI()` directly elsewhere.
- **Voice** (`pet/eleven.py`): ElevenLabs is speech-to-text only. Pette never speaks. Scribe (`ELEVENLABS_API_KEY`) records locally and makes one request. Live mode streams to a public agent's websocket and returns the first `user_transcript`, ignoring agent audio/replies. The agent's spoken greeting swallows the first words unless its First message is cleared in the dashboard (`first_message` overrides are refused by that agent). `main.make_listen()` picks Scribe, live, or Groq Whisper.
- **Brain contract:** `parse_draft` requires exactly the keys `{platform, recipient, subject, message}`, with platform in `PLATFORMS` (`slack`, `discord`, `email`, `none`). The prompt requires a summary of the change. `attach_code()` adds `code` / `code_label` / `code_lang` from the diff, then the selection, then the file. The model never writes the code. `dispatch.compose()` fits message + code to each platform (Discord 2,000). Changing the schema means updating `SYSTEM_PROMPT`, `parse_draft`, `Pipeline._drafted` and the tests together.
- **Dependency injection:** `Pipeline`, `Brain`, `eleven.LiveTranscriber`/`scribe` and `voice.transcribe` take their collaborators (client, capture, listen, send, spawn, connect, audio, post) as parameters. Tests pass fakes and `spawn=lambda f: f()` to run synchronously. Follow this pattern rather than mocking modules.
- **UI** (`pet/ui.py`, shadcn zinc tokens): bubble → `Orb` → `PetWidget`, stacked and bottom-anchored.
  - `Overlay._reanchor` keeps Pette fixed on screen while the bubble resizes. Change bubble content only through `Overlay` methods.
  - `Orb` paints the camera preview, hand skeleton, gesture chip and status ring/pill. `Overlay.set_status()` drives the ring and Pette's pose.
  - `ANIMATIONS` in `sprites.py` maps state names to sheet frames. The frame order is a contract with `tools/prep_sprites.py`. Never load the raw `images/pette-spritesheet.png`, because it has an opaque checkerboard baked in.
- **World-model window** (`pet/worldview.py` + `web/`): a React + Tailwind + shadcn/ui + Aceternity page built by Vite into one file, `pet/web/index.html` (committed; `web/node_modules` is ignored). Edit `web/src`, then run `cd web; npm run build`. Python pushes `WorldModel.snapshot()` + live state via `window.__pette(json)`. The page closes itself by setting `document.title = "close:…"`. Only facts go to the page, never message text or code.
- **IDE bridge:** the VS Code extension and `context.read_active_file` share the JSON schema `{path, workspace, language, content, selection, line, diagnostics[{line, severity, message, source, code}], updatedAt}` at `~/.flowstate_current.json`, overridden by `FLOWSTATE_ACTIVE_FILE`. Keep both sides in sync. `context.git_changes()` runs `git diff HEAD -- <file>` in the file's folder (untracked = whole file as new).

### Webcam, focus, memory
- **Webcam:** `vision.VisionPolicy` holds the debounce rules as pure logic and is unit-tested. `CameraWorker` (a QThread) emits event names, blink baselines, and a `frame` preview (`preview()`: a mirrored 168px `QImage` that owns its pixels, hand points, `gesture_name()`). The preview is only painted on the orb, never stored or sent. `Watcher` owns the camera (on by default in `main.py`) and maps events to reactions. `Overlay.pinch()`: idle → talk; countdown → cancel; plain message → dismiss.
- **Focus gate** (`pet/focus.py`): `Overlay.context_allowed` is False while the foreground app isn't VS Code or a terminal.
  - `classify()` returns None (neutral) for Pette's own process and for taskbar/tray window classes. Otherwise clicking Pette or the tray would hide her.
  - After a temporary hide, use `Overlay.reveal()`, never `show()`, so the gate isn't overridden.
  - `Pipeline._can_start()` blocks clicks/hotkeys while hidden.
- **Memory** (`db/`): `Store` is a thread-safe SQLite fact log, and `WorldModel` holds small explainable rules over it (usual platform, send countdown, blink threshold, presence). `snapshot()` feeds the window.
  - Only pass facts in: never message text, code, audio or frames. `tests/test_db.py` checks the schema for content columns.
  - `Pipeline` and `Watcher` take an optional `memory=` and `world=`. Keep them optional so tests and features run without a db.
  - Webcam events are ignored and not logged while paused.
- **Checks:** `py tools/run_checks.py` regenerates `test_results/` (REPORT.md, junit.xml, ui_states.png). Re-run it and commit the results after significant changes. `tools/sandbox.py` is the manual test panel around `main.build()`; add new user-facing features to its `SCENARIOS`.
- MediaPipe models live in `models/*.task` (gitignored) and are downloaded with `py tools/get_models.py`. `fixtures/buggy_app` is the demo file, and it is excluded from pytest by `pytest.ini`.

## Conventions

- Windows 11 first; PySide6 for UI and screen capture (not `mss`).
- Audio, file contents, diffs and camera frames stay in memory; `Pipeline` drops `ctx` after each draft. Never persist or log them, or API keys. No screenshots.
- File text, diffs and model output are untrusted. The model can only choose among the allowlisted platforms. It never chooses URLs, commands or arbitrary mentions.
