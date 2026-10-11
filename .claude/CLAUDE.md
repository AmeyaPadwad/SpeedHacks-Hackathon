# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`README.md` is the spec being built: **FlowState**, a hotkey + voice + screenshot + active-file agent that drafts Slack, Discord or email messages for approval. It is shown on screen as the **Pette** sprite. All five README modules and the VS Code extension are implemented.

The computer-vision layer draws on the two `brainstorming/` specs:

- `BUILD.md`: webcam face, eye and pinch signals.
- `VISUAL_CODING_COMPANION.md`: screen context to a coding-agent harness, plus a proactive policy.

Both specs are references only. This project is Python and PySide6, not the Electron stack in `VISUAL_CODING_COMPANION.md`.

## Git workflow

Commit and push often, after every meaningful step, without asking first. Small or work-in-progress commits are fine. Work on the `dev` branch, never directly on `main`.

## Commands

```powershell
py -m pip install -r requirements.txt
py main.py [--no-voice] [--demo] [--camera]   # run; needs OPENAI_API_KEY unless --demo (see .env.example)
py tools/get_models.py                  # MediaPipe models for the webcam features
py -m pytest -q                         # all tests; conftest forces QT_QPA_PLATFORM=offscreen
py -m pytest tests/test_pipeline.py::test_pause_drops_late_results -q   # single test
py tools/prep_sprites.py                # rebuild assets/pette_sheet.png (needs Pillow + numpy)
```

Offscreen Qt has no fonts, so grabbed screenshots show text as boxes.

## Architecture

`pet/pipeline.py` is the hub. Read it first: it connects `Overlay` signals to the flow and owns all state.

- **Threading:** `Pipeline` lives on the Qt main thread. Slow work (record, transcribe, draft, dispatch) runs through `_run()` in daemon threads, and results come back through the `_deliver(gen, fn)` signal, which Qt queues onto the main thread. Never touch widgets from a worker. The `keyboard` hotkey fires on its own thread, so it must emit `pipeline.hotkey_pressed`, never call `trigger()` directly.
- **Generation guard:** pause increments `gen`. `_on_deliver` drops any result whose `gen` is stale. Any new async step must go through `_run()` to keep this guarantee.
- **Approval:** `Bubble._consume()` makes Send single-use and ties it to `Proposal.id`. `Pipeline.pending` maps that ID to a `Draft`, and dispatch happens only in `_approved(pid)`. Never dispatch from a timer, from the brain result, or without a pending entry.
- **Errors:** each module raises one recoverable exception type (`VoiceError`, `BrainError`, `DispatchError`) with a short message suitable for the user. The pipeline shows it in the bubble. There are no automatic retries. A `VoiceError` falls back to `Overlay.ask_text()` for a typed command.
- **Brain contract:** `parse_draft` requires exactly the keys `{platform, recipient, subject, message}`, with platform in `PLATFORMS`. `none` means "answer the question", shown as text with no Send button. Changing the schema means updating `SYSTEM_PROMPT`, `parse_draft`, `Pipeline._drafted` and the tests together.
- **Dependency injection:** `Pipeline`, `Brain` and `voice.transcribe` take their collaborators (client, capture, record, send, spawn) as parameters. Tests pass fakes and `spawn=lambda f: f()` to run synchronously. Follow this pattern rather than mocking modules.
- **UI:**
  - `Overlay._reanchor` keeps Pette fixed on screen while the bubble resizes. Change bubble content only through `Overlay` methods.
  - `ANIMATIONS` in `sprites.py` maps state names to sheet frames. The frame order is a contract with `tools/prep_sprites.py`.
  - Never load the raw `images/pette-spritesheet.png`, because it has an opaque checkerboard baked in.
- **IDE bridge:** the VS Code extension and `context.read_active_file` share the JSON schema `{path, workspace, language, content, selection, line, diagnostics[{line, severity, message, source, code}], updatedAt}` at `~/.flowstate_current.json`, overridden by `FLOWSTATE_ACTIVE_FILE`. Keep both sides in sync.

### CV → coding agents
- **Screen CV:** `context.capture()` returns a `Context(screenshot_jpeg, active_file, screen)`, where `screen` holds the `screen_cv.ScreenFindings` for the same frame. `screen_cv` is a red-hue locator, not OCR. Its shape and density filters (`MIN_ASPECT`, `MAX_DENSITY`) were tuned against false positives from real UI, so re-check on a real screen after changing them.
- **Investigate flow** (`Pipeline.investigate`, the `agent` brain platform, or an approved `offer_investigation`): `_capture_then`, then `analyst.analyze`, then a proposal, then `_run_agent` on approval. `pending` values are a `Draft`, an `AgentTask` or the `OFFER` marker. `_approved` dispatches on type.
- **Agents** (`pet/agents.py`) are propose-only by construction:
  - Claude Code runs with `-p --permission-mode plan --tools Read,Grep,Glob` and the prompt on stdin.
  - Hermes runs with `-z ... -t todo`, because `-z` auto-approves tools.
  - Never add write or terminal tools, `--yolo`, or permission bypasses. `tests/test_agents.py` asserts the argument lists.
  - Only add an adapter after checking the CLI's real `--help`.
- **Webcam:** `vision.VisionPolicy` holds the debounce rules as pure logic and is unit-tested. `CameraWorker` (a QThread) emits event names only, never frames or landmarks. `Watcher` owns the camera and the poll timer, and gates proactive offers on presence. Gestures call `Overlay.pinch()`, which must never approve or cancel a proposal.
- **Proactive offers:** `watcher.InterventionPolicy` handles persistence, fingerprint dedupe, the cooldown and the daily cap, and it defers while the user is away, has eyes closed, or is busy. An offer only asks. The screen is captured only after the user approves.
- **Focus gate** (`pet/focus.py`): `Overlay.context_allowed` is False while the foreground app isn't VS Code or a terminal.
  - `classify()` returns None (neutral) for Pette's own process and for taskbar/tray window classes. Otherwise clicking Pette or the tray would hide her.
  - After a temporary hide, use `Overlay.reveal()`, never `show()`, so the gate isn't overridden.
  - `Pipeline._can_start()` blocks hotkeys and offers while hidden. The watcher treats hidden as busy, so offers are deferred, not lost.
- **Gestures in the message flow:** `Overlay.pinch()` does three things. Idle: push-to-talk. A plain message is showing: dismiss it. A proposal is pending: emit `pinch_on_proposal`, and `Pipeline._pinch_on_proposal` re-dictates only if that proposal is a message `Draft`.
  - A gesture must never approve or send.
  - `Overlay.nudge()` re-titles a pending proposal "Still …?" when you return from being away.
- **Memory** (`db/`): `Store` is a thread-safe SQLite fact log, and `WorldModel` holds small explainable rules over it.
  - Only pass facts in: never message text, screenshots, frames or code. `tests/test_db.py` checks the schema for content columns.
  - `Pipeline` and `Watcher` take an optional `memory=` and `world=`. Keep them optional so tests and features run without a db.
  - Webcam events are ignored and not logged while paused.
- **Checks:** `py tools/run_checks.py` regenerates `test_results/` (REPORT.md, junit.xml, ui_states.png). Re-run it and commit the results after significant changes.
- MediaPipe models live in `models/*.task` (gitignored) and are downloaded with `py tools/get_models.py`. `fixtures/buggy_app` is a deliberate bug for demos and is excluded from pytest by `pytest.ini`.

## Conventions

- Windows 11 first; PySide6 for UI and screen capture (not `mss`).
- Screenshot, audio and file contents stay in memory, and `Pipeline._reset()` drops the context after each request. Never persist or log them, or API keys.
- Screen text, file text and model output are untrusted. The model can only choose among the allowlisted platforms. It never chooses URLs, commands or arbitrary mentions.
