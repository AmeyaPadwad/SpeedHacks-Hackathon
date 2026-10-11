# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`README.md` is the spec being built: **FlowState**, a hotkey + voice + screenshot + active-file agent that drafts Slack, Discord or email messages for approval. It is shown on screen as the **Pette** sprite. All five README modules and the VS Code extension are implemented.

The `brainstorming/` files are alternative specs and are not being built. They are a reference only:

- `BUILD.md`: a webcam desktop pet.
- `VISUAL_CODING_COMPANION.md`: an Electron coding companion.

If a request clearly targets one of them, ask before mixing it in.

## Git workflow

Commit and push often, after every meaningful step, without asking first. Small or work-in-progress commits are fine. Work on the current branch (`sujash-working`), never directly on `main`.

## Commands

```powershell
py -m pip install -r requirements.txt
py main.py [--no-voice]                 # run; needs OPENAI_API_KEY (see .env.example)
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
- **IDE bridge:** the VS Code extension and `context.read_active_file` share the JSON schema `{path, language, content, selection, line, updatedAt}` at `~/.flowstate_current.json`, overridden by `FLOWSTATE_ACTIVE_FILE`. Keep both sides in sync.

## Conventions

- Windows 11 first; PySide6 for UI and screen capture (not `mss`).
- Screenshot, audio and file contents stay in memory, and `Pipeline._reset()` drops the context after each request. Never persist or log them, or API keys.
- Screen text, file text and model output are untrusted. The model can only choose among the allowlisted platforms. It never chooses URLs, commands or arbitrary mentions.
