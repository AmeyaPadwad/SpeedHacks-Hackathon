# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Hackathon project with two overlapping concepts; only the overlay UI exists in code:

- `README.md`: **FlowState**, a hotkey + voice + screenshot + active-file agent that drafts a Slack/Discord/email message for approval. Its planned modules are input, context capture, AI router, floating UI and dispatch connectors.
- `brainstorming/BUILD.md`: **Pette**, a Windows desktop pet with webcam face/eye/pinch signals and screenshot-to-AI advice. It has its own file layout, security rules and acceptance checklist.
- `brainstorming/VISUAL_CODING_COMPANION.md`: an Electron-based alternative spec. It is not being built.

The implemented overlay UI uses the Pette sprite for the FlowState confirmation window. When the specs disagree, ask which one applies.

There are no tests, linter or build config yet. `main.py` shows a canned proposal labeled DEMO, and Send does nothing. Don't present it as real AI or dispatch output.

## Git workflow

Commit and push often, after every meaningful step, without asking first. Small or work-in-progress commits are fine. Work on the current branch (`sujash-working`), never directly on `main`.

## Commands

```powershell
py -m pip install -r requirements.txt   # PySide6 only; tools/prep_sprites.py also needs Pillow + numpy
py main.py                              # run the overlay (tray icon + floating pet)
python tools/prep_sprites.py            # regenerate assets/pette_sheet.png from images/pette-spritesheet.png
```

Offscreen smoke test, used instead of a test suite so far: set `QT_QPA_PLATFORM=offscreen`, build `Overlay(tray=False)`, then drive it with `propose()` / `bubble.primary.click()` / `set_paused()`. Offscreen rendering has no fonts, so text shows as boxes.

## Architecture

- `pet/sprites.py`: loads `assets/pette_sheet.png`, a 3x2 grid of equal cells, row-major. `ANIMATIONS` maps a state name to frame indices. The frame order is a contract with `tools/prep_sprites.py`.
- `pet/ui.py`:
  - `Overlay` is the top-level frameless always-on-top window and the integration surface: `propose(Proposal)`, `say(text)`, `set_state()`, `set_paused()`, plus the signals `approved(id)`, `cancelled(id)`, `ask_requested` and `pause_toggled`.
  - `PetWidget` paints the sprite and separates click from drag.
  - `Bubble` holds the message and the Send/Cancel buttons.
- `main.py` only wires demo behavior to the `Overlay` signals.

Invariants to keep:

- **Approval is single-use and bound to the proposal id.** `Bubble._consume()` clears the proposal and disables the buttons. Anything that executes an action must be triggered by `approved(id)`, never by a gesture or a timer.
- **Pause clears the pending proposal and blocks new bubbles.** `propose()` and `say()` return early while paused, and callers should also drop late async results themselves, as `main.ask` does.
- `Overlay._reanchor` keeps the pet fixed on screen while the bubble grows upward. Change bubble content only through it.
- `images/pette-spritesheet.png` has an opaque checkerboard baked in. Use the processed `assets/pette_sheet.png`, never the raw image, in the app.
- The tray icon is built from `sprites.frames[0]`. `sys._MEIPASS` is already handled in `asset_path` for PyInstaller.

## Conventions from the specs

- Windows 11 first. Python 3.10+, PySide6 UI.
- LLM calls: one request at a time, rate-capped, strict JSON validated before display, and the action allowlist is `none`, `copy_text`, `open_url` (HTTPS only). Screenshots and model replies are untrusted data.
- Never persist keys, screenshots or camera data, and never log them.
- Don't label simulated detections or demo output as real.
