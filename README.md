# Pette - the on-screen face of SOMA

**Pette** is a small animated companion on your desktop. **SOMA** is the brain behind her: it looks at your screen (with your consent), searches the folders you let it index, and ranks what you might want to do next with its own PyTorch model. Pette shows each suggestion as a glass card. Nothing happens until you press **Open** / **Copy** (or pinch).

Pette is only the UI. She sends no messages, calls no cloud AI and records no audio. SOMA runs locally in its own process ([SOMA/README.md](SOMA/README.md)).

## Quick start (Windows 11, Git Bash, [uv](https://docs.astral.sh/uv/))

```bash
./run_pette_soma.sh              # first run sets up .venv (Pette) and SOMA/.venv (SOMA), then starts both
./run_pette_soma.sh --focus-only # extra flags go to Pette: --focus-only, --no-camera, --no-world
./run_pette_soma.sh --diagnose   # one SOMA step on the current screen: why it would suggest something or stay silent
```

Running the script is your opt-in for SOMA to capture the screen (stored in `SOMA/data/consent.json`). Quit Pette from the tray; SOMA stops with her.

Then, from Pette's tray: **Add a folder for SOMA…** and pick your notes, a repo or a project folder. SOMA indexes it (text, code, images; size-capped) and starts pointing at those files in its suggestions.

## How it works

```
your screen ──► SOMA: capture (consent, exclusions) → OCR → screen + memory fusion → ranked next steps → safety gate
your folders ─┘                                                                                      │
                                                                localhost API (token) ◄───────────────┘
                                                                        │
Pette: glass card ("SOMA suggests · Open file")  ── Open / Copy / Not now / pinch ──► feedback back to SOMA
```

| SOMA sees | Card | Open / Copy does |
|---|---|---|
| A `file.py:40` reference or traceback, in a folder you indexed | *Open calc.py at line 40* | Opens it in VS Code at that line |
| Something related in your indexed files | *Compare with your earlier document "…"* | Opens that file |
| An error | *Explain the error "…"* | Copies the text for your coding agent |
| A file in your editor | *Review your changes to "app.py" before you commit*, *Write a test for the code in "app.py"* | Copies the text |
| A passing test run | *The tests pass: commit your work* | Copies the text |
| A web page | *Save what you learn from "…" to your notes* | Copies the text |

- **Click Pette** (or `Ctrl+Shift+Space`) to have SOMA look at the screen right now; otherwise it checks every couple of seconds and stays quiet most of the time (45 s between cards, no repeats for 5 minutes).
- A card stays 15 s (a bar drains under it). Ignoring it is not a *Not now*.
- **Learning:** every answer goes back to SOMA's online ranker. Pette also mutes a kind of suggestion you answer *Not now* to at least 80 % of the time (after 4 answers). The world-model panel shows why.
- **Opening never runs anything:** VS Code (`code -g file:line`) when its CLI is on PATH, images in their viewer, anything else is only selected in Explorer.

### The camera orb
A circle above Pette shows **you** (a mirrored webcam preview, processed and drawn locally only), your hand skeleton, and a status ring: blue *Looking*, amber *Suggestion*, green *Done ✓*, grey *Not now*, violet *Indexing*, red *SOMA offline*.

| Webcam signal | Effect |
|---|---|
| **Pinch** while a card is up | Same as **Open** / **Copy** |
| **Away → back** | Pette dozes while you're away; an open card is re-titled *"While you were away · …"* |
| **Eyes closed** | Pette closes her eyes too |

The camera is on by default. `--no-camera` or the tray turns it off, Pause releases it. Frames are never stored, logged or uploaded.

### The world-model panel
Docked beside Pette: live status, suggestions shown / done / not now per kind (and which are muted), the folders SOMA indexed, the explainable rules in effect, presence over the last 24 hours, and recent kinds + outcomes. Facts only: never the suggestion text, file names or screen content.

### Hotkeys
| Keys | Action | Setting |
|---|---|---|
| Ctrl+Shift+Space | Look at my screen now (same as clicking Pette) | `FLOWSTATE_HOTKEY` |
| Ctrl+Alt+S | SOMA suggestions on / off (off = SOMA stops capturing) | `PETTE_SOMA_HOTKEY` |
| Ctrl+Alt+C | Hide / show the camera orb (pinch still works) | `PETTE_CAMERA_HOTKEY` |
| Ctrl+Alt+W | Hide / show the world-model panel | `PETTE_WORLD_HOTKEY` |

## Configuration

| Variable | Purpose |
|---|---|
| `PETTE_SOMA_ENDPOINT` | SOMA endpoint file (`{url, token}`), written by `SOMA/scripts/serve.py`; the launcher sets it. |
| `SOMA_CONFIG`, `SOMA_CHECKPOINT`, `SOMA_INTERVAL`, `SOMA_SYNTHETIC` | Launcher: SOMA config (default `configs/pette.yaml`), model checkpoint, seconds between steps, synthetic screens. |
| `FLOWSTATE_SHOW_IN` | With `--focus-only`: comma-separated process names where Pette appears. |
| `FLOWSTATE_DB` | Pette's fact log, default `db/pette_cache.sqlite3`. |
| Hotkeys | See the table above. |

`SOMA/configs/pette.yaml` lets SOMA's top-ranked candidate through. SOMA's default config wants 55 % confidence, which its untrained model never reaches (about 30 % per option), so it stayed silent on real screens; the cooldown and duplicate window keep cards rare instead.

## Architecture

| Part | File | What it does |
|---|---|---|
| Hub | `pet/soma.py` | `SomaClient` (localhost-only, token) and `Suggester`: polls SOMA, shows cards when Pette is free, look-now, folder add/remove, feedback, muting, capture exclusion. |
| Do it | `pet/actions.py` | Open a file safely or copy text. Nothing else. |
| Overlay | `pet/ui.py`, `pet/sprites.py` | Pette, the camera orb, the frosted-glass bubble with the card's time-left bar. |
| Webcam | `pet/vision.py`, `pet/watcher.py` | MediaPipe face and hand tracking: presence, eyes closed, pinch, the orb preview. |
| World model | `db/`, `pet/worldview.py`, `web/` | SQLite fact log → explainable rules → the React/shadcn panel. |
| Focus gate | `pet/focus.py` | Optional (`--focus-only`): show Pette only over VS Code and terminals. |
| Hotkeys | `pet/hotkeys.py` | Global hotkeys that only emit Qt signals. |
| SOMA | `SOMA/` | The backend: capture, OCR, memory, PyTorch model, safety gate, local API (`scripts/serve.py`). |

### Safety rules
- **Nothing runs by itself.** SOMA only suggests; Pette acts only on Open / Copy / pinch, and opening never executes a file.
- **Screen capture is opt-in** (the launcher), pausable (tray, `Ctrl+Alt+S`), skips password managers, private windows and sign-in prompts, and keeps screenshots in memory only. Pette's own windows are excluded from capture.
- **Your files:** SOMA indexes only folders you pick, never a whole home directory or drive; removing a folder forgets everything from it. Suggestions only point at files inside those folders.
- **Pette's db stores facts only:** event names, timings, and which kind of suggestion you answered how ([db/README.md](db/README.md)).
- Nothing leaves your machine: SOMA's API binds to 127.0.0.1 with a per-start token.

## Development

```bash
.venv/Scripts/python.exe -m pytest -q                 # Pette (Qt offscreen; fake SOMA)
cd SOMA && .venv/Scripts/python.exe -m pytest -q      # SOMA
.venv/Scripts/python.exe tools/run_checks.py          # both suites + SOMA service + UI states -> test_results/
.venv/Scripts/python.exe tools/sandbox.py             # the real app + a panel driving a scripted fake SOMA
.venv/Scripts/python.exe tools/get_models.py          # MediaPipe models (the launcher does this)
cd web && npm ci && npm run build                     # rebuild pet/web/index.html after editing the panel
```

### Manual testing sandbox
`tools/sandbox.py` runs the real app next to a control panel around a **fake SOMA** (same API, scripted): queue an open-file, your-files or copy-only card, make *look now* find nothing, stop SOMA to see the offline state, simulate webcam events, and watch every request Pette makes. `--real-soma <endpoint file>` points it at the real service. A checklist saves to `test_results/MANUAL_TEST.md`.

## Status
- Pette ↔ SOMA verified live on this machine: SOMA serving real-screen suggestions, Pette showing them as cards.
- SOMA's model is trained on synthetic data only; suggestion quality on real screens needs consented real sessions (see [SOMA/README.md](SOMA/README.md)).
- Test results: [test_results/REPORT.md](test_results/REPORT.md).
