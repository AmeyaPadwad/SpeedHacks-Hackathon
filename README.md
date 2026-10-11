# FlowState AI - The Context-Aware Voice Agent

FlowState is a background desktop assistant designed for hackathons. It eliminates context-switching by combining voice commands, screen context and IDE integration to draft actions without leaving your workflow. It lives on your desktop as **Pette**, a small animated companion.

## The Concept
Developers and knowledge workers lose their "flow state" every time they have to switch windows to ask ChatGPT a question, run a script or send a message. FlowState lets you press a hotkey and speak. The AI sees what you see, reads the file you are working on, and drafts the message for your approval.

## How it works (the UX)
1. You are working in your IDE (e.g., VS Code) on a specific file.
2. You press the global hotkey (`Ctrl+Shift+Space`) or click Pette.
3. FlowState takes one screenshot of your screen and reads your active file (via the companion VS Code extension).
4. You speak your command: *"Send a Slack to Sarah asking her to review this code."*
5. Whisper (on Groq) transcribes it, and a Groq-hosted model (Qwen 3.8 27B) gets the text, screenshot and file contents.
6. Pette's speech bubble shows the drafted Slack message. You click **Send** (or **Cancel**).
7. You never left your IDE.

Questions that aren't messages get a short answer in the bubble instead. Requests about code on screen ("why is this test failing?", "ask Claude to fix this") go to a coding agent (see below).

## Coding agents
Pette also *watches for blockers* and hands them to your coding agent (Claude Code, Hermes, or any agent via clipboard). Investigating never takes a screenshot: the agent works from the file you have open.

1. **Active file.** The VS Code extension reports the active file's path, contents, selection, workspace root, and its errors and warnings, so the agent gets the real code with exact messages and line numbers.
2. **Analysis.** Groq (gpt-oss-120b) turns the file, diagnostics and selection into an observation, evidence, likely cause, next step and confidence. Inferences are labelled as hypotheses. Without Groq, a **diagnostics only** analysis is used and clearly labelled.
3. **Approval → coding agent.** The bubble shows the analysis with **Ask Claude Code** (or Hermes, or **Copy task**). On approval:
   - **Claude Code** runs `claude -p --permission-mode plan --tools Read,Grep,Glob` in the workspace. It gets the whole active file on stdin, can read the rest of the workspace, and proposes a root cause, a diff and a test. It cannot edit files or run commands.
   - **Hermes** runs `hermes -z` with only the planning toolset, because `-z` auto-approves tools. The file is passed inline.
   - **Clipboard** copies the task for any agent.

   The answer opens in a read-only panel with a Copy button. **Nothing is applied to your files.**
4. **Proactive help.** If an IDE error persists for 20 seconds, Pette asks *"Looks like you're stuck. Investigate?"* It asks once per issue, then waits 10 minutes before asking about it again, and at most 8 times a day. It never asks while you're away, have your eyes closed, or are mid-interaction. Nothing is uploaded or sent to an agent until you click Investigate.
5. **Webcam (local MediaPipe, off by default).** Turn it on from the tray or with `--camera`.
   - **Face away for 3 seconds:** Pette dozes, and proactive offers wait until you're back.
   - **Both eyes closed:** Pette closes her eyes too.
   - **Pinch (thumb and index finger):** opens Pette or dismisses a message. A gesture never approves or cancels anything.
   - **Privacy:** a mint dot shows whenever the camera is on. Frames and landmarks never leave the camera thread and are never stored or uploaded. These signals are heuristics, not gaze or emotion detection.

### Webcam signals → faster messaging
| Signal | In the message workflow |
|---|---|
| **Pinch** (idle) | Push-to-talk: Pette opens and starts listening, with no hotkey needed |
| **Pinch** on a draft | Throw it away and dictate again (*"no wait, say it differently"*). A gesture never sends anything; only a click does |
| **Pinch** on a plain message | Dismiss it |
| **Away → back** | Drafts wait while you're away. When you return the bubble asks *"Still send?"* and Pette wakes up |
| **Eyes closed** | Pette mirrors you. Proactive offers wait until your eyes are open |

### Memory: `db/` cache and world model
Pette keeps a small local SQLite cache in `db/pette_cache.sqlite3` (gitignored, 30-day retention) and learns from it:
- **Usual platform per contact:** "message Sarah …" goes to the platform you usually send her on. The title says "(usual)", and a platform you name always wins. Frequent contacts are also listed in the GPT prompt.
- **Eyes-closed calibration:** the threshold follows your open-eye baseline (0.5–0.85).
- **Offer back-off:** if you decline most proactive offers, you get fewer of them.
- **Presence stats:** away episodes, eye closures and pinches.

It stores facts only (event names, timestamps, recipient and platform, approve/cancel outcomes), never message text, screenshots, frames or code. Use **What Pette learned**, **Remember my patterns** and **Clear Pette memory** in the tray. Details are in [db/README.md](db/README.md).

Try it on the included fixture: open `fixtures/buggy_app/dashboard.py` in VS Code (with the extension), run its tests so the error shows, then press `Ctrl+Alt+Space`.

## Quick start (Windows 11, Python 3.10+)

```powershell
py -m pip install -r requirements.txt
$env:GROQ_API_KEY = "gsk_..."
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."   # optional; see Configuration
py main.py              # speak commands
py main.py --no-voice   # type commands instead
py main.py --demo       # offline: typed commands, keyword drafter instead of the LLM (no API key needed)
py main.py --camera     # also turn on the webcam signals at launch
py tools/get_models.py  # once: download the MediaPipe face/hand models into models/
```

Settings load from a `.env` file in the repo root if present; real environment variables take precedence. In `--demo` mode every draft is labeled **DEMO (offline, no AI)**. Webhooks still fire for real when you click Send.

Pette appears in the bottom-right corner with a tray icon, **only while VS Code or a terminal is the app in front**. That includes Windows Terminal, PowerShell, cmd, Git Bash (mintty), WezTerm, Alacritty, Tabby, Hyper and Warp. Switch to a browser or chat app and she hides. Hotkeys and proactive offers also pause until you're back, and a pending proposal waits for you. Clicking Pette or her tray menu doesn't count as switching apps.

- To show her everywhere, turn off **Show only in VS Code / Terminal** in the tray, or run with `--always-show`.
- To use a different app list, set `FLOWSTATE_SHOW_IN`, e.g. `FLOWSTATE_SHOW_IN=Code.exe,Cursor.exe,WindowsTerminal.exe`.

Then:

- Press `Ctrl+Shift+Space`, or click Pette, and speak.
- Press `Ctrl+Alt+Space` (or use **Investigate** in the tray) to send the open file to your coding agent.
- Drag Pette to move it.
- The tray also has toggles for the camera and proactive help, and a choice of coding agent.
- **Pause all** stops everything, releases the camera and drops pending drafts. **Quit** exits.

**VS Code extension (optional, recommended):** copy `extensions/vscode` into `%USERPROFILE%\.vscode\extensions\flowstate-context`, then reload VS Code. It writes the active file's path, language, contents (first 12,000 characters) and selection to `~/.flowstate_current.json`. Run **FlowState: Toggle sharing active file** from the Command Palette to turn it off. Without the extension, message drafts use the screenshot only and Investigate has no file to hand over.

## Configuration

All settings are environment variables. See [.env.example](.env.example).

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required. Used for Whisper transcription, drafting and analysis on [Groq](https://console.groq.com). |
| `GROQ_MODEL` | Drafting model (must accept images), default `qwen/qwen3.8-27b`. |
| `GROQ_TEXT_MODEL` | Analysis model, default `openai/gpt-oss-120b`. |
| `GROQ_STT_MODEL` | Transcription model, default `whisper-large-v3-turbo`. |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook. Messages post to the webhook's channel. |
| `DISCORD_WEBHOOK_URL` | Discord webhook. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | Email via SMTP with STARTTLS (port 587 by default). |
| `FLOWSTATE_HOTKEY` | Global hotkey, default `ctrl+shift+space`. |
| `FLOWSTATE_INVESTIGATE_HOTKEY` | Investigate-the-open-file hotkey, default `ctrl+alt+space`. |
| `FLOWSTATE_AGENT` | `claude`, `hermes` or `clipboard`. Default: the first installed. |
| `FLOWSTATE_SHOW_IN` | Comma-separated process names where Pette appears. Default: VS Code plus common terminals. |
| `FLOWSTATE_AGENT_TIMEOUT` | Seconds before an agent run is abandoned, default 240. |
| `FLOWSTATE_CONTACTS` | Contacts file, default `~/.flowstate_contacts.json`. |
| `FLOWSTATE_ACTIVE_FILE` | Active-file snapshot path, default `~/.flowstate_current.json`. Set the same value for VS Code and the app. |

**Contacts:** map spoken names to real IDs so messages actually ping people. Email needs an address, either from this file or spoken in full.

```json
{ "sarah": { "slack": "U012ABCDEF", "discord": "123456789012345678", "email": "sarah@example.com" } }
```

A platform with no webhook or SMTP host configured still drafts, but the bubble title says "(not configured)" and Send reports the error.

## Architecture

**Flow:** hotkey or click → screenshot + active file → voice → Whisper → Groq JSON draft → bubble → user approves → dispatch.

| Module | File | What it does |
|---|---|---|
| 1. Input engine | `pet/voice.py` | Global hotkey (`keyboard`). Records until silence (`speech_recognition` + PyAudio) and transcribes with Whisper on Groq. If the mic fails or nothing is heard, the bubble asks you to type the command. |
| 2. Context capture | `pet/context.py` | One-shot Qt screenshot of the primary screen, taken with Pette hidden. Saved as an in-memory JPEG (quality 75, longest edge 1,280px). Reads the VS Code snapshot. |
| 3. AI brain | `pet/brain.py` | Groq (via `pet/llm.py`) with JSON response mode. The output must be exactly `{platform, recipient, subject, message}` with platform `slack`, `discord`, `email` or `none`; anything else is rejected. Screen and file text are treated as untrusted. |
| 4. Floating UI | `pet/ui.py`, `pet/sprites.py` | Frameless, always-on-top, draggable Pette with idle, listening, thinking, speaking and sleeping animations; a speech bubble with Send/Cancel or a text field; and the tray menu. |
| 5. Dispatch | `pet/dispatch.py` | Slack and Discord webhooks, plus SMTP email. Discord pings only the resolved contact (`allowed_mentions`). |
| Main loop | `pet/pipeline.py`, `main.py` | Runs the flow on the Qt main thread, with slow steps on worker threads. |
| IDE bridge | `extensions/vscode/` | Writes the active-file snapshot, diagnostics and workspace root on editor, selection, diagnostics or save changes (debounced, atomic write). |
| Webcam CV | `pet/vision.py` | MediaPipe Face and Hand Landmarker on a camera thread. A deterministic debounce policy emits face away/back, eyes closed/open and pinch. |
| Analysis | `pet/analyst.py` | Groq structured analysis, or a labelled local fallback. |
| Agents | `pet/agents.py` | Propose-only adapters for Claude Code, Hermes and the clipboard. |
| Focus gate | `pet/focus.py` | Polls the foreground window's process every 300 ms (Win32 via ctypes) and shows or hides Pette. Pette's own windows and the taskbar don't count as a switch. |
| Watcher | `pet/watcher.py` | Proactive policy: persistence, dedupe, cooldown and daily cap, plus presence gating and camera ownership. |

### Safety rules
- **Nothing is sent without a click.** Each draft has an ID, and Send works once. Cancel discards it.
- **Pause** ignores in-flight results, clears pending drafts and blocks new requests until you resume.
- Screenshots, audio and file contents live only in memory and are dropped after each request. Nothing is logged or written to disk, apart from the snapshot file written by the VS Code extension.
- **Rate caps:** at most one AI request every 3 seconds and 50 per session. There are no automatic retries.
- **Coding agents only propose.** Each one runs read-only or planning-only. Agent output is shown for review and never applied.
- **Webcam data stays local.** It is off by default and released on Pause. Resuming doesn't turn it back on. Webcam data is never sent anywhere, including to coding agents.
- Data sent to Groq, Slack, Discord or your SMTP server is subject to those providers' terms.

## Development

```powershell
py -m pytest -q                         # all tests (Qt runs offscreen)
py -m pytest tests/test_brain.py -q     # one file
py -m pip install Pillow numpy
py tools/prep_sprites.py                # rebuild assets/pette_sheet.png from images/pette-spritesheet.png
```

The tests use fake clients, so they need no API keys, mic or network. `tests/test_e2e.py` runs a full session (messages, focus gate, webcam events, proactive investigation, agent, pause, memory) and launches the real `main.py`.

```powershell
py tools/run_checks.py               # tests + live system checks + UI screenshots -> test_results/REPORT.md
py tools/run_checks.py --webhooks    # also post a labelled test message to Slack/Discord
py tools/cv_check.py                 # live webcam signal checker
```

Latest results: [test_results/REPORT.md](test_results/REPORT.md).

### Manual testing sandbox
`tools/sandbox.py` runs the real app (same hotkeys, tray, focus gate, pipeline and agents) next to a control panel, so you can check every feature by hand without touching real channels or your real memory:

```powershell
py tools/sandbox.py                  # real Groq + voice; sandbox webhooks, IDE snapshot and db
py tools/sandbox.py --demo           # offline: typed commands, keyword drafter
py tools/sandbox.py --live-webhooks  # really post to the Slack/Discord webhooks in .env
py tools/sandbox.py --real-ide       # use the VS Code extension's snapshot instead of the fake one
```

- **Webhook inbox:** Slack and Discord posts land in the panel instead of real channels. Tick *Make webhooks fail* to test the HTTP 500 path. Email is left unconfigured unless `--live-webhooks`.
- **Fake VS Code:** one click sets the active file to `fixtures/buggy_app/dashboard.py` with or without its error, or to no file.
- **Simulated webcam:** face away/back, eyes closed/open and pinch, with no camera needed. `--camera` still uses the real one.
- **Triggers:** ask, investigate, pause, *Fast proactive* (3 s instead of 20 s) and *Reset offers*.
- **Checklist:** 24 scenarios with how-to and expected results. Mark each Pass/Fail/Skip, add notes, and save to `test_results/MANUAL_TEST.md`.

Contacts (`sarah`, `alex`) and the memory db live in a temp folder that is deleted when you close the panel. Other flags (`--no-voice`, `--camera`, `--always-show`) are passed to `main.py`.

## Status
- **Checked:**
  - 113 tests, plus live system checks: see [test_results/REPORT.md](test_results/REPORT.md).
  - Live Slack and Discord webhook posts.
  - Live Groq drafting (Qwen 3.8 27B with screenshot + file) and analysis (gpt-oss-120b).
  - A live Claude Code read-only investigation of `fixtures/buggy_app`: correct root cause and a diff, with no files modified.
  - The camera opens and MediaPipe runs on Windows 11 with Python 3.13.
- **Not yet checked:**
  - Groq Whisper on a real spoken command (drafting and analysis were checked live on Groq).
  - Real-person accuracy of face, eye and pinch detection.
  - A live Hermes run.
  - Multi-monitor setups and PyInstaller packaging.
- **Known limits:** captures the primary monitor only. Slack webhooks post to one fixed channel (the recipient is @-mentioned, not DMed). The `keyboard` hotkey may need the app to run at the same privilege level as the focused window.
