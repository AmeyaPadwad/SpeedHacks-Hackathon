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

## Coding input collector (macOS, Python 3.11 or 3.12)

`--input-gathering` collects a coding task and feeds its reviewed context into the existing agent flow:
**hotkey → foreground-window screenshot → voice → local Whisper → relevant saved files → review → agent → message draft → Send**.
For example, say **"Send Sarah a Slack summary of how pet/inputs/macos.py works"**, review the
inputs, and click **Use context**. Pette drafts the message; click **Send** to post it through your
configured Slack/Discord webhook or SMTP account. A debugging request follows the existing
analysis and coding-agent proposal flow instead. Darwin search remains a separate teammate integration.
Pette is the main UI: the animated desktop pet listens while you speak, thinks during transcription
and file selection, and opens a cream-and-mint review card from its speech bubble. The card has
separate file/screenshot tabs and keeps approval/export buttons visible on smaller screens.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-input.txt
bash tools/setup_whisper.sh
.venv/bin/python main.py --input-gathering --project /path/to/project
```

The setup script builds [whisper.cpp v1.9.5](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5)
and downloads the English `small.en` model (about 466 MiB). It needs Git, CMake, and Xcode command-line
tools. Runtime/model files are ignored by Git; the collector discovers these local paths automatically.
Alternatively, use an installed `whisper-cli` and configure `FLOWSTATE_WHISPER_BIN` and
`FLOWSTATE_WHISPER_MODEL`. Python 3.13+ is not supported by this mode's audio resampler.

**Free hosted file selection:** create a key at [Groq API Keys](https://console.groq.com/keys) using
a Groq account on the Free plan, then add these entries to the project's `.env` and restart:

```dotenv
FLOWSTATE_INPUT_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
FLOWSTATE_INPUT_MODEL=openai/gpt-oss-20b
```

Groq's [Free plan limits](https://console.groq.com/docs/rate-limits) include GPT-OSS 20B. Limits depend
on the account and can change; exceeding them returns a recoverable rate-limit error. Free-plan usage
does not require buying OpenAI credits. The collector uses the installed OpenAI SDK with
[Groq's compatible endpoint](https://console.groq.com/docs/openai); no additional dependency is needed.
It does not automatically switch providers or upgrade account plans. Optional paid OpenAI selection
is available by explicitly setting `FLOWSTATE_INPUT_PROVIDER=openai` and `OPENAI_API_KEY`.

1. Connect your project with `--project` or Pette's **Choose project…** tray action. The default is the directory from which
   you launched the app, not another terminal's current directory.
2. Switch to your IDE, terminal, or browser. Press `Ctrl+Shift+Space` to capture that foreground window
   and open the microphone. You can also click Pette to start/stop. When audio starts arriving,
   its bubble shows a red **RECORDING** label and elapsed time. Press again or click
   **Stop & transcribe**; recording stops at 60 seconds. The bubble labels microphone startup,
   stopping, and local transcription. Use **Type a task** in the tray menu for text input.
3. Click **Review context** in Pette's bubble and edit the transcript. Name important files/folders in
   your task, enter optional hints, or click **Tell Pette more** to add voice hints without replacing the task.
4. Check **Let Groq find the relevant files** to start selection for your current task, or choose up
   to eight files manually. **Find relevant files** retries selection after editing your task/hints.
   Use `GROQ_API_KEY` for the default provider. Local speech transcription has no API fee;
   hosted file selection is subject to the chosen provider's account plan and quota.
5. Review each selected excerpt and the screenshot (use **View screenshot at full size**).
   Uncheck inputs you do not want included. Under **Send the result to**, choose **Slack** or **Discord**
   to draft a message, even when your task only says "explain this file". The optional person field
   adds a recipient/mention; leave it blank to post directly to the webhook's configured channel.
   **Let the agent decide** keeps the usual question/coding/message routing.
   **Use context** passes the reviewed packet to the agent once,
   hides the review card, and shows the draft/result in Pette. **Send** posts the reviewed message;
   if AI selection is enabled and the task/hints changed, it first runs selection again for review.
   **Export JSON** remains available from the review card after approval. Escape or **Cancel / New**
   cancels collection, discards any pending message and ignores late agent results. Closing the
   review card cancels collection and returns to Pette. After a result is sent or dismissed, the
   hotkey or **Type a task** starts a new request. During processing or a pending draft, the hotkey
   leaves the current flow in place; use **Cancel collection** in the tray to discard it.
   Use the tray to review context, type a task, pause collection, or quit.

The message preview also has a **Send via** Slack/Discord selector. Switching it changes the
destination of the same draft without another AI request. An unconfigured channel disables Send
and shows a setup hint. A failed post keeps the draft and displays its error alongside **Retry**;
you can switch channels before trying again. Each retry requires a fresh click. Webhooks post to
their configured channel, rather than opening a direct message to the named person.

**macOS permissions:** enable Microphone, Screen Recording, and Input Monitoring for the terminal/app
launching FlowState in System Settings → Privacy & Security; Accessibility may also be required by
the keyboard listener. Restart after changing permissions. Permission failures show recoverable
messages. **Type a task** and manual file selection work without a microphone, model, or API key.
Clicking Pette uses the most recently focused external window if Pette has taken focus. The remembered
window must still be visible; switch to your IDE/browser and use the hotkey to choose a new target.
Capture uses one window only and has a native command fallback if Quartz capture is unavailable.
Pette shows whether a window was captured; the **Window snapshot** tab displays the precise error
and an **Open Screen Recording settings** button when access is denied.
Microphone capture runs in a separate process so a stuck native audio driver can be terminated.
Startup without audio times out after 15 seconds; stopping times out after 5 seconds. Either failure
returns you to task editing with an error, where you can type your task. **Cancel / New** also stops
the microphone process and discards its audio.
To stop the app, use the tray's **Quit** action or press **Ctrl+C** in the launching terminal.
SIGINT/SIGTERM are handled through the Qt event loop and run the collection cleanup.

**File selection:** the LLM starts with a bounded file inventory, then can request project-relative
listings, literal searches, and line-range reads. It gets at most three model rounds, twelve reads
(including search reads), and 48,000 source characters. Only excerpts actually read can be selected.
Groq uses one declared `select_context` function with JSON arguments and matched tool-result messages.
The function only produces a file-selection plan; the client validates paths and reads bounded saved
files. OpenAI selection uses JSON response mode. Groq parse/function failures are reported with their
actual error codes, without logging model-generated source or credentials.
At most eight excerpts are retained, each bounded to fit a 4,000-character Darwin text item including
its path and reason. Filename matching tolerates spoken punctuation such as `mac-os.py` for `macos.py`.
Selected line ranges can be subsets of the source actually read. Ambiguous file hints can produce a clarification question; edit the hints and
retry. Manual selection reads the first 200 lines with the same exclusions and output bounds.
Exports include `file_selection_status`, `file_selection_provider`, and `file_selection_model`. Status distinguishes selection
that was not requested, succeeded, returned no files, needs clarification, failed, was manual, or
had every file unchecked. Empty packets include an explanation in `warnings`; clarification questions
are retained there too. Older exports load with status `unknown`.

The collector respects root/nested `.gitignore` rules and excludes dependency/build directories,
binary/large files, `.env*`, private keys, credential files, and symlinks. The inventory caps at 2,000
eligible paths/20,000 visited entries. These rules are not automatic secret redaction: source files
and screenshots may still contain sensitive material, so review them. Saved files can change during
collection; excerpts show the bytes read, with line ranges and truncation notices. Unsaved editor
buffers, terminal output/history, live terminal cwd, browser DOM, and automatic OCR are not collected.

**Data lifecycle:** the screenshot stays local until **Use context** or explicit export. The selected AI provider receives the task,
hints, project paths and bounded excerpts only after opting into file selection; it does not receive
the audio or screenshot during file selection. **Use context** sends the final task, checked file excerpts
and checked screenshot to the Groq drafting agent. Screenshots are converted to a bounded JPEG;
the agent receives a fair excerpt budget across every checked file, with truncation labelled.
If `GROQ_MODEL` selects text-only GPT-OSS, screenshot requests use `GROQ_VISION_MODEL` instead
(default `qwen/qwen3.8-27b`). The coding analyst and CLI adapters receive the checked file excerpts,
with their original path/line-range headers, and no screenshot. `whisper-cli` uses a private temporary directory for audio and transcript,
deleted after success, failure, or cancellation. Packets stay in memory until explicit export, which
contains selected source and raw-base64 PNG bytes. No input content or keys are logged. Exporting or
handing off a packet does not itself upload context to Darwin.

### Teammate handoff

`CollectionService` can still be used independently of the messaging pipeline. It exposes
`progress`, `changed`, `error`, and `packet_ready` Qt signals; hotkey events enter through
`hotkey_pressed`, and cancellation through `cancel_pressed`. All signals delivered to the UI use Qt's
main thread. `approve()` is single-use for the session, and cancellation invalidates late workers.

```python
from pathlib import Path
from pet.inputs.packet import ContextPacket, to_darwin_search_input
from pet.inputs.service import CollectionService

def receive_packet(packet: ContextPacket):
    search_input = to_darwin_search_input(packet)
    # Teammate calls Darwin search with this query/context in their own worker.

# Keep the service alive in your Qt app; InputPanel(service) supplies the review UI.
service = CollectionService(Path("/path/to/project"), on_packet_ready=receive_packet)

# Or consume an explicitly exported packet in another process:
packet = ContextPacket.from_json(Path("flowstate-context.json").read_text())
search_input = to_darwin_search_input(packet)
```

Version 1 contains `schema_version`, `session_id`, `captured_at`, `transcript`, editable final `task`,
local `project_root`, `window` metadata, `files` (relative path, lines, content, reason, truncation),
`warnings`, and optional screenshot. `to_darwin_search_input()` is a pure adapter producing `query`
and up to ten `context` items: eight text excerpts, one metadata text item, and one PNG file item.
The file uses raw base64, not a data URL; the image is capped at 4.9 MB. The adapter omits the
`project_root` field, although visible paths can still appear in window titles/source/screenshots.
Darwin search/refinement IDs, anonymous tokens, matches, and execution belong to the consuming
integration. See [Darwin's documentation](https://darwin.so/docs) for its current API contract.

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
| `GROQ_API_KEY` | Used for messaging transcription, drafting, analysis and opt-in input file selection on [Groq](https://console.groq.com). Input mode transcribes locally. |
| `GROQ_MODEL` | Drafting model (must accept images), default `qwen/qwen3.8-27b`. |
| `GROQ_VISION_MODEL` | Screenshot drafting model when `GROQ_MODEL` selects text-only GPT-OSS; default `qwen/qwen3.8-27b`. |
| `GROQ_TEXT_MODEL` | Analysis model, default `openai/gpt-oss-120b`. |
| `GROQ_STT_MODEL` | Transcription model, default `whisper-large-v3-turbo`. |
| `FLOWSTATE_INPUT_PROVIDER` | File-selection provider: `groq` (default) or `openai`. |
| `FLOWSTATE_INPUT_MODEL` | Input file-selection model override. Defaults to `openai/gpt-oss-20b` for Groq, independently of `GROQ_MODEL`; uses `OPENAI_MODEL` for OpenAI. |
| `OPENAI_API_KEY` | Optional; used only when input file selection explicitly selects `openai`. |
| `OPENAI_MODEL` | Optional OpenAI file-selection model, default `gpt-4o`. |
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
| `FLOWSTATE_WHISPER_BIN` | Input mode's `whisper-cli` executable; auto-detects the setup script's runtime. |
| `FLOWSTATE_WHISPER_MODEL` | Input mode's local model; defaults to `models/ggml-small.en.bin` in this repository. |

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
- In messaging mode, screenshots, audio and file contents live only in memory and are dropped after each request. The coding collector uses temporary Whisper files and explicit packet export as described above. Nothing is logged; the VS Code extension writes its snapshot separately.
- **Rate caps:** at most one AI request every 3 seconds and 50 per session. There are no automatic retries.
- **Coding agents only propose.** Each one runs read-only or planning-only. Agent output is shown for review and never applied.
- **Webcam data stays local.** It is off by default and released on Pause. Resuming doesn't turn it back on. Webcam data is never sent anywhere, including to coding agents.
- Data sent to Groq, OpenAI, Slack, Discord or your SMTP server is subject to those providers' terms.

## Development

```powershell
py -m pytest -q                         # all tests (Qt runs offscreen)
py -m pytest tests/test_brain.py -q     # one file
py -m pip install Pillow numpy
py tools/prep_sprites.py                # rebuild assets/pette_sheet.png from images/pette-spritesheet.png
```

The tests use fake clients, so they need no API keys, mic or network. `tests/test_e2e.py` runs a full session (messages, focus gate, webcam events, proactive investigation, agent, pause, memory) and launches the real `main.py`.
For input mode: `.venv/bin/python -m pip install pytest`, then `.venv/bin/python -m pytest -q`.

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

**Coding input mode validation (macOS 13.5.2, Python 3.11.10):** 124 tests passed across both modes;
the new tests cover bounded project access, model selection/clarification, native capture targeting,
record/stop/cancel, actual worker-thread UI delivery, the recording indicator, termination of simulated
stuck microphone processes, spoken filenames, grounded subranges, automatic selection on opt-in,
selection status exports, Pette controls/animations, pause/cancellation, typed tasks, review approval,
packet serialization, Darwin input limits, provider/key routing, model settings, and provider-specific
authentication/quota errors, remembered-window capture, capture fallback, and SIGINT/SIGTERM shutdown.
The real Qt panel/tray
launched and quit successfully. A native whisper.cpp build with `small.en` transcribed the bundled
11-second public sample in 14.6 seconds including model startup; latency varies with hardware/audio.
Both denied-access handling and native capture of a permission-enabled Terminal window were verified.
The full input-gathering entrypoint exited cleanly after a real SIGINT. Live microphone recording
and global hotkeys still need manual verification. A live OpenAI request with the configured key
reached `gpt-4o` but returned `429 credit_balance_exhausted`.
Groq is now the input collector's default. Live `openai/gpt-oss-20b` selection with the declared function
successfully returned grounded excerpts from `pet/inputs/macos.py` for the spoken `mac-os.py` request.
Automated selection tests use controlled responses. Darwin search is intentionally owned by the consuming teammate.
