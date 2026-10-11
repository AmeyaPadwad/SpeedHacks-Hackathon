# FlowState AI - The Context-Aware Voice Agent

FlowState is a background desktop assistant designed for hackathons. It eliminates context-switching by combining voice commands, screen context and IDE integration to draft actions without leaving your workflow. It lives on your desktop as **Pette**, a small animated companion.

## The Concept
Developers and knowledge workers lose their "flow state" every time they have to switch windows to ask ChatGPT a question, run a script or send a message. FlowState lets you press a hotkey and speak. The AI sees what you see, reads the file you are working on, and drafts the message for your approval.

## How it works (the UX)
1. You are working in your IDE (e.g., VS Code) on a specific file.
2. You press the global hotkey (`Ctrl+Shift+Space`) or click Pette.
3. FlowState takes one screenshot of your screen and reads your active file (via the companion VS Code extension).
4. You speak your command: *"Send a Slack to Sarah asking her to review this code."*
5. Whisper transcribes it, and GPT-4o gets the text, screenshot and file contents.
6. Pette's speech bubble shows the drafted Slack message. You click **Send** (or **Cancel**).
7. You never left your IDE.

Questions that aren't messages ("what does this error mean?") get a short answer in the bubble instead.

## Coding input collector (macOS, Python 3.11 or 3.12)

`--input-gathering` collects a coding task and its context for a teammate's Darwin integration:
**hotkey → foreground-window screenshot → voice → local Whisper → relevant saved files → review → context packet**.
This mode does not search Darwin, contact a provider, execute code, or dispatch messages.
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
GROQ_MODEL=openai/gpt-oss-20b
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
   Uncheck inputs you do not want included. **Use context** emits the reviewed packet once;
   if AI selection is enabled and the task/hints changed, it first runs selection again for review.
   **Export JSON** saves it only after approval. Escape or **Cancel / New** cancels collection and
   discards pending results. Closing the review card cancels collection and returns to Pette.
   Use the tray to review context, type a task, pause collection, or quit.

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

**Data lifecycle:** the screenshot stays local until packet handoff/export. The selected AI provider receives the task,
hints, project paths and bounded excerpts only after opting into file selection; it does not receive
the audio or screenshot. `whisper-cli` uses a private temporary directory for audio and transcript,
deleted after success, failure, or cancellation. Packets stay in memory until explicit export, which
contains selected source and raw-base64 PNG bytes. No input content or keys are logged. Exporting or
handing off a packet does not itself upload context to Darwin.

### Teammate handoff

The modules in `pet/inputs/` are independent of the messaging pipeline. `CollectionService` exposes
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
$env:OPENAI_API_KEY = "sk-..."
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."   # optional; see Configuration
py main.py              # speak commands
py main.py --no-voice   # type commands instead
py main.py --demo       # offline: typed commands, keyword drafter instead of GPT-4o (no OpenAI needed)
```

Settings load from a `.env` file in the repo root if present; real environment variables take precedence. In `--demo` mode every draft is labeled **DEMO (offline, no AI)**. Webhooks still fire for real when you click Send.

Pette appears in the bottom-right corner with a tray icon. Then:

- Press `Ctrl+Shift+Space`, or click Pette, and speak.
- Drag Pette to move it.
- Use the tray to **Pause all** (stops everything and drops pending drafts) or **Quit**.

**VS Code extension (optional, recommended):** copy `extensions/vscode` into `%USERPROFILE%\.vscode\extensions\flowstate-context`, then reload VS Code. It writes the active file's path, language, contents (first 12,000 characters) and selection to `~/.flowstate_current.json`. Run **FlowState: Toggle sharing active file** from the Command Palette to turn it off. Without the extension, FlowState uses the screenshot only.

## Configuration

All settings are environment variables. See [.env.example](.env.example).

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for the messaging mode; used for input file selection only when its provider is explicitly `openai`. |
| `OPENAI_MODEL` | Messaging model and optional OpenAI file-selection model, default `gpt-4o`. |
| `FLOWSTATE_INPUT_PROVIDER` | File-selection provider: `groq` (default) or `openai`. |
| `GROQ_API_KEY` | Required for Groq file selection; create a key using a Groq Free-plan account. |
| `GROQ_MODEL` | Groq file-selection model, default `openai/gpt-oss-20b`. |
| `FLOWSTATE_INPUT_MODEL` | Optional input-model override; otherwise the selected provider's model setting is used. |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook. Messages post to the webhook's channel. |
| `DISCORD_WEBHOOK_URL` | Discord webhook. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | Email via SMTP with STARTTLS (port 587 by default). |
| `FLOWSTATE_HOTKEY` | Global hotkey, default `ctrl+shift+space`. |
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

**Flow:** hotkey or click → screenshot + active file → voice → Whisper → GPT-4o JSON draft → bubble → user approves → dispatch.

| Module | File | What it does |
|---|---|---|
| 1. Input engine | `pet/voice.py` | Global hotkey (`keyboard`). Records until silence (`speech_recognition` + PyAudio) and transcribes with Whisper. If the mic fails or nothing is heard, the bubble asks you to type the command. |
| 2. Context capture | `pet/context.py` | One-shot Qt screenshot of the primary screen, taken with Pette hidden. Saved as an in-memory JPEG (quality 75, longest edge 1,280px). Reads the VS Code snapshot. |
| 3. AI brain | `pet/brain.py` | GPT-4o with JSON response mode. The output must be exactly `{platform, recipient, subject, message}` with platform `slack`, `discord`, `email` or `none`; anything else is rejected. Screen and file text are treated as untrusted. |
| 4. Floating UI | `pet/ui.py`, `pet/sprites.py` | Frameless, always-on-top, draggable Pette with idle, listening, thinking, speaking and sleeping animations; a speech bubble with Send/Cancel or a text field; and the tray menu. |
| 5. Dispatch | `pet/dispatch.py` | Slack and Discord webhooks, plus SMTP email. Discord pings only the resolved contact (`allowed_mentions`). |
| Main loop | `pet/pipeline.py`, `main.py` | Runs the flow on the Qt main thread, with slow steps on worker threads. |
| IDE bridge | `extensions/vscode/` | Writes the active-file snapshot on editor or selection change and on save (debounced, atomic write). |

### Safety rules
- **Nothing is sent without a click.** Each draft has an ID, and Send works once. Cancel discards it.
- **Pause** ignores in-flight results, clears pending drafts and blocks new requests until you resume.
- In messaging mode, screenshots, audio and file contents live only in memory and are dropped after each request. The coding collector uses temporary Whisper files and explicit packet export as described above. Nothing is logged; the VS Code extension writes its snapshot separately.
- **Rate caps:** at most one AI request every 3 seconds and 50 per session. There are no automatic retries.
- Data sent to Groq, OpenAI, Slack, Discord or your SMTP server is subject to those providers' terms.

## Development

```powershell
py -m pytest -q                         # all tests (Qt runs offscreen)
py -m pytest tests/test_brain.py -q     # one file
py -m pip install Pillow numpy
py tools/prep_sprites.py                # rebuild assets/pette_sheet.png from images/pette-spritesheet.png
```

The tests use fake clients, so they need no API keys, mic or network.
For input mode: `.venv/bin/python -m pip install pytest`, then `.venv/bin/python -m pytest -q`.

## Status
- **Checked:** 34 unit tests covering JSON validation, dispatch payloads, the approval and cancel flow, voice-to-typing fallback, pause dropping late results, and the busy guard. The app launches on Windows 11 with Python 3.13.
- **Not yet checked:** a live end-to-end run with a real API key, mic and webhooks; multi-monitor setups; PyInstaller packaging.
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
