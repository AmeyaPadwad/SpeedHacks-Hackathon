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
| `OPENAI_API_KEY` | Required. Used for Whisper transcription and GPT-4o drafting. |
| `OPENAI_MODEL` | Drafting model, default `gpt-4o`. |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook. Messages post to the webhook's channel. |
| `DISCORD_WEBHOOK_URL` | Discord webhook. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | Email via SMTP with STARTTLS (port 587 by default). |
| `FLOWSTATE_HOTKEY` | Global hotkey, default `ctrl+shift+space`. |
| `FLOWSTATE_CONTACTS` | Contacts file, default `~/.flowstate_contacts.json`. |
| `FLOWSTATE_ACTIVE_FILE` | Active-file snapshot path, default `~/.flowstate_current.json`. Set the same value for VS Code and the app. |

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
- Screenshots, audio and file contents live only in memory and are dropped after each request. Nothing is logged or written to disk, apart from the snapshot file written by the VS Code extension.
- **Rate caps:** at most one AI request every 3 seconds and 50 per session. There are no automatic retries.
- Data sent to OpenAI, Slack, Discord or your SMTP server is subject to those providers' terms.

## Development

```powershell
py -m pytest -q                         # all tests (Qt runs offscreen)
py -m pytest tests/test_brain.py -q     # one file
py -m pip install Pillow numpy
py tools/prep_sprites.py                # rebuild assets/pette_sheet.png from images/pette-spritesheet.png
```

The tests use fake clients, so they need no API keys, mic or network.

## Status
- **Checked:** 34 unit tests covering JSON validation, dispatch payloads, the approval and cancel flow, voice-to-typing fallback, pause dropping late results, and the busy guard. The app launches on Windows 11 with Python 3.13.
- **Not yet checked:** a live end-to-end run with a real API key, mic and webhooks; multi-monitor setups; PyInstaller packaging.
- **Known limits:** captures the primary monitor only. Slack webhooks post to one fixed channel (the recipient is @-mentioned, not DMed). The `keyboard` hotkey may need the app to run at the same privilege level as the focused window.
