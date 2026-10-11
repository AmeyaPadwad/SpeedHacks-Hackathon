# FlowState AI - The Context-Aware Voice Agent

FlowState is a background desktop assistant designed for hackathons. You click Pette, say who to tell what, and she sends it: the message, a summary of what you changed, and the code itself. It lives on your desktop as **Pette**, a small animated companion.

## The Concept
Developers lose their "flow state" every time they switch windows to tell a teammate what they just changed. FlowState lets you click once and speak. It reads the file you're working on and its uncommitted git changes, writes the update, and posts it to Slack, Discord or email, with the code attached.

## The workflow
There is one workflow:

1. You're working in VS Code (or a terminal) on a file.
2. **Click Pette** (or press `Ctrl+Shift+Space`). The orb above her turns violet: *Listening*.
3. Say what you want: *"Slack Sarah what I changed in this file."*
4. **ElevenLabs** transcribes it, and your words appear in Pette's speech bubble.
5. The text goes to **Groq** with the active file and its `git diff`. It writes the message, including a short summary of what changed.
6. Pette shows the draft with the code attached (*"＋ dashboard.py · uncommitted changes"*), counts down a few seconds with **Cancel** and **Send now**, then **dispatches** it.
7. The orb shows **Sent ✓** or **Not sent**. You never left your editor.

Questions that aren't messages ("what does this file do?") get a short answer in the bubble, and nothing is sent.

### Darwin search
Say *"Search Darwin for an agent that books meetings"* (or *"find a tool that …"*). Pette calls [Darwin Search](https://darwin.so/docs/search/using-results) (`POST https://api.darwin.so/api/v3/search`, no key), and the top matches (name, readiness, why it matched, `searchId`) become a Slack draft. Pinch or the countdown posts it. Search only finds capabilities; Pette never connects to or runs them (`pet/darwin.py`).

### What gets sent
- **Your message:** written by Groq from your words, with a 1–3 sentence summary of the change.
- **The code:** attached by Pette, never written by the model. It's the uncommitted `git diff` of the active file, or your selection, or the file itself. On Slack and Discord it's a fenced ```diff``` / code block, trimmed to fit (Discord's 2,000-character limit is respected). On email it's plain text.
- **No screenshots**, ever.

### Voice: ElevenLabs transcription only
Pette never talks back. ElevenLabs is used for speech-to-text, and the transcript is shown in the bubble:
- **Scribe** (`ELEVENLABS_API_KEY` set): records until you stop talking, then one `scribe_v1` request. Recommended.
- **Live** (`ELEVENLABS_AGENT_ID`, public agent, no key): streams the mic to the agent's websocket and takes its transcript of your turn. The agent's replies and voice are dropped. Clear the agent's **First message** in the ElevenLabs dashboard. Otherwise its greeting talks over the start of your command, and the first words are lost.
- No ElevenLabs settings (or `FLOWSTATE_VOICE=whisper`): Groq Whisper.

Click Pette again while she's listening to stop. If the mic or transcription fails, a text box opens and typing works the same way.

### The camera orb
A circle above Pette shows **you** (mirrored webcam, 168px, processed and drawn locally only). It also shows your **hand** skeleton and a chip when you **pinch**, the one recognised gesture. A glowing ring shows the status: violet *Listening* → blue *Thinking* → amber *Sending in Ns* → green *Sent ✓* / red or grey *Not sent*.

| Webcam signal | Effect |
|---|---|
| **Pinch** (idle) | Nothing: click Pette to talk |
| **Pinch** while a draft is counting down | **Send it now** (same as Send now). Cancel is the button |
| **Away → back** | Pette dozes while you're away. On return, a pending send is re-titled *"Still sending?"* |
| **Eyes closed** | Pette closes her eyes too |

The camera is on by default (`--no-camera`, or the tray toggle, turns it off). Pause releases it. Frames are never stored, logged or uploaded. The db gets event names and timings only.

### The world-model window
A panel docked beside Pette shows what she has learned from her local db (`db/world_model.py`) and the **behaviour each fact drives**, live:
- **Usual platform per person:** *"message Sarah"* with no platform → Slack, *because 3 sent, 4 drafted to Sarah there*.
- **Adaptive countdown:** cancel a lot and Pette waits 6 s before sending. Almost never cancel and she waits 2.5 s. The default is 4 s.
- **Blink calibration**, sent/cancelled/failed counts, presence today, and recent *who and where* (never *what*).
- A live status card that mirrors the orb (camera, hand, presence).

It's built with **React + Tailwind + [shadcn/ui](https://ui.shadcn.com)** components, plus **[Aceternity UI](https://ui.aceternity.com)** effects (spotlight header, background-gradient live card, moving-border status pill). It's compiled to one self-contained file, `pet/web/index.html`, and shown in Qt WebEngine. The Qt overlay follows the same shadcn tokens (zinc palette, 12px radius, hairline borders), with an Aceternity-style rotating gradient border on the bubble while a send counts down.

## Quick start (Windows 11, Python 3.10+)

```powershell
py -m pip install -r requirements.txt
py tools/get_models.py   # once: MediaPipe face/hand models for the orb
# .env: GROQ_API_KEY, SLACK_WEBHOOK_URL / DISCORD_WEBHOOK_URL, ELEVENLABS_API_KEY or ELEVENLABS_AGENT_ID
py main.py               # click Pette and talk
py main.py --no-voice    # type instead of speaking
py main.py --demo        # offline keyword drafter (labeled DEMO); dispatch still runs for real
py main.py --no-camera --no-world
```

Settings load from `.env` in the repo root; real environment variables take precedence. Pette appears bottom-right **only while VS Code or a terminal is in front** (Windows Terminal, PowerShell, cmd, Git Bash, WezTerm, Alacritty, Tabby, Hyper, Warp). Use `--always-show` or the tray toggle to show her everywhere, or `FLOWSTATE_SHOW_IN` to choose the apps. Drag her to move. The tray has the world-model window, the camera, memory controls, **Pause all** and **Quit**.

**VS Code extension (needed for code + summaries):** copy `extensions/vscode` into `%USERPROFILE%\.vscode\extensions\flowstate-context` and reload VS Code. It writes the active file's path, language, contents (first 12,000 characters), selection and workspace to `~/.flowstate_current.json`. Pette runs `git diff HEAD -- <file>` in that file's folder herself.

## Configuration

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required for drafting (and for Whisper if ElevenLabs isn't set). |
| `GROQ_MODEL` | Drafting model, default `qwen/qwen3.8-27b`. |
| `GROQ_STT_MODEL` | Whisper fallback, default `whisper-large-v3-turbo`. |
| `ELEVENLABS_API_KEY` | Scribe speech-to-text (recommended). |
| `ELEVENLABS_AGENT_ID`, `ELEVENLABS_BRANCH_ID` | Live transcription through a public ElevenLabs agent (no key). Clear its First message. |
| `FLOWSTATE_VOICE` | `whisper` forces Groq Whisper. |
| `SLACK_WEBHOOK_URL`, `DISCORD_WEBHOOK_URL` | Incoming webhooks. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | Email via SMTP + STARTTLS. |
| `FLOWSTATE_HOTKEY` | Same as clicking Pette, default `ctrl+shift+space`. |
| `FLOWSTATE_CONTACTS` | Contacts file, default `~/.flowstate_contacts.json`. |
| `FLOWSTATE_ACTIVE_FILE` | Active-file snapshot path, default `~/.flowstate_current.json`. |
| `FLOWSTATE_SHOW_IN` | Comma-separated process names where Pette appears. |

**Contacts** map spoken names to IDs so messages ping people:
```json
{ "sarah": { "slack": "U012ABCDEF", "discord": "123456789012345678", "email": "sarah@example.com" } }
```
A platform without a webhook or SMTP host isn't sent. The bubble says it isn't configured and the orb shows *Not sent*.

## Architecture

**Flow:** click / hotkey / pinch → ElevenLabs transcript → bubble → file + `git diff` → Groq JSON draft → code attached → countdown (Cancel) → dispatch → orb + world model update.

| Module | File | What it does |
|---|---|---|
| Workflow | `pet/pipeline.py` | The one state machine (idle → listening → thinking → pending → sending → sent/failed/cancelled) on the Qt main thread. Slow steps run on workers and come back through `_deliver`. Emits `status` for the orb and the world window. |
| Voice | `pet/eleven.py`, `pet/voice.py` | ElevenLabs Scribe or live agent transcription (agent audio ignored). Groq Whisper fallback. Global hotkey. |
| Context | `pet/context.py` | VS Code snapshot plus `git diff HEAD` of the active file (untracked = new file). No screenshots. |
| AI brain | `pet/brain.py`, `pet/llm.py` | Groq JSON mode. The output must be exactly `{platform, recipient, subject, message}` with platform `slack`, `discord`, `email` or `none`. `attach_code` adds the diff/selection/file locally. |
| Dispatch | `pet/dispatch.py` | Slack/Discord webhooks and SMTP. `compose()` fits message + code into each platform's limit. Discord pings only the resolved contact. |
| Overlay | `pet/ui.py`, `pet/sprites.py` | Pette, the camera orb (frame, hand skeleton, gesture chip, status ring) and the shadcn-styled bubble with countdown. |
| Webcam | `pet/vision.py`, `pet/watcher.py` | MediaPipe face + hand on a camera thread: debounced presence/eyes/pinch events, plus the orb preview and gesture name. |
| World model | `db/`, `pet/worldview.py`, `web/` | SQLite fact log → explainable rules (`WorldModel.snapshot()`) → the React/shadcn window. |
| Focus gate | `pet/focus.py` | Shows Pette only over VS Code and terminals. |

### Safety rules
- **Cancel is always there.** Every send counts down (2.5–6 s, set by the world model) with Cancel and Send now. A pinch cancels, and a gesture never sends. Pause stops the countdown and drops the draft.
- The code attached to a message comes from git and the editor, not from the model. Model output can only choose an allowlisted platform. It never chooses URLs or mentions; mentions come from your contacts file.
- File contents, diffs and audio stay in memory for one request. Camera frames are only drawn on screen. None of them are logged or written to disk. The db stores facts only (event names, timings, recipient + platform + outcome). See [db/README.md](db/README.md).
- **Rate caps:** at most one AI request every 3 seconds and 50 per session. No automatic retries.
- Data sent to Groq, ElevenLabs, Slack, Discord or your SMTP server is subject to their terms.

## Development

```powershell
py -m pytest -q                      # all tests (Qt offscreen; no keys, mic, camera or network)
py tools/run_checks.py               # tests + live checks (Groq, ElevenLabs transcription, camera) + UI states
py tools/sandbox.py                  # manual testing: the real app + a control panel (below)
cd web; npm install; npm run build   # rebuild pet/web/index.html after editing the world-model UI
```

### Manual testing sandbox
`tools/sandbox.py` runs the real app next to a control panel, so you can test every feature by hand without touching real channels or your real memory:
- **Webhook inbox:** Slack/Discord posts land in the panel (with the code block), not in your channels. *Make webhooks fail* tests the HTTP 500 path. `--live-webhooks` posts for real.
- **Fake VS Code:** points the active file at a throwaway git repo holding the buggy fixture with an *uncommitted fix* (a real diff to summarise), the committed file (no changes), or no file. `--real-ide` uses the extension instead.
- **Simulated webcam:** face away/back, eyes closed/open, pinch. The real camera still feeds the orb unless `--no-camera`.
- **Triggers:** click Pette, pause, world-model window, camera on/off.
- **Checklist:** 23 scenarios with steps and expected results. Mark Pass/Fail/Skip, add notes, and save to `test_results/MANUAL_TEST.md`.

## Status
- **Checked:** all unit/integration tests and live checks in [test_results/REPORT.md](test_results/REPORT.md). The full send was checked through the real app into the sandbox inbox (Slack post with summary + ```diff``` block). The real webcam was checked: preview frames, hand detection, and *open hand* / *pinch* recognised. Live ElevenLabs transcription was checked with a synthesized voice. Live Slack/Discord webhooks and live Groq drafting were checked earlier.
- **Known limits:**
  - Live (agent) transcription loses the first words while the agent's greeting plays, until its First message is cleared or `ELEVENLABS_API_KEY` (Scribe) is set.
  - Slack webhooks post to one fixed channel (the recipient is @-mentioned).
  - The `keyboard` hotkey may need the same privilege level as the focused window.
