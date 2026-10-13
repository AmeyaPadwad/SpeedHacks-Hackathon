# Pette - The Context-Aware Voice Agent

**Pette** is a background desktop assistant designed for hackathons. She lives on your desktop as a small animated companion. You click Pette, say who to tell what, and she sends it: your message, a short summary of what you changed, and the code itself.

## The concept
Developers lose their flow every time they switch windows to tell a teammate what they just changed. With Pette you click once and speak. Pette reads the file you're working on and its uncommitted git changes, writes the update with Groq, and posts it to Slack, Discord or email with the code attached.

## The workflow
There is one workflow:

1. You're working in VS Code (or a terminal) on a file.
2. **Click Pette** (or press `Ctrl+Shift+Space`). The orb above her turns violet: *Listening*.
3. Say what you want, for example *"Slack Sarah what I changed in this file."*
4. **ElevenLabs** transcribes it, and your words appear in Pette's speech bubble.
5. The text goes to **Groq** together with the active file and its `git diff`. Groq writes the message, including a short summary of the change.
6. Pette shows the draft with the code attached (*"＋ dashboard.py · uncommitted changes"*) and counts down a few seconds with **Cancel** and **Send now**:
   - **Pinch** (thumb to index finger, on camera) or click **Send now** to send immediately.
   - Click **Cancel** to stop it.
   - Do nothing and it sends when the countdown ends.
7. The orb shows **Sent ✓** or **Not sent**, and the bubble says why if it failed.

Questions that aren't messages ("what does this file do?") get a short answer in the bubble. Nothing is sent.

### Darwin search
Say *"Search Darwin for an agent that books meetings"* or *"find a tool that converts PDFs"*. Pette calls [Darwin Search](https://darwin.so/docs/search/using-results) (`POST https://api.darwin.so/api/v3/search`, no key needed) instead of Groq. The top 5 matches become a Slack draft to "team", each with its name, readiness and why it matched, plus the `searchId`. Pinch, Send now or the countdown posts it. Pette only searches; she never connects to or runs the capabilities Darwin finds (`pet/darwin.py`).

### What gets sent
- **Your message:** written by Groq from your words, with a 1–3 sentence summary of the change.
- **The code:** attached by Pette, never written by the model. It's the uncommitted `git diff` of the active file, or your selection, or else the file itself.
  - Slack and Discord: a fenced ```diff``` or code block, trimmed to fit (Discord's 2,000-character limit is respected).
  - Email: plain text.
- **No screenshots**, ever.

### Voice: ElevenLabs transcription only
Pette never talks back. ElevenLabs is used only for speech-to-text, and the transcript is shown in the bubble:
- **Scribe** (`ELEVENLABS_API_KEY` set, recommended): Pette records until you stop talking, then makes one `scribe_v1` request.
- **Live** (only `ELEVENLABS_AGENT_ID` set, public agent, no key): Pette streams the mic to the agent's websocket and takes its transcript of your turn. The agent's replies and voice are dropped. Clear the agent's **First message** in the ElevenLabs dashboard, or its greeting talks over the start of your command.
- **No ElevenLabs settings** (or `FLOWSTATE_VOICE=whisper`): Groq Whisper.

Click Pette again while she's listening to stop. If the mic or transcription fails, a text box opens, and typing works the same way.

### The camera orb
A circle above Pette shows **you** (a mirrored 168px webcam preview, processed and drawn locally only). It also shows your hand skeleton and a *pinch* chip when you pinch. Pinch is the one recognised gesture. A glowing ring shows the status: violet *Listening* → blue *Thinking* → amber *Sending in Ns* → green *Sent ✓*, or red/grey *Not sent*.

| Webcam signal | Effect |
|---|---|
| **Pinch** while a draft is counting down | Sends it now (same as **Send now**) |
| **Pinch** at any other time | Nothing; click Pette to talk |
| **Away → back** | Pette dozes while you're away. When you return, a pending send is re-titled *"Still sending?"* |
| **Eyes closed** | Pette closes her eyes too |

The camera is on by default. `--no-camera` or the tray toggle turns it off, and Pause releases it. Frames are never stored, logged or uploaded; the db gets event names and timings only.

### The world-model window
A panel docked beside Pette shows, live, what she has learned from her local db (`db/world_model.py`) and the behaviour each fact drives:
- **Usual platform per person:** *"message Sarah"* with no platform goes to Slack *because 3 sent, 4 drafted to Sarah there*. A platform you name always wins.
- **Adaptive countdown:** 4 s by default. If you cancel often, Pette waits 6 s; if you almost never cancel, 2.5 s.
- **Also shown:** blink calibration; sent, cancelled and failed counts; presence today; and recent *who and where* (never *what*).
- **Live status card:** mirrors the orb (camera, hand, presence).

The window is built with React, Tailwind and [shadcn/ui](https://ui.shadcn.com), with [Aceternity UI](https://ui.aceternity.com) effects (spotlight header, background-gradient card, moving-border status pill). Vite compiles it into one self-contained file, `pet/web/index.html`, which Qt WebEngine shows. The Qt overlay uses the same shadcn tokens, and the bubble gets a rotating gradient border while a send counts down.

### SOMA: on-screen action suggestions
[SOMA](SOMA/README.md) (in `SOMA/`, its own venv) watches the screen with your consent and ranks next steps with its own PyTorch model. With `--soma`, Pette shows its suggestions as a glass card while she is on screen and idle: **Do it** / **Not now**.

```bash
./run_pette_soma.sh              # Git Bash: sets up both venvs with uv, starts SOMA, then Pette
./run_pette_soma.sh --demo       # extra flags go to Pette
./run_pette_soma.sh --diagnose   # why SOMA would speak or stay silent on the current screen
```

| SOMA sees | Card | Do it |
|---|---|---|
| A file in your editor | *Send the team a summary of your changes to "app.py"* / *Ask a teammate to review "app.py"* | The normal draft + countdown (Cancel still works) |
| A passing test run in a terminal | *Tell the team the tests pass* | Same |
| An error | *Ask the team for help with the error on screen?*, *Explain the error "…"* | Draft to the team, or copy the text for your coding agent |
| A web page (with `--always-show`) | *Search Darwin for tools related to "…"* | Darwin search |

- Nothing runs without **Do it**. Accepted suggestions become fixed Pette commands, so screen text never reaches Groq. An unanswered card disappears after 15 s.
- Accept / dismiss / ignore go back to SOMA (its online ranker learns your taste when SOMA's feedback storage is granted) and into Pette's db as facts (*kind, outcome*).
- **Ctrl+Alt+S** or the tray turns suggestions off (SOMA stops capturing). **Pause all** pauses SOMA too.
- Running the script is your opt-in for screen capture (stored in `SOMA/data/consent.json`). Screenshots stay in memory; password managers, private windows and sign-in prompts are skipped. Pette's own windows are excluded from capture.
- `SOMA/configs/pette.yaml` lets SOMA's top-ranked candidate through. SOMA's default config wants 55 % confidence, which the untrained model never reaches (about 30 % per option), so it stayed silent on real screens. A 45 s cooldown and a 5-minute duplicate window keep cards rare.

### Hotkeys
| Keys | Action | Setting |
|---|---|---|
| Ctrl+Shift+Space | Talk (same as clicking Pette) | `FLOWSTATE_HOTKEY` |
| Ctrl+Alt+C | Hide / show the camera orb (the camera keeps running, so pinch still works) | `PETTE_CAMERA_HOTKEY` |
| Ctrl+Alt+W | Hide / show the world-model panel | `PETTE_WORLD_HOTKEY` |
| Ctrl+Alt+S | SOMA suggestions on / off (with `--soma`) | `PETTE_SOMA_HOTKEY` |

## Quick start (Windows 11, Python 3.10+)

```powershell
py -m pip install -r requirements.txt
py tools/get_models.py   # once: MediaPipe face/hand models for the orb
# .env: GROQ_API_KEY, SLACK_WEBHOOK_URL / DISCORD_WEBHOOK_URL, ELEVENLABS_API_KEY (or ELEVENLABS_AGENT_ID)
py main.py               # click Pette and talk
py main.py --no-voice    # type instead of speaking
py main.py --demo        # offline keyword drafter (labelled DEMO); dispatch still runs for real
py main.py --no-camera --no-world --always-show
```

Settings load from `.env` in the repo root (see [.env.example](.env.example)); real environment variables take precedence. Setting names keep their `FLOWSTATE_` prefix (and the snapshot stays at `~/.flowstate_current.json`) so existing setups keep working.

Pette appears bottom-right **only while VS Code or a terminal is in front**: Windows Terminal, PowerShell, cmd, Git Bash, WezTerm, Alacritty, Tabby, Hyper and Warp.
- To show her everywhere, use `--always-show` or the tray toggle.
- To pick the apps, set `FLOWSTATE_SHOW_IN`.
- Drag her to move her.
- The tray menu has the world-model window, the camera, memory controls, **Pause all** and **Quit**.

**VS Code extension (needed for code and summaries):**
1. Copy `extensions/vscode` into `%USERPROFILE%\.vscode\extensions\flowstate-context`.
2. Reload VS Code.

The extension writes the active file's path, language, contents (first 12,000 characters), selection and workspace to `~/.flowstate_current.json`. Pette runs `git diff HEAD -- <file>` in that file's folder herself.

## Configuration

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required for drafting, and for Whisper if ElevenLabs isn't set. |
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
| `FLOWSTATE_DB` | Memory db path, default `db/pette_cache.sqlite3`. |
| `PETTE_CAMERA_HOTKEY`, `PETTE_WORLD_HOTKEY`, `PETTE_SOMA_HOTKEY` | Hide the camera orb / the world panel, toggle SOMA suggestions. Defaults `ctrl+alt+c`, `ctrl+alt+w`, `ctrl+alt+s`. |
| `PETTE_SOMA_ENDPOINT` | SOMA endpoint file (`{url, token}`, written by `SOMA/scripts/serve.py`). Set → suggestions on. |

**Contacts** map spoken names to IDs, so messages @-mention real people:
```json
{ "sarah": { "slack": "U012ABCDEF", "discord": "123456789012345678", "email": "sarah@example.com" } }
```
If a platform has no webhook or SMTP host, nothing is sent: the bubble says it isn't configured and the orb shows *Not sent*.

## Architecture

**Flow:** click or hotkey → ElevenLabs transcript → bubble → file + `git diff` (or Darwin search) → Groq JSON draft → code attached → countdown (Cancel / Send now / pinch) → dispatch → orb and world-model update.

| Module | File | What it does |
|---|---|---|
| Workflow | `pet/pipeline.py` | The one state machine (idle → listening → thinking → pending → sending → sent / failed / cancelled), on the Qt main thread. Slow steps run on worker threads and return through `_deliver`. Emits `status` for the orb and the world window. |
| Voice | `pet/eleven.py`, `pet/voice.py` | ElevenLabs Scribe or live agent transcription (agent audio ignored), Groq Whisper fallback, global hotkey. |
| Context | `pet/context.py` | VS Code snapshot plus `git diff HEAD` of the active file (an untracked file counts as new). No screenshots. |
| AI brain | `pet/brain.py`, `pet/llm.py` | Groq JSON mode. The output must be exactly `{platform, recipient, subject, message}`, with platform `slack`, `discord`, `email` or `none`. `attach_code` adds the diff, selection or file locally. |
| Darwin | `pet/darwin.py` | Darwin Search v3 → top matches as a Slack draft. Search only. |
| Dispatch | `pet/dispatch.py` | Slack and Discord webhooks, plus SMTP. `compose()` fits the message and code into each platform's limit. Discord pings only the resolved contact. |
| Overlay | `pet/ui.py`, `pet/sprites.py` | Pette, the camera orb (preview, hand skeleton, pinch chip, status ring), and the frosted-glass bubble with the countdown. |
| Webcam | `pet/vision.py`, `pet/watcher.py` | MediaPipe face and hand tracking on a camera thread: debounced presence, eyes-closed and pinch events, plus the orb preview. |
| World model | `db/`, `pet/worldview.py`, `web/` | SQLite fact log → explainable rules (`WorldModel.snapshot()`) → the React/shadcn window. |
| Focus gate | `pet/focus.py` | Shows Pette only over VS Code and terminals. |
| SOMA companion | `pet/soma.py`, `SOMA/` | Polls SOMA's localhost API, shows suggestion cards, maps accepted ones to Pette commands, sends feedback. |

`agent.py` in the repo root is a standalone Groq drafting experiment and isn't used by the app.

### Safety rules
- **Every send counts down** (2.5–6 s, set by the world model), with **Cancel** always available. Pause stops the countdown and drops the draft. A pinch sends only a draft that is already counting down.
- The attached code comes from git and the editor, not from the model. The model can only pick an allowlisted platform. It never chooses URLs or mentions; mentions come from your contacts file.
- File contents, diffs and audio stay in memory for one request. Camera frames are only drawn on screen. None of them are logged or written to disk. The db stores facts only (event names, timings, recipient, platform and outcome); see [db/README.md](db/README.md).
- **SOMA suggestions never run by themselves.** Only **Do it** acts, accepted templates map to fixed commands, and the resulting draft still counts down.
- **Rate caps:** at most one AI request every 3 seconds and 50 per session, with no automatic retries.
- Data sent to Groq, ElevenLabs, Darwin, Slack, Discord or your SMTP server is subject to their terms.

## Development

```powershell
py -m pytest -q                      # unit/integration tests (Qt offscreen; no keys, mic, camera or network)
py tools/run_checks.py               # tests + live checks (Groq, ElevenLabs, camera) + UI states -> test_results/
py tools/sandbox.py                  # manual testing: the real app + a control panel
py tools/cv_check.py                 # live webcam signal checker
cd web; npm install; npm run build   # rebuild pet/web/index.html after editing the world-model UI
```

### Manual testing sandbox
`tools/sandbox.py` runs the real app next to a control panel, so you can test by hand without touching real channels or your real memory:
- **Webhook inbox:** Slack and Discord posts land in the panel, code block included. *Make webhooks fail* tests the HTTP 500 path. `--live-webhooks` posts for real.
- **Fake VS Code:** points the active file at a throwaway git repo holding the buggy fixture with an uncommitted fix (a real diff to summarise), the committed file (no changes), or no file. `--real-ide` uses the extension instead.
- **Simulated webcam:** face away/back, eyes closed/open and pinch. The real camera still feeds the orb unless `--no-camera`.
- **Triggers:** click Pette, pause, the world-model window, camera on/off.
- **Checklist:** scenarios with steps and expected results. Mark each Pass, Fail or Skip and save the results to `test_results/MANUAL_TEST.md`.

## Status
- **Verified live:**
  - A full send through the real app (a Slack post with a summary and a `diff` block).
  - Groq drafting.
  - ElevenLabs Scribe transcription (100% of words on a test sentence).
  - Real webcam preview and hand detection.
  - Slack and Discord webhooks.
  - Darwin Search API responses.
- **Not yet updated:** the newest changes are in place, but not every test has been updated for them:
  - Darwin search.
  - Pinch-to-send, with pinch as the only gesture.
  - The worker error fix.

  Tests and sandbox scenarios that still expect the old pinch behaviour (pinch to talk or to cancel; open-hand/point/peace/fist labels) will fail until they are updated.
- **Known limits:**
  - Live (agent) transcription loses the first words while the agent's greeting plays. Fix it by clearing the agent's First message or by using Scribe.
  - Slack webhooks post to one fixed channel; the recipient is @-mentioned.
  - The `keyboard` hotkey may need the same privilege level as the focused window.
