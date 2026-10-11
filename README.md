# FlowState AI - The Context-Aware Voice Agent

FlowState is a background desktop assistant designed for hackathons. It eliminates context-switching by combining voice commands, screen context, and IDE integration to execute tasks without leaving your workflow.

## The Concept
Developers and knowledge workers lose their "flow state" every time they have to switch windows to ask ChatGPT a question, run a script, or send a message. FlowState allows you to press a hotkey and speak. The AI sees what you see, reads the file you are working on, and drafts the necessary actions or messages for your approval.

## How it works (The UX):
1. You are working in your IDE (e.g., VS Code) on a specific file.
2. You press the global hotkey (`Ctrl+Shift+Space`).
3. You speak your command: *"Send a Slack to Sarah asking her to review this code."*
4. FlowState instantly takes a screenshot of your active window and reads your currently active file (via a companion VS Code extension).
5. It transcribes your voice and sends the text, screenshot, and file contents to an LLM.
6. A floating confirmation window appears with the drafted Slack message. You click "Send".
7. You never left your IDE.

## The Architecture & Execution Plan

### 1. Project Structure & Environment
*   **Language:** Python 3.10+
*   **Dependencies:** `keyboard`, `speech_recognition`, `mss`, `PySide6`, `openai`, `requests`

### 2. Module 1: Input Engine (System Hotkey & Voice Capture)
*   A background process listens for `Ctrl+Shift+Space`.
*   Triggers `speech_recognition` to record audio from the microphone until silence.
*   Sends the audio to OpenAI's Whisper API for instant transcription.

### 3. Module 2: Context Capture Engine (Vision & Text)
*   **Visual Context:** Uses `mss` to instantly grab a screenshot of the primary monitor.
*   **File Context (The VS Code Extension approach):** A lightweight companion VS Code extension constantly writes the active file's path and contents to a temporary local file (e.g., `~/.flowstate_current.txt`). The Python background app reads this file when the hotkey is pressed to get the exact code context.

### 4. Module 3: AI Brain (The Router)
*   Sends a prompt to `gpt-4o`.
*   **Inputs:** Base64 screenshot + Active file text + Transcribed voice command.
*   **Prompt Instruction:** Understand the context from the screen/code, determine the platform and recipient based on the voice command, and output a strict JSON draft message.

### 5. Module 4: Floating UI (Confirmation)
*   Uses `PySide6` (or `Tkinter`) to spawn a borderless, always-on-top window in the center of the screen.
*   Displays the drafted message and recipient with **[ Send ]** and **[ Cancel ]** buttons to ensure safety before dispatching.

#### Overlay UI status: built (demo wiring)
The overlay is implemented with the **Pette** spritesheet. Hotkey, voice, capture, LLM and dispatch are not wired yet. The "Ask" flow in `main.py` returns a canned, labeled **DEMO** proposal, and **Send** does nothing.

```powershell
py -m pip install -r requirements.txt
py main.py
```

*   **Pet:** borderless, always-on-top, translucent window, bottom-right of the primary screen. Drag to move (kept inside the screen's available area). Click to ask; click a plain message to dismiss it.
*   **Bubble:** sits above the pet and shows a title and message. A pending proposal shows **[ Send ]** and **[ Cancel ]** and never times out on its own. Approval is single-use and bound to the proposal ID, so a second click does nothing.
*   **States:** `idle`, `thinking`, `speaking`, `sleeping` (paused). Each loops frames from the spritesheet at 15 FPS with a gentle bob.
*   **Tray:** Ask Pette, Pause all, Bring to front, Quit. Pause drops any pending proposal, blocks new bubbles and puts Pette to sleep.
*   **Integration API** (`pet/ui.py`): `overlay.propose(Proposal(...))`, `overlay.say(text)`, `overlay.set_state(...)`; signals `approved(id)`, `cancelled(id)`, `ask_requested`.

**Assets:** `images/pette-spritesheet.png` has an opaque checkerboard baked in. `python tools/prep_sprites.py` (needs `Pillow` and `numpy`) removes it and writes `assets/pette_sheet.png`, a transparent 3x2 sheet that the app loads. Frame order is in `pet/sprites.py`.

**Checked:** an offscreen Qt smoke test for single-use approval, pause behavior and layout. **Not yet checked:** on-screen run on Windows, multi-monitor behavior, real fonts, packaging.

### 6. Module 5: Dispatch Connectors
*   Simple API wrappers to dispatch the approved message.
*   Uses Slack Webhooks, Discord Webhooks, or basic SMTP/Gmail integrations for the demo.

### 7. Integration & Main Loop
*   Ties everything together in a single `main.py` entry point that initializes the listener and orchestrates the Context -> Voice -> AI -> UI -> Dispatch pipeline.
