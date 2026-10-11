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

### 1. Project Structure
*   **Language:** Python 3.10+
*   **Dependencies:** `keyboard`, `speech_recognition`, `mss`, `PySide6`, `openai`, `requests`

### 2. Module 1: Input Engine (System Hotkey & Voice Capture)
*   A background process listens for `Ctrl+Shift+Space`.
*   Triggers `speech_recognition` to record audio from the microphone until silence.
*   Sends the audio to OpenAI's Whisper API for instant transcription.

### 3. Module 2: Context Capture Engine (Vision & Text)
*   **Visual Context:** Uses `mss` to instantly grab a screenshot of the primary monitor.
*   **File Context (The VS Code Extension approach):** A lightweight companion VS Code extension constantly writes the active file's path and contents to a temporary local file (e.g., `~/.flowstate_current.txt`). The Python background app reads this file when the hotkey is pressed to get the exact code context.

### 4. Module 3: AI Brain & Darwin Search
*   **Intent Extraction:** Sends a prompt to `gpt-4o-mini`.
*   **Inputs:** Base64 screenshot + Active file text + Transcribed voice command.
*   **Prompt Instruction:** Understand the context and output a clean, plain-text summary of the task (e.g., "Create a Jira ticket titled 'Database Timeout Error'. Context: ...").
*   **Agent Discovery:** Passes the clean task string to the **Darwin Search API** (`POST /api/v3/search`) to dynamically find the best agent for the job, rather than relying on hardcoded API integrations.

### 5. Module 4: Floating UI (Confirmation)
*   Uses `PySide6` (or `Tkinter`) to spawn a borderless, always-on-top window in the center of the screen.
*   Displays the user's task, the Agent selected by Darwin (e.g., "Jira Automator"), and **[ Approve ]** / **[ Cancel ]** buttons to ensure safety before dispatching.

### 6. Module 5: Dispatch (Darwin Act API)
*   When the user approves, the app calls the **Darwin Act API** (`POST /api/v3/act`).
*   It passes the `searchId` and `agentId` to start a thread with the selected agent, completely offloading the execution of the task (Slack, Jira, Gmail, etc.) to the Darwin network.

### 7. Integration & Main Loop
*   Ties everything together in a single `main.py` entry point that initializes the listener and orchestrates the Context -> Voice -> AI -> UI -> Dispatch pipeline.
