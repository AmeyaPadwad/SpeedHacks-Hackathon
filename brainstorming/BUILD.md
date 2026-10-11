# BUILD — Unblock Pet in four hours

## Goal and hackathon fit

Build a floating desktop pet that helps a hackathon builder find a tool for a blocked task. Click the pet, describe the task, get relevant tools/agents from **Darwin**, and copy a connection prompt into an existing AI client.

**Pitch:** “Stop spending your build session searching for the tool that can do the next step. Your desktop companion finds it and prepares the handoff.”

Keep the pet from the original plan. Remove webcam reactions and screenshot understanding: they consume build time without being essential to this bottleneck. The four-hour MVP saves time on discovery and setup; autonomous task completion is a future extension.

**Ship:** working source launch on the demo laptop, README, and a 90-second demo. Assume one builder comfortable with Python. Build on the laptop's existing OS; Windows distribution is deferred if the demo machine is macOS. The four hours include setup, verification, and rehearsal.

## Required sponsor: Darwin

Darwin supplies the core discovery step. Its current [developer quickstart](https://darwin.so/docs/get-started/quickstart) documents anonymous Search with no API key:

```http
POST https://api.darwin.so/api/v3/search
Content-Type: application/json

{"query":"MCP tool to audit a website for WCAG accessibility issues","maxResults":3}
```

Read `response.agents`. Show returned names, reasons, uncertainties, readiness, and required setup. Preserve exact `agentId`, `capabilityId`, and `connectionPrompt`. Handle `response.question` and `response.noMatchReason`. Copy the returned prompt only after a user click. Search does not execute the task; Darwin-managed Act is a credential-gated pilot and outside this MVP.

Use HTTPX directly. The current quickstart documents v3 while the [Python SDK README](https://github.com/darwin-studios/darwin-python) still describes v2. Inspect a real response before implementing the parser. Docs checked October 11, 2026; live availability and result quality remain untested.

**By minute 20:** obtain one useful live result. Start with accessibility-tool discovery; select another task if results are weak. Ask the sponsor representative for help if access or the contract differs, and confirm submission requirements. This technical integration does not guarantee prize eligibility. A labeled fixture can unblock development, but cannot replace the required live sponsor integration.

## P0 and scope cuts

- Draggable, always-on-top pet, drawn locally with simple idle/searching/success/paused expressions.
- Click opens one compact task panel with typed context, capped at 1,000 characters.
- Explicit **Find a tool with Darwin** button explaining that entered text goes to Darwin.
- Up to three result cards with match details, limitations, and setup requirements.
- Exact connection-prompt preview and **Copy connection prompt** button; disable when absent.
- Loading, clarification, no-match, timeout, and offline/error states with editable input and manual retry.
- Visible Pause/Resume and Quit controls.

**Cut:** webcam, MediaPipe, face/eye/hand detection, screenshot capture/upload, Gemini or another LLM, autonomous execution, payments, calls/messages, persistent memory, CLI adapters, tray integration, generated art, installer/signing, and PyInstaller. No second sponsor integration.

**Stretch only after P0 passes before 3:00:** remembered position or one extra animation.

## Stack and setup

One Python process: **PySide6 + HTTPX**. QPainter draws a pet with circles, ears, and eyes. No server, database, Node, GPU, or agent framework. Use an installed Python compatible with PySide6 and freeze dependencies after setup succeeds.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install PySide6 httpx
.venv/bin/python -c "import PySide6, httpx; print('Imports OK')"
.venv/bin/python -m pip freeze > requirements.lock.txt
.venv/bin/python main.py
```

On Windows create the venv with `py -m venv .venv` and use `.venv/Scripts/python.exe` for subsequent commands. Ignore `.venv/`, `.env`, logs, and build outputs in git.

```text
main.py               QApplication and lifecycle
pet/ui.py             pet, task panel, results, pause/quit
pet/darwin.py         HTTP worker, parser, error handling
tests/test_darwin.py   small stdlib unittest suite
README.md             setup, sponsor use, demo, limitations
```

Qt owns the main thread. Run HTTP in one QThread worker and deliver results through signals; never mutate widgets from the worker. Allow one request at a time and disable duplicate submits. Enforce a 20-second wall-clock deadline; socket timeouts alone are not a total deadline. No automatic retries.

Assign each query a generation ID. Pause and Quit invalidate it; discard stale signals. Pause stops new requests and clears results, but cannot recall transmitted text. Shut down the worker cleanly on Quit. Render returned text as plain text, cap field lengths, handle missing/unknown fields, and never execute returned prompts. Preserve sponsored-placement labels when present. Missing setup does not prove a tool is ready.

Only typed queries go to Darwin after clicking Search. No camera, screen, or clipboard reads. Keep queries and responses in memory; do not log them. Copy writes the clipboard only on an explicit click. The README must disclose cloud processing and relevant provider terms.

## Four-hour milestones

| Elapsed | Work and checkpoint |
|---|---|
| 0:00–0:20 | Install dependencies, inspect a live Darwin response, validate demo query, show an empty Qt window. |
| 0:20–1:00 | Draw and drag pet; task panel, Pause/Resume/Quit. |
| 1:00–1:50 | Worker, parser, live results, loading/errors. **A typed task must produce real sponsor results by 1:50.** |
| 1:50–2:30 | Result details, exact prompt preview/copy, clarification, stale-result handling. |
| 2:30–3:10 | Verify live flow and failure paths; fix demo blockers. |
| 3:10–3:35 | README, measured comparison, clean launch, backup recording. |
| 3:35–4:00 | Feature freeze, rehearse 90-second demo, fix launch/demo issues. |

If behind, simplify art, reduce to one result, then fall back to a normal Qt window if floating behavior is troublesome. Preserve live sponsor search and prompt handoff. Keep at least 30 minutes for checks and rehearsal. Source launch is the intended deliverable.

## Acceptance checks

- [ ] Fresh launch works on the demo laptop with documented steps.
- [ ] Pet drags and stays responsive during a live request.
- [ ] A new task produces real results visibly attributed to Darwin.
- [ ] Match details/setup are readable; absent prompt disables Copy.
- [ ] Copy exactly matches the preview and requires a click.
- [ ] Empty input, no matches, clarification, timeout, HTTP errors, and malformed replies are recoverable.
- [ ] Duplicate submits cannot overlap; late results after Pause are ignored.
- [ ] Quit stops the worker and closes the application.
- [ ] No queries/replies appear in persisted files or logs.

Use a few stdlib unittest cases with mocked responses for parsing, malformed replies, and missing prompts. Manually test pause during a request and one live query. Record actual results; this document is a plan, not evidence that the app has been implemented or tested.

## 90-second demo and evidence of time saved

1. **0–15 seconds:** Show a website and the pet. “I need an accessibility audit before shipping; finding and configuring the right tool interrupts my work.”
2. **15–45 seconds:** Enter the task and show a live Darwin result, match explanation, and setup requirements.
3. **45–65 seconds:** Preview/copy the connection prompt and show the prepared handoff in an existing AI client. Call it discovery and handoff. Claim a completed audit only if a provider actually ran it.
4. **65–90 seconds:** Show measured manual discovery time versus the pet's time to a usable handoff, highlight Darwin, and describe approved execution as the next step.

Measure the same task to the same endpoint: a relevant tool with usable setup/handoff instructions. Include API wait time, label rehearsed comparisons, and avoid invented “hours saved” claims. Clearly label backup recordings and sample data.
