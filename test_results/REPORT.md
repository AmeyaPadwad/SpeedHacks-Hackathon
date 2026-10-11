# Pette check report

- **When:** 2026-10-11 16:18 local, took 55s
- **Code:** `dev` @ `b2eace0` (checks run on the working tree before this report was committed)
- **Result:** 7 pass, 1 fail, 3 skipped, 1 info
- **Regenerate:** `py tools/run_checks.py` (add `--webhooks` to also post a test message)

| Check | Result | Detail |
|---|---|---|
| Environment | INFO | Windows 11 (10.0.26200), Python 3.13.9, PySide6 6.8.3, OpenCV 5.0.0, MediaPipe 1.1.0, openai 2.46.0 |
| Unit + integration tests | PASS | 106/106 passed, 0 failed, 0 errors, 0 skipped in 30.4s |
| MediaPipe models | PASS | face + hand landmarker present |
| Webcam: face / eyes / hands (live) | PASS | 106 frames in 4s, face in 106/106, hand in 0/106, open-eye blink score avg 0.25 (closed threshold 0.6); camera released |
| Foreground app detection | PASS | foreground = Code.exe -> show Pette |
| VS Code bridge snapshot | SKIP | no snapshot at ~/.flowstate_current.json (extension not installed or no file open) |
| db cache + world model | PASS | write -> learn -> read round trip in a temp db |
| Groq (qwen/qwen3.8-27b / whisper-large-v3-turbo) | PASS | 1-token request succeeded |
| ElevenLabs transcription | **FAIL** | live mode, 6.1s: “The dashboard bug is fixed.” (42% of words). The agent's spoken greeting swallows the start of the command: clear 'First message' in the ElevenLabs agent, or set ELEVENLABS_API_KEY for Scribe |
| Slack webhook | SKIP | configured; live post skipped (run with --webhooks) |
| Discord webhook | SKIP | configured; live post skipped (run with --webhooks) |
| UI states (screenshots) | PASS | 10/10 states rendered on the real display -> test_results/ui_states.png |

## Tests by file

| File | Passed | Failed |
|---|---|---|
| `test_brain` | 16 | 0 |
| `test_context` | 6 | 0 |
| `test_db` | 11 | 0 |
| `test_dispatch` | 10 | 0 |
| `test_e2e` | 3 | 0 |
| `test_eleven` | 6 | 0 |
| `test_focus` | 10 | 0 |
| `test_llm` | 4 | 0 |
| `test_message_cv` | 12 | 0 |
| `test_pipeline` | 13 | 0 |
| `test_sandbox` | 4 | 0 |
| `test_vision` | 8 | 0 |
| `test_voice` | 3 | 0 |

## UI states

![Pette UI states](ui_states.png)

Privacy: live checks record counts and timings only. No camera frames, screen images, file contents or message text were saved.
