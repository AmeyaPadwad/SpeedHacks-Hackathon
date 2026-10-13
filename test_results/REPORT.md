# Pette check report

- **When:** 2026-10-13 01:44 local, took 41s
- **Code:** `updated-pette` @ `230d9b2` (checks run on the working tree before this report was committed)
- **Result:** 6 pass, 0 fail, 2 skipped, 1 info
- **Regenerate:** `py tools/run_checks.py` (add `--webhooks` to also post a test message)

| Check | Result | Detail |
|---|---|---|
| Environment | INFO | Windows 11 (10.0.26200), Python 3.12.12, PySide6 6.8.3, OpenCV 5.0.0, MediaPipe 1.1.0 |
| Unit + integration tests | PASS | 81/81 passed, 0 failed, 0 errors, 0 skipped in 21.4s |
| MediaPipe models | PASS | face + hand landmarker present |
| Webcam: face / eyes / hands (live) | SKIP | --no-camera |
| Foreground app detection | PASS | foreground = comet.exe -> hide Pette |
| db cache + world model | PASS | write -> learn -> read round trip in a temp db |
| SOMA service (synthetic screen) | PASS | 4.2s, decision: SPEAK Write a test for the code in "handlers.py"  (SUGGEST_NEXT_STEP#2 p=0.33 >= threshold 0.10 and beats NO_S |
| SOMA tests | SKIP | --no-soma-tests |
| UI states (screenshots) | PASS | 15/15 states rendered on the real display -> test_results/ui_states.png |

## Tests by file

| File | Passed | Failed |
|---|---|---|
| `test_actions` | 5 | 0 |
| `test_browser` | 13 | 0 |
| `test_db` | 8 | 0 |
| `test_e2e` | 3 | 0 |
| `test_focus` | 10 | 0 |
| `test_hotkeys` | 4 | 0 |
| `test_sandbox` | 4 | 0 |
| `test_soma` | 13 | 0 |
| `test_ui` | 8 | 0 |
| `test_vision` | 13 | 0 |

## UI states

![Pette UI states](ui_states.png)

Privacy: live checks record counts and timings only. No camera frames, screen images, file contents or suggestion text were saved. SOMA's check runs on a synthetic screen in a temp data dir.
