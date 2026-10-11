# db — Pette's local cache and world model

`pette_cache.sqlite3` (created at first run, gitignored) is a small local log that Pette learns from.
Path override: `FLOWSTATE_DB`. Rows older than 30 days are pruned at startup.

| Table | Rows | Never contains |
|---|---|---|
| `events` | `face_away`/`face_back`/`away_ended` (with duration), `eyes_closed`/`eyes_open`, `pinch` and UI events | camera frames, landmarks, raw scores |
| `messages` | recipient name, platform, outcome (`drafted`/`sent`/`cancelled`/`failed`) | message text, subject |
| `offers` | proactive/agent proposals and whether you accepted or declined them | screenshots, code, analysis text |
| `blink_baseline` | 30-second summaries (mean, p95, sample count) of open-eye blink scores | per-frame data |

**What the world model changes** (`world_model.py`):
- **Platform routing:** "message Sarah" picks the platform you usually send her on. Your frequent contacts are also listed in the AI prompt.
- **Eyes-closed calibration:** the threshold is set 0.15 above your open-eye baseline (kept between 0.5 and 0.85). This cuts false triggers from glasses, lighting or eye shape.
- **Interruption back-off:** if you decline most proactive offers, the cooldown triples. If you accept most, offers come sooner, but never more often than every 5 minutes.
- **Presence stats:** shown in the tray under **What Pette learned**.

Tray: **Remember my patterns** (on by default) turns logging on or off. **Clear Pette memory** deletes everything.
