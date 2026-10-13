# db — Pette's local cache and world model

`pette_cache.sqlite3` (created at first run, gitignored) is a small local log that Pette learns from.
Path override: `FLOWSTATE_DB`. Rows older than 30 days are pruned at startup.

| Table | Rows | Never contains |
|---|---|---|
| `events` | `face_away`/`face_back`/`away_ended` (with duration), `eyes_closed`/`eyes_open`, `pinch` | camera frames, landmarks, raw scores |
| `offers` | kind of SOMA suggestion (`soma:review_changes`, `soma:open_relevant_file`, ...) and what you did: `shown`/`accepted`/`dismissed`/`ignored` | suggestion text, file names or paths, screenshots |
| `blink_baseline` | 30-second summaries (mean, p95, sample count) of open-eye blink scores | per-frame data |

The old `messages` table (from when Pette sent Slack/Discord/email) is dropped at startup.

**What the world model changes** (`world_model.py`):
- **Muting:** a kind of suggestion you answer *Not now* at least 80 % of the time (after 4 answers) is no longer shown. Ignoring a card never counts as *Not now*.
- **Eyes-closed calibration:** the threshold is set 0.15 above your open-eye baseline (kept between 0.5 and 0.85). This cuts false triggers from glasses, lighting or eye shape.
- **Shown in the world-model panel:** suggestions shown/done/not now per kind, presence stats.

Tray: **Remember my patterns** (on by default) turns logging on or off. **Clear Pette memory** deletes everything (and un-mutes every kind).
SOMA keeps its own memory (`SOMA/data/`); remove an indexed folder from Pette's tray to forget it there.
