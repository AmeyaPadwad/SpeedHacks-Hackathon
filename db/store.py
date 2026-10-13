"""SQLite event cache. Thread-safe (one lock); every write is a tiny row, so the UI thread can call it."""
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

RETENTION_DAYS = 30
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (      -- webcam + UI signals: face_away, eyes_closed, pinch, ...
    ts REAL NOT NULL, kind TEXT NOT NULL, duration REAL, detail TEXT
);
DROP TABLE IF EXISTS messages;          -- messaging was removed; drop its old recipient/platform facts
CREATE TABLE IF NOT EXISTS offers (      -- SOMA suggestions shown and what you did with them
    ts REAL NOT NULL, kind TEXT NOT NULL, outcome TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blink_baseline ( -- open-eye blink-score summaries for threshold calibration
    ts REAL NOT NULL, mean REAL NOT NULL, p95 REAL NOT NULL, samples INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events(kind, ts);
"""
TABLES = ("events", "offers", "blink_baseline")


def default_path() -> Path:
    return Path(os.environ.get("FLOWSTATE_DB") or Path(__file__).resolve().parent / "pette_cache.sqlite3")


class Store:
    def __init__(self, path: Path | str | None = None, clock=time.time):
        self.path = str(path or default_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self.enabled = True
        self.prune()

    # --- writes -------------------------------------------------------------
    def _insert(self, table: str, cols: tuple, values: tuple):
        if not self.enabled:
            return
        with self._lock:
            self._db.execute(f"INSERT INTO {table} (ts, {', '.join(cols)}) VALUES (?{', ?' * len(cols)})",
                             (self._clock(), *values))
            self._db.commit()

    def log_event(self, kind: str, duration: float | None = None, **detail):
        self._insert("events", ("kind", "duration", "detail"),
                     (kind, duration, json.dumps(detail) if detail else None))

    def log_offer(self, kind: str, outcome: str):
        """kind: 'soma:<template or action>'; outcome: shown | accepted | dismissed | ignored. Never the text."""
        self._insert("offers", ("kind", "outcome"), (kind, outcome))

    def log_blink_baseline(self, mean: float, p95: float, samples: int):
        self._insert("blink_baseline", ("mean", "p95", "samples"), (mean, p95, samples))

    # --- reads --------------------------------------------------------------
    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def counts(self) -> dict[str, int]:
        return {t: self.query(f"SELECT COUNT(*) FROM {t}")[0][0] for t in TABLES}

    # --- housekeeping -------------------------------------------------------
    def prune(self, days: int = RETENTION_DAYS):
        cutoff = self._clock() - days * 86400
        with self._lock:
            for t in TABLES:
                self._db.execute(f"DELETE FROM {t} WHERE ts < ?", (cutoff,))
            self._db.commit()

    def clear(self):
        with self._lock:
            for t in TABLES:
                self._db.execute(f"DELETE FROM {t}")
            self._db.commit()
            self._db.execute("VACUUM")

    def close(self):
        with self._lock:
            self._db.close()
