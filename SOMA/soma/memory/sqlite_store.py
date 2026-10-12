"""Thread-safe SQLite store for memory metadata, permissions, provenance and feedback.

Raw files stay where they are. The store keeps only a short, redacted
``text_summary`` (capped at ``SUMMARY_CAP`` characters), a reference to the
embedding in the vector index, and JSON provenance. There is deliberately no
column for raw file contents, screenshots or screen text.

Tables:

- ``memory_items`` and ``feedback_events``: exactly the schema in plan section 6.
- ``sources``: one row per indexed root (allowed flag, trust 0..1, blocked flag).
- ``suggestion_log``: which suggestion was shown and when (no screen text).
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from soma.types import FeedbackEvent, FeedbackOutcome, MemoryItem, SourceType, new_id

SUMMARY_CAP = 500
"""Maximum characters of ``text_summary`` ever written to disk."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
  id TEXT PRIMARY KEY,
  source_uri TEXT,
  source_type TEXT NOT NULL,
  user_approved INTEGER NOT NULL,
  text_summary TEXT,
  embedding_ref TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  expires_at TEXT,
  tags_json TEXT,
  provenance_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback_events (
  id TEXT PRIMARY KEY,
  suggestion_id TEXT NOT NULL,
  outcome TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  context_ref TEXT,
  consented_for_training INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sources (
  root TEXT PRIMARY KEY,
  allowed INTEGER NOT NULL DEFAULT 1,
  trust REAL NOT NULL DEFAULT 0.5,
  blocked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS suggestion_log (
  id TEXT PRIMARY KEY,
  action TEXT NOT NULL,
  candidate_key TEXT NOT NULL,
  screen_version INTEGER NOT NULL,
  shown_at TEXT NOT NULL,
  confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_source_uri ON memory_items(source_uri);
CREATE INDEX IF NOT EXISTS idx_feedback_context ON feedback_events(context_ref);
"""

DEFAULT_TRUST = 0.5


# ---------------------------------------------------------------- helpers


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def to_iso(t: datetime | str | None) -> str:
    """Normalise a datetime (or ISO string, or ``None`` for now) to ISO-8601 UTC."""
    if t is None:
        return utc_now_iso()
    if isinstance(t, str):
        return t
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).isoformat()


def parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string to an aware UTC datetime (naive values are taken as UTC)."""
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def norm_path(p: str | Path) -> str:
    """Resolved, case-normalised path string for reliable comparisons on Windows."""
    return os.path.normcase(str(Path(p).expanduser().resolve()))


def path_within(path: str | Path, root: str | Path) -> bool:
    """True when ``path`` is ``root`` or lies inside it (after resolving)."""
    p, r = norm_path(path), norm_path(root)
    if p == r:
        return True
    return p.startswith(r.rstrip("\\/") + os.sep)


def cap_summary(text: str | None, cap: int = SUMMARY_CAP) -> str:
    """Trim a summary to ``cap`` characters (with an ellipsis marker when cut)."""
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


# ---------------------------------------------------------------- store


class MemoryStore:
    """SQLite-backed, thread-safe store for :class:`MemoryItem` and feedback.

    ``path`` may be a file path or ``":memory:"``. Derived files (thumbnails)
    live in :attr:`derived_dir`, which deletion purges per item.
    """

    def __init__(self, path: str | Path, summary_cap: int = SUMMARY_CAP) -> None:
        self.path = str(path)
        self.summary_cap = summary_cap
        self._lock = threading.RLock()
        in_memory = self.path == ":memory:"
        if not in_memory:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            created = not p.exists()
            self.derived_dir = p.parent / f"{p.stem}_derived"
        else:
            created = False
            self.derived_dir = Path(os.devnull).parent / "soma_derived_unused"
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if not in_memory:
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
        if created:
            self._restrict_permissions(Path(self.path))

    # -- lifecycle
    @staticmethod
    def _restrict_permissions(p: Path) -> None:
        """Owner-only read/write where the OS supports it (best-effort on Windows)."""
        try:
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def table_columns(self, table: str) -> list[tuple[str, str]]:
        """``(name, declared_type)`` for each column of ``table`` (schema inspection)."""
        rows = self._query(f"PRAGMA table_info({table})")
        return [(r["name"], r["type"]) for r in rows]

    # -- memory items
    @staticmethod
    def _row_to_item(r: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=r["id"],
            source_type=SourceType(r["source_type"]),
            user_approved=bool(r["user_approved"]),
            text_summary=r["text_summary"] or "",
            source_uri=r["source_uri"],
            tags=json.loads(r["tags_json"] or "[]"),
            provenance=json.loads(r["provenance_json"] or "{}"),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            expires_at=r["expires_at"],
            embedding_ref=r["embedding_ref"],
        )

    def upsert_item(self, item: MemoryItem) -> MemoryItem:
        """Insert or replace ``item``. Fills timestamps and caps the summary."""
        now = utc_now_iso()
        existing = self.get_item(item.id)
        item.created_at = item.created_at or (existing.created_at if existing else now)
        item.updated_at = now if (existing or not item.updated_at) else item.updated_at
        item.text_summary = cap_summary(item.text_summary, self.summary_cap)
        item.embedding_ref = item.embedding_ref or item.id
        self._exec(
            """INSERT OR REPLACE INTO memory_items
               (id, source_uri, source_type, user_approved, text_summary, embedding_ref,
                created_at, updated_at, expires_at, tags_json, provenance_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (item.id, item.source_uri, SourceType(item.source_type).value, int(item.user_approved),
             item.text_summary, item.embedding_ref, item.created_at, item.updated_at,
             item.expires_at, json.dumps(list(item.tags)), json.dumps(item.provenance, default=str)),
        )
        return item

    add_item = upsert_item

    def get_item(self, item_id: str) -> MemoryItem | None:
        rows = self._query("SELECT * FROM memory_items WHERE id = ?", (item_id,))
        return self._row_to_item(rows[0]) if rows else None

    def list_items(self, approved_only: bool = False,
                   source_types: Iterable[SourceType] | None = None) -> list[MemoryItem]:
        """All items, optionally only user-approved and/or of given source types."""
        sql, params = "SELECT * FROM memory_items WHERE 1=1", []
        if approved_only:
            sql += " AND user_approved = 1"
        if source_types is not None:
            types = [SourceType(t).value for t in source_types]
            if not types:
                return []
            sql += f" AND source_type IN ({','.join('?' * len(types))})"
            params += types
        sql += " ORDER BY created_at"
        return [self._row_to_item(r) for r in self._query(sql, params)]

    def find_by_source_uri(self, uri: str) -> MemoryItem | None:
        rows = self._query("SELECT * FROM memory_items WHERE source_uri = ?", (uri,))
        return self._row_to_item(rows[0]) if rows else None

    def items_under(self, folder: str | Path) -> list[MemoryItem]:
        """Items whose ``source_uri`` lies inside ``folder``."""
        return [i for i in self.list_items()
                if i.source_uri and path_within(i.source_uri, folder)]

    def set_approved(self, item_id: str, approved: bool) -> None:
        self._exec("UPDATE memory_items SET user_approved = ?, updated_at = ? WHERE id = ?",
                   (int(approved), utc_now_iso(), item_id))

    def update_summary(self, item_id: str, summary: str) -> None:
        self._exec("UPDATE memory_items SET text_summary = ?, updated_at = ? WHERE id = ?",
                   (cap_summary(summary, self.summary_cap), utc_now_iso(), item_id))

    def delete_item(self, item_id: str) -> bool:
        return self._exec("DELETE FROM memory_items WHERE id = ?", (item_id,)).rowcount > 0

    def count_items(self) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM memory_items")[0]["n"])

    # -- feedback
    def add_feedback(self, event: FeedbackEvent) -> FeedbackEvent:
        """Persist a feedback event (edited text is not stored, only the outcome)."""
        event.observed_at = event.observed_at or utc_now_iso()
        self._exec(
            """INSERT OR REPLACE INTO feedback_events
               (id, suggestion_id, outcome, observed_at, context_ref, consented_for_training)
               VALUES (?,?,?,?,?,?)""",
            (event.id, event.suggestion_id, FeedbackOutcome(event.outcome).value,
             event.observed_at, event.context_ref, int(event.consented_for_training)),
        )
        return event

    def list_feedback(self, consented_only: bool = False) -> list[FeedbackEvent]:
        sql = "SELECT * FROM feedback_events"
        if consented_only:
            sql += " WHERE consented_for_training = 1"
        sql += " ORDER BY observed_at"
        return [FeedbackEvent(suggestion_id=r["suggestion_id"], outcome=FeedbackOutcome(r["outcome"]),
                              consented_for_training=bool(r["consented_for_training"]),
                              context_ref=r["context_ref"], id=r["id"], observed_at=r["observed_at"])
                for r in self._query(sql)]

    def delete_feedback_referencing(self, ref: str) -> int:
        """Remove feedback whose ``context_ref`` points at ``ref`` (e.g. a memory id)."""
        return self._exec("DELETE FROM feedback_events WHERE context_ref = ?", (ref,)).rowcount

    def delete_feedback(self, event_id: str) -> bool:
        return self._exec("DELETE FROM feedback_events WHERE id = ?", (event_id,)).rowcount > 0

    # -- sources
    def set_source(self, root: str | Path, allowed: bool = True, trust: float = DEFAULT_TRUST,
                   blocked: bool = False) -> None:
        """Register or update an indexed source root."""
        trust = min(1.0, max(0.0, float(trust)))
        self._exec("INSERT OR REPLACE INTO sources (root, allowed, trust, blocked) VALUES (?,?,?,?)",
                   (norm_path(root), int(allowed), trust, int(blocked)))

    def get_source(self, root: str | Path) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM sources WHERE root = ?", (norm_path(root),))
        if not rows:
            return None
        r = rows[0]
        return {"root": r["root"], "allowed": bool(r["allowed"]), "trust": float(r["trust"]),
                "blocked": bool(r["blocked"])}

    def list_sources(self) -> list[dict[str, Any]]:
        return [{"root": r["root"], "allowed": bool(r["allowed"]), "trust": float(r["trust"]),
                 "blocked": bool(r["blocked"])} for r in self._query("SELECT * FROM sources")]

    def set_source_blocked(self, root: str | Path, blocked: bool = True) -> None:
        """Block (or unblock) a source root; blocked sources are never retrieved."""
        src = self.get_source(root)
        if src is None:
            self.set_source(root, allowed=not blocked, blocked=blocked)
        else:
            self._exec("UPDATE sources SET blocked = ? WHERE root = ?", (int(blocked), src["root"]))

    def is_source_blocked(self, uri: str | Path | None) -> bool:
        """True when ``uri`` lies inside any blocked (or disallowed) source root."""
        if not uri:
            return False
        for s in self.list_sources():
            if (s["blocked"] or not s["allowed"]) and path_within(uri, s["root"]):
                return True
        return False

    def source_trust(self, uri: str | Path | None, default: float = DEFAULT_TRUST) -> float:
        """Trust of the most specific source root containing ``uri``."""
        if not uri:
            return default
        best, best_len = default, -1
        for s in self.list_sources():
            if path_within(uri, s["root"]) and len(s["root"]) > best_len:
                best, best_len = s["trust"], len(s["root"])
        return best

    # -- suggestion log
    def log_suggestion(self, action: str, candidate_key: str, screen_version: int,
                       confidence: float, shown_at: str | None = None) -> str:
        """Record that a suggestion was shown. Stores no screen text."""
        sid = new_id("log_")
        self._exec("""INSERT INTO suggestion_log
                      (id, action, candidate_key, screen_version, shown_at, confidence)
                      VALUES (?,?,?,?,?,?)""",
                   (sid, action, candidate_key, int(screen_version), shown_at or utc_now_iso(),
                    float(confidence)))
        return sid

    def list_suggestions(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM suggestion_log ORDER BY shown_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
