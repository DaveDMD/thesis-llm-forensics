"""EvidenceIndexer — Tier-2 SQLite derivation from Tier-1 JSONL.

The indexer is strictly derivative: the SQLite database can be dropped and
rebuilt from the JSONL log at any time. It exists for query convenience
(joins, filters, aggregations), not as primary evidence.

Schema
------
manifests         one row per ``manifest`` event
sessions          one row per ``session_open`` event
events            one row per Tier-1 record (full envelope minus payload columns)
detection_events  derived view over rows where event_type='detection_event'

The implicit_session_id is computed here (not by the logger), so it can be
recomputed if the 30-minute gap policy ever changes, without invalidating
the underlying evidence. The cutoff is configurable via ``gap_seconds``.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    record_id          TEXT PRIMARY KEY,
    ts_iso             TEXT NOT NULL,
    ts_monotonic_ns    INTEGER NOT NULL,
    stream             TEXT NOT NULL,
    run_id             TEXT NOT NULL,
    session_id         TEXT,
    implicit_session_id TEXT,
    user_pseudonym     TEXT,
    event_type         TEXT NOT NULL,
    payload_json       TEXT NOT NULL,
    prev_hash          TEXT NOT NULL,
    record_hash        TEXT NOT NULL,
    line_offset        INTEGER NOT NULL,
    source_log         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run     ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_user    ON events(user_pseudonym);
CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts_iso);

CREATE TABLE IF NOT EXISTS manifests (
    record_id           TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    experiment_phase    TEXT,
    model_id            TEXT,
    config_hash         TEXT,
    salt_fingerprint    TEXT,
    git_commit          TEXT,
    git_dirty           INTEGER,
    payload_json        TEXT NOT NULL,
    FOREIGN KEY(record_id) REFERENCES events(record_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    run_id              TEXT,
    opened_at           TEXT,
    closed_at           TEXT,
    user_pseudonym      TEXT
);

CREATE VIEW IF NOT EXISTS v_detection_events AS
SELECT record_id, ts_iso, session_id, user_pseudonym, payload_json
FROM events WHERE event_type = 'detection_event';
"""


class EvidenceIndexer:
    """Idempotent JSONL → SQLite indexer."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ── ingestion ─────────────────────────────────────────────────────────

    def ingest_log(
        self,
        log_path: str | Path,
        *,
        gap_seconds: int = 1800,
    ) -> dict:
        """Ingest a Tier-1 log file. Returns a stats dict."""
        log_path = Path(log_path)
        inserted = 0
        skipped = 0
        manifests = 0

        cur = self.conn.cursor()
        with open(log_path, "rb") as fh:
            offset = 0
            for line in fh:
                line_offset = offset
                offset += len(line)
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    cur.execute(
                        """INSERT OR IGNORE INTO events
                           (record_id, ts_iso, ts_monotonic_ns, stream, run_id,
                            session_id, implicit_session_id, user_pseudonym,
                            event_type, payload_json, prev_hash, record_hash,
                            line_offset, source_log)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            rec["record_id"],
                            rec["ts_iso"],
                            rec["ts_monotonic_ns"],
                            rec["stream"],
                            rec["run_id"],
                            rec.get("session_id"),
                            None,  # filled by _compute_implicit_sessions
                            rec.get("user_pseudonym"),
                            rec["event_type"],
                            json.dumps(rec["payload"], sort_keys=True,
                                       separators=(",", ":"), ensure_ascii=False),
                            rec["prev_hash"],
                            rec["record_hash"],
                            line_offset,
                            str(log_path),
                        ),
                    )
                    if cur.rowcount == 0:
                        skipped += 1
                    else:
                        inserted += 1
                        if rec["event_type"] == "manifest":
                            self._index_manifest(cur, rec)
                            manifests += 1
                        elif rec["event_type"] == "session_open":
                            cur.execute(
                                """INSERT OR REPLACE INTO sessions
                                   (session_id, run_id, opened_at, user_pseudonym)
                                   VALUES (?,?,?,?)""",
                                (
                                    rec.get("session_id"),
                                    rec["run_id"],
                                    rec["ts_iso"],
                                    rec.get("user_pseudonym"),
                                ),
                            )
                        elif rec["event_type"] == "session_close":
                            cur.execute(
                                "UPDATE sessions SET closed_at = ? WHERE session_id = ?",
                                (rec["ts_iso"], rec.get("session_id")),
                            )
                except KeyError as exc:
                    raise ValueError(
                        f"malformed record at offset {line_offset}: missing {exc.args[0]}"
                    ) from exc

        self.conn.commit()
        updated_implicit = self._compute_implicit_sessions(gap_seconds=gap_seconds)
        return {
            "inserted": inserted,
            "skipped_already_indexed": skipped,
            "manifests": manifests,
            "implicit_sessions_updated": updated_implicit,
        }

    def _index_manifest(self, cur: sqlite3.Cursor, rec: dict) -> None:
        p = rec["payload"]
        git = p.get("git", {}) or {}
        cur.execute(
            """INSERT OR REPLACE INTO manifests
               (record_id, run_id, experiment_phase, model_id,
                config_hash, salt_fingerprint, git_commit, git_dirty, payload_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                rec["record_id"],
                p.get("run_id"),
                p.get("experiment_phase"),
                (p.get("model") or {}).get("id"),
                p.get("experiment_config_hash"),
                p.get("salt_fingerprint"),
                git.get("commit_sha"),
                int(bool(git.get("dirty"))) if "dirty" in git else None,
                json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            ),
        )

    # ── implicit sessionisation ───────────────────────────────────────────

    def _compute_implicit_sessions(self, *, gap_seconds: int) -> int:
        """Re-derive implicit_session_id for rows without an explicit session_id.

        Policy: contiguous events from the same user_pseudonym with gap
        < ``gap_seconds`` belong to the same implicit session. IDs are
        synthetic strings ``"impl-{user[:12]}-{seq}"``.
        """
        cur = self.conn.cursor()
        cur.execute(
            """SELECT record_id, ts_iso, user_pseudonym
               FROM events
               WHERE session_id IS NULL AND user_pseudonym IS NOT NULL
               ORDER BY user_pseudonym, ts_iso"""
        )
        rows = cur.fetchall()
        updates: list[tuple[str, str]] = []
        from datetime import datetime
        current_user: Optional[str] = None
        current_id: Optional[str] = None
        current_seq = 0
        prev_ts: Optional[datetime] = None
        for record_id, ts_iso, user in rows:
            ts = datetime.fromisoformat(ts_iso)
            if user != current_user or prev_ts is None or \
               (ts - prev_ts).total_seconds() > gap_seconds:
                current_user = user
                current_seq += 1
                current_id = f"impl-{user[:12]}-{current_seq}"
            updates.append((current_id, record_id))
            prev_ts = ts
        cur.executemany(
            "UPDATE events SET implicit_session_id = ? WHERE record_id = ?",
            updates,
        )
        self.conn.commit()
        return len(updates)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EvidenceIndexer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
