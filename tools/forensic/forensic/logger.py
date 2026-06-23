"""ForensicLogger — Tier-1 append-only JSON Lines stream with SHA-256 chain.

Design properties
-----------------
* Append-only on the filesystem: the logger opens the file in ``"ab"`` mode
  and never seeks before existing bytes.
* Cryptographic chain: every record stores ``prev_hash`` (== the previous
  record's ``record_hash``) and its own ``record_hash`` computed over the
  canonical JSON of the record minus the ``record_hash`` field itself.
* Multi-process safety: every write acquires an exclusive ``fcntl.flock`` on
  a sibling lock file; the in-memory ``_last_hash`` is refreshed from the
  file's tail under the lock to tolerate other writers.
* Durability: ``flush`` + ``os.fsync`` after every record so a crash cannot
  leave half-written lines.
* Genesis: when the log is empty the manifest record's ``prev_hash`` is
  ``GENESIS_HASH`` (sixty-four zeros). The first record SHOULD be a
  ``manifest`` event but the logger does not enforce this at write time;
  the verifier checks it.

The class is intentionally I/O-bound and synchronous. For very high-throughput
RAG runs (~3000 queries) the cost is dominated by the model call,
not by the logger, so simplicity wins over batching.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Optional, Type

from .hashing import canonical_json, hash_record
from .schema import GENESIS_HASH, SCHEMA_VERSION, EventType, Stream

log = logging.getLogger(__name__)


class ForensicLogger:
    """Append events to a Tier-1 JSON Lines file with SHA-256 chaining."""

    def __init__(
        self,
        log_path: str | Path,
        *,
        run_id: str,
        stream: Stream | str = Stream.FORENSIC,
        create_parents: bool = True,
    ) -> None:
        self.log_path = Path(log_path)
        if create_parents:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.log_path.with_suffix(self.log_path.suffix + ".lock")
        self.run_id = run_id
        self.stream = Stream(stream).value
        self._fh = open(self.log_path, "ab", buffering=0)
        self._thread_lock = threading.Lock()
        self._last_hash = self._read_tail_hash()
        log.info(
            "ForensicLogger opened path=%s stream=%s run_id=%s tail=%s",
            self.log_path, self.stream, self.run_id, self._last_hash[:12],
        )

    # ── public API ────────────────────────────────────────────────────────

    def append(
        self,
        event_type: EventType | str,
        payload: Mapping[str, Any],
        *,
        session_id: Optional[str] = None,
        user_pseudonym: Optional[str] = None,
    ) -> str:
        """Append one record. Returns the new record's ``record_hash``."""
        evt = EventType(event_type).value
        record_id = str(uuid.uuid4())
        ts_iso = datetime.now(timezone.utc).isoformat()
        ts_mono = time.monotonic_ns()

        with self._thread_lock, open(self.lock_path, "w") as lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                # Re-sync with the file in case another process appended.
                self._last_hash = self._read_tail_hash()

                envelope = {
                    "schema_version": SCHEMA_VERSION,
                    "record_id": record_id,
                    "ts_iso": ts_iso,
                    "ts_monotonic_ns": ts_mono,
                    "stream": self.stream,
                    "run_id": self.run_id,
                    "session_id": session_id,
                    "user_pseudonym": user_pseudonym,
                    "event_type": evt,
                    "payload": dict(payload),
                    "prev_hash": self._last_hash,
                }
                envelope["record_hash"] = hash_record(envelope)

                line = canonical_json(envelope) + b"\n"
                self._fh.write(line)
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._last_hash = envelope["record_hash"]
                return envelope["record_hash"]
            finally:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()

    # Context-manager sugar.
    def __enter__(self) -> "ForensicLogger":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ── internals ─────────────────────────────────────────────────────────

    def _read_tail_hash(self) -> str:
        """Return the record_hash of the last line, or GENESIS_HASH if empty.

        Implemented as a backwards byte scan to avoid O(n) reads on long logs.
        """
        size = self.log_path.stat().st_size if self.log_path.exists() else 0
        if size == 0:
            return GENESIS_HASH

        # Read up to 256 KiB from the tail; one record is unlikely to exceed
        # this even with top-5 logprobs over a 512-token completion.
        window = min(size, 1 << 18)
        with open(self.log_path, "rb") as fh:
            fh.seek(size - window)
            tail = fh.read(window)

        # Find the last non-empty line in the window.
        lines = [ln for ln in tail.split(b"\n") if ln.strip()]
        if not lines:
            raise RuntimeError(
                f"log {self.log_path} is non-empty but contains no complete records"
            )
        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"final line of {self.log_path} is not valid JSON: {exc}"
            ) from exc
        rh = last.get("record_hash")
        if not isinstance(rh, str) or len(rh) != 64:
            raise RuntimeError(
                f"final record of {self.log_path} has invalid record_hash"
            )
        return rh

    # ── optional OpenTimestamps anchoring ─────────────────────────────────

    def anchor_with_opentimestamps(self, *, ots_binary: str = "ots") -> Path:
        """Invoke ``ots stamp`` on the current log file.

        Produces a ``.ots`` sidecar next to the log. Schedule this daily
        (cron / systemd timer) per the L-C decision; it is intentionally
        not invoked on every append. Raises ``FileNotFoundError`` if the
        ``ots`` binary is not installed — the caller decides whether to
        treat that as fatal or as a warning.
        """
        result = subprocess.run(
            [ots_binary, "stamp", str(self.log_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ots stamp failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        sidecar = self.log_path.with_suffix(self.log_path.suffix + ".ots")
        if not sidecar.exists():
            raise RuntimeError(f"expected sidecar {sidecar} not produced")
        return sidecar
