"""EvidenceVerifier — end-to-end integrity check for a Tier-1 log file.

Checks performed
----------------
1. Each line is well-formed JSON with the required envelope keys.
2. ``schema_version`` is recognised.
3. ``record_hash`` recomputed from canonical(record \\ {record_hash}) matches
   the stored value.
4. ``prev_hash`` of record N matches ``record_hash`` of record N-1
   (genesis: sixty-four zeros).
5. ``ts_monotonic_ns`` is strictly non-decreasing.
6. The first record's ``event_type`` is ``manifest`` (warning if not).
7. Optional: an ``.ots`` sidecar exists alongside the log (informational).

The verifier is read-only and never writes to the log under inspection.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .hashing import hash_record
from .schema import ENVELOPE_KEYS, GENESIS_HASH, SCHEMA_VERSION


@dataclasses.dataclass
class VerificationIssue:
    line_number: int          # 1-indexed
    record_id: str | None
    severity: str             # "error" | "warning"
    code: str                 # short token, e.g. "chain_break"
    message: str


@dataclasses.dataclass
class VerificationReport:
    log_path: str
    total_records: int
    issues: list[VerificationIssue]
    has_manifest_first: bool
    ots_sidecar_present: bool
    # Offline OpenTimestamps binding: True if the .ots provably commits to the
    # current log's SHA256; False if a sidecar exists but does not bind; None if
    # no sidecar or the opentimestamps library is unavailable. Independent of
    # the (possibly still-pending) Bitcoin attestation.
    ots_commitment_verified: bool | None = None
    # Informational attestation state: "confirmed" | "pending" | "none" | None.
    # Never gates verification; the Bitcoin proof is pending for hours/days.
    ots_attestation_status: str | None = None

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "log_path": self.log_path,
            "total_records": self.total_records,
            "ok": self.ok,
            "has_manifest_first": self.has_manifest_first,
            "ots_sidecar_present": self.ots_sidecar_present,
            "ots_commitment_verified": self.ots_commitment_verified,
            "ots_attestation_status": self.ots_attestation_status,
            "issues": [dataclasses.asdict(i) for i in self.issues],
        }


_SUPPORTED_SCHEMAS = {"1.0", "1.1"}


class EvidenceVerifier:
    """Verifies the Tier-1 hash chain and structural invariants."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)

    def verify(self) -> VerificationReport:
        issues: list[VerificationIssue] = []
        total = 0
        has_manifest_first = False
        prev_hash = GENESIS_HASH
        prev_mono: int | None = None

        with open(self.log_path, "rb") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                rec, parse_err = _safe_load(raw)
                if parse_err is not None:
                    issues.append(VerificationIssue(
                        lineno, None, "error", "malformed_json", parse_err,
                    ))
                    continue
                total += 1

                rid = rec.get("record_id")

                # 1. Required keys
                missing = ENVELOPE_KEYS - rec.keys()
                if missing:
                    issues.append(VerificationIssue(
                        lineno, rid, "error", "missing_keys",
                        f"missing envelope keys: {sorted(missing)}",
                    ))
                    continue

                # 2. Schema version
                if rec["schema_version"] not in _SUPPORTED_SCHEMAS:
                    issues.append(VerificationIssue(
                        lineno, rid, "error", "schema_unknown",
                        f"schema_version={rec['schema_version']!r} not supported "
                        f"(known: {sorted(_SUPPORTED_SCHEMAS)})",
                    ))
                    continue

                # 3. record_hash
                expected = hash_record(rec)
                if expected != rec["record_hash"]:
                    issues.append(VerificationIssue(
                        lineno, rid, "error", "record_hash_mismatch",
                        f"computed {expected[:16]}… != stored {rec['record_hash'][:16]}…",
                    ))

                # 4. prev_hash chain
                if rec["prev_hash"] != prev_hash:
                    issues.append(VerificationIssue(
                        lineno, rid, "error", "chain_break",
                        f"prev_hash {rec['prev_hash'][:16]}… does not match "
                        f"previous record_hash {prev_hash[:16]}…",
                    ))
                prev_hash = rec["record_hash"]

                # 5. Monotonic timestamp (allow gaps across process restarts —
                #    monotonic clocks are per-process, so a reset is acceptable
                #    only when run_id changes or after a manifest record).
                mono = rec["ts_monotonic_ns"]
                if prev_mono is not None and mono < prev_mono:
                    if rec["event_type"] != "manifest":
                        issues.append(VerificationIssue(
                            lineno, rid, "warning", "monotonic_regression",
                            f"ts_monotonic_ns {mono} < previous {prev_mono} "
                            "(expected only across a fresh manifest)",
                        ))
                prev_mono = mono

                # 6. Manifest-first
                if lineno == 1 or (total == 1):
                    has_manifest_first = (rec["event_type"] == "manifest")
                    if not has_manifest_first:
                        issues.append(VerificationIssue(
                            lineno, rid, "warning", "no_manifest_first",
                            "first record is not of type 'manifest'",
                        ))

        ots_sidecar = self.log_path.with_suffix(self.log_path.suffix + ".ots")
        return VerificationReport(
            log_path=str(self.log_path),
            total_records=total,
            issues=issues,
            has_manifest_first=has_manifest_first,
            ots_sidecar_present=ots_sidecar.exists(),
            ots_commitment_verified=verify_ots_commitment(
                self.log_path, ots_sidecar
            ),
            ots_attestation_status=ots_attestation_status(ots_sidecar),
        )


def _safe_load(raw: bytes) -> tuple[dict, str | None]:
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return {}, str(exc)


def verify_ots_commitment(
    log_path: str | Path, sidecar_path: str | Path
) -> bool | None:
    """Offline check that an OpenTimestamps sidecar commits to the log's hash.

    Returns
    -------
    ``True``
        the ``.ots`` is a SHA256 detached timestamp whose committed digest
        equals ``sha256(log bytes)`` — the sidecar provably belongs to this
        exact log content.
    ``False``
        a sidecar exists but does not bind to the current log (wrong digest,
        unexpected hash op, or unparseable file).
    ``None``
        no sidecar present, or the ``opentimestamps`` library is unavailable —
        the check degrades to the informational ``ots_sidecar_present`` flag.

    This validates the *commitment structure* only; no network call and no
    confirmed Bitcoin attestation are required, so the result is available
    immediately even while the blockchain proof is still pending.
    """
    log_path = Path(log_path)
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        return None
    try:
        from opentimestamps.core.op import OpSHA256
        from opentimestamps.core.serialize import StreamDeserializationContext
        from opentimestamps.core.timestamp import DetachedTimestampFile
    except Exception:
        return None
    try:
        with open(sidecar_path, "rb") as fh:
            detached = DetachedTimestampFile.deserialize(
                StreamDeserializationContext(fh)
            )
    except Exception:
        return False
    if not isinstance(detached.file_hash_op, OpSHA256):
        return False
    digest = hashlib.sha256(log_path.read_bytes()).digest()
    return detached.timestamp.msg == digest


def ots_attestation_status(sidecar_path: str | Path) -> str | None:
    """Classify the OpenTimestamps attestation state of a sidecar, offline.

    Returns ``"confirmed"`` if the timestamp carries a Bitcoin block-header
    attestation, ``"pending"`` if it only holds calendar pending attestations
    (the normal state for hours/days after stamping, until ``ots upgrade``),
    ``"none"`` if it carries no attestation, or ``None`` when no sidecar exists,
    the file is unparseable, or the ``opentimestamps`` library is unavailable.

    Purely informational: it never gates verification. No network is used —
    confirmation is read from the sidecar itself once it has been upgraded.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        return None
    try:
        from opentimestamps.core.notary import (
            BitcoinBlockHeaderAttestation,
            PendingAttestation,
        )
        from opentimestamps.core.serialize import StreamDeserializationContext
        from opentimestamps.core.timestamp import DetachedTimestampFile
    except Exception:
        return None
    try:
        with open(sidecar_path, "rb") as fh:
            detached = DetachedTimestampFile.deserialize(
                StreamDeserializationContext(fh)
            )
        attestations = [att for _msg, att in detached.timestamp.all_attestations()]
    except Exception:
        return None
    if any(isinstance(a, BitcoinBlockHeaderAttestation) for a in attestations):
        return "confirmed"
    if any(isinstance(a, PendingAttestation) for a in attestations):
        return "pending"
    return "none"
