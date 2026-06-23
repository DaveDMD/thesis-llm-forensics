"""Deterministic serialisation and hashing primitives.

Canonical JSON follows the same rules used by the chain:
    * keys sorted lexicographically
    * compact separators (',' between items and ':' between key and value)
    * UTF-8 output, non-ASCII characters preserved (ensure_ascii=False)
    * NaN/Infinity rejected (JSON does not represent them)

Two records that are semantically identical MUST serialise to the same byte
string. This is the single property the entire chain depends on.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from typing import Any, Mapping


def _reject_nonfinite(obj: Any) -> Any:
    """Reject NaN/Infinity recursively (json.dumps would otherwise emit them)."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(
                f"Non-finite float {obj!r} cannot appear in a forensic record"
            )
        return obj
    if isinstance(obj, dict):
        return {k: _reject_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_reject_nonfinite(v) for v in obj]
    return obj


def canonical_json(obj: Mapping[str, Any]) -> bytes:
    """Serialise *obj* to canonical JSON bytes used for hashing."""
    safe = _reject_nonfinite(dict(obj))
    return json.dumps(
        safe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_record(record: Mapping[str, Any]) -> str:
    """SHA-256 of canonical(record \\ {record_hash})."""
    stripped = {k: v for k, v in record.items() if k != "record_hash"}
    return sha256_hex(canonical_json(stripped))


def pseudonymize(user_id: str, salt: bytes) -> str:
    """HMAC-SHA256 keyed pseudonym for a user identifier.

    The *salt* must be kept confidential (never logged in clear); only its
    SHA-256 fingerprint is recorded inside the manifest. Re-identification
    requires holding the salt.
    """
    if not isinstance(salt, (bytes, bytearray)) or len(salt) < 16:
        raise ValueError("salt must be ≥ 16 raw bytes")
    return hmac.new(salt, user_id.encode("utf-8"), hashlib.sha256).hexdigest()


def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file from disk."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
