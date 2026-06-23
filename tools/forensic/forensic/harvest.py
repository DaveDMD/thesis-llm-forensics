"""Persistence + dedup for the attack-harvest campaigns.

The harvest runs many attack campaigns to (a) extract as many Pile secrets and
infer as many members as possible, and (b) accumulate the residues they leave
into a reusable pool for later (generalising) detector training. This module
holds the small, pure, testable pieces: a normalized dedup key for secrets, a
deduplicator, and append/load helpers for the JSONL ledgers/pools.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

_WS = re.compile(r"\s+")


def normalized_secret(value: str) -> str:
    """Collapse whitespace + lowercase, so trivial variants dedup together."""
    return _WS.sub(" ", (value or "")).strip().lower()


def secret_key(kind: str, value: str) -> tuple[str, str]:
    """Dedup key for an extracted secret: (kind, normalized value)."""
    return (kind or "", normalized_secret(value))


def dedup_secrets(secrets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate secrets by (kind, normalized value), keeping first seen."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for s in secrets:
        k = secret_key(s.get("kind", ""), s.get("value", ""))
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows as JSON lines (creating the file/dir if needed). Returns count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file (empty list if missing)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


__all__ = [
    "normalized_secret",
    "secret_key",
    "dedup_secrets",
    "append_jsonl",
    "load_jsonl",
]
