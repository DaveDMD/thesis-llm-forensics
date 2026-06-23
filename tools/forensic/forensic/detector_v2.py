"""Detector v2 — Passata 2 (defender-specialised instance).

A SECOND INSTANCE of the session detector. v1 (``forensic.pile_detector`` /
``.features`` / ``.mia_pile``) is left byte-for-byte intact as the frozen
Passata-1 reference; v2 lives here and differs in exactly ONE behaviour: it
recognises two enterprise secret FORMATS that v1's catalogue misses — a 16-digit
PAN and a ``CONF-XXXX-XXXX-XXXX`` access code. The scoring math
(``fit_session_scorer`` / ``posthoc_detect`` / GroupKFold) is reused UNCHANGED
from v1, so a v1-vs-v2 comparison isolates exactly the value of the defender
knowing the shape of its own secrets.

Anti-circularity: v2 injects only the FORM (regex), never a secret value nor any
label; the FP cost of the wider recogniser is reported as the honest
counterweight (a recogniser that flags everything would "win" detection while
flooding false alarms — that trade-off is what keeps v2 honest).
"""
from __future__ import annotations

import re

# Reused UNCHANGED from v1 — the shared scoring math and the v1 recognisers that
# v2 only EXTENDS on top of. Nothing here is edited for Passata 2.
from .features import REDACTION_PLACEHOLDER
from .features import contains_secret_like_pattern as _contains_secret_like_v1
from .features import redact_secret_like as _redact_secret_like_v1
from .mia_pile import secret_spans as _secret_spans_v1
from .online_detector import (  # re-exported so "v2 detector" is one import surface
    detection_metrics,
    fit_session_scorer,
    posthoc_detect,
    split_sessions,
)
from .pile_detector import aggregate_sessions as _aggregate_sessions_v1

__all__ = [
    "SECRET_LIKE_V2_PATTERNS",
    "contains_secret_like_v2",
    "secret_spans_v2",
    "redact_secret_like_v2",
    "aggregate_sessions",
    "fit_session_scorer",
    "posthoc_detect",
    "split_sessions",
    "detection_metrics",
]

# ── v2 EXTENDED recognition: the ONLY behavioural delta vs v1 ─────────────────
# The two enterprise formats v1 misses (organic blind spot, see canary_dataset
# RICH_KINDS). Kept deliberately specific to bound the FP cost: the PAN ``\d{16}``
# is the looser of the two, so its false-alarm rate is measured, not assumed away.
_PAN_RE = re.compile(r"\b\d{16}\b")
_CONFCODE_RE = re.compile(r"\bCONF-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b")
SECRET_LIKE_V2_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan_16", _PAN_RE),
    ("confcode", _CONFCODE_RE),
)


def contains_secret_like_v2(raw_text: str) -> bool:
    """v1 recogniser OR the two extended enterprise formats."""
    if _contains_secret_like_v1(raw_text):
        return True
    text = raw_text or ""
    return bool(_PAN_RE.search(text) or _CONFCODE_RE.search(text))


def secret_spans_v2(text: str) -> list[tuple[int, int, str]]:
    """v1 secret catalogue PLUS the two extended formats (used by the defence)."""
    spans = list(_secret_spans_v1(text))
    for kind, rx in SECRET_LIKE_V2_PATTERNS:
        for m in rx.finditer(text or ""):
            spans.append((m.start(), m.end(), kind))
    spans.sort(key=lambda s: (s[0], s[1]))
    return spans


def redact_secret_like_v2(
    text: str, placeholder: str = REDACTION_PLACEHOLDER
) -> tuple[str, int]:
    """v1 redaction union, then the extended formats. Returns ``(redacted, n)``.

    The placeholder is itself neither PAN- nor CONF-shaped, so the extended pass
    never re-matches an already-redacted span (no double counting).
    """
    if not text:
        return text, 0
    redacted, n = _redact_secret_like_v1(text, placeholder)
    for _kind, rx in SECRET_LIKE_V2_PATTERNS:
        redacted, k = rx.subn(placeholder, redacted)
        n += k
    return redacted, n


def aggregate_sessions(
    feature_rows: list, *, response_key: str = "feature_response_raw"
) -> list[dict]:
    """Session aggregation with v2 secret recognition.

    Identical to v1 EXCEPT ``feature_session_secret_like_rate`` is recomputed with
    ``contains_secret_like_v2`` over the raw response (``response_key``) when it is
    present on the per-request row. When no raw response is available the row's
    existing v1 boolean is left untouched — v2 never silently diverges from v1
    without the text that justifies the difference. All other session features and
    the schema are produced by the unchanged v1 aggregator.
    """
    patched = []
    for r in feature_rows:
        raw = r.get(response_key)
        if raw is not None:
            r = {
                **r,
                "feature_response_contains_secret_like_pattern": contains_secret_like_v2(raw),
            }
        patched.append(r)
    return _aggregate_sessions_v1(patched)
