"""Response-side-channel features.

The observation: an attacker can learn from the **characteristics** of
the response, beyond its explicit content — **latency**, **output token count**,
**refusal phrasing/coherence**, and **whitespace/formatting artefacts** can reveal
retrieval success and model state (relevant to RAG membership inference). These are
honest, anti-circular OBSERVABLES of the response (no ground-truth label, no planted
keyword), used both as **detector features** and as the basis of an **innovative
domain-specific metric** (side-channel leakage).

This module provides per-response observables (whitespace/formatting) and the
session-level side-channel aggregates (latency variability, refusal rate, formatting
profile), merged into ``pile_detector.aggregate_sessions`` alongside the textual and
structural features.
"""
from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

_WS = re.compile(r"\s")
_NEWLINE_RUN = re.compile(r"\n{2,}")


def trailing_whitespace_len(text: str) -> int:
    t = text or ""
    return len(t) - len(t.rstrip())


def leading_whitespace_len(text: str) -> int:
    t = text or ""
    return len(t) - len(t.lstrip())


def whitespace_ratio(text: str) -> float:
    """Fraction of the response that is whitespace (a formatting-artefact signal)."""
    t = text or ""
    if not t:
        return 0.0
    return sum(1 for c in t if _WS.match(c)) / len(t)


def repeated_newline_runs(text: str) -> int:
    """Number of runs of >=2 consecutive newlines (formatting artefact)."""
    return len(_NEWLINE_RUN.findall(text or ""))


def response_sidechannel_features(response: str) -> dict[str, Any]:
    """Per-response observable side-channel features (added to the feature row)."""
    return {
        "feature_response_whitespace_ratio": round(whitespace_ratio(response), 6),
        "feature_response_trailing_whitespace": float(trailing_whitespace_len(response)),
        "feature_response_repeated_newline_runs": float(repeated_newline_runs(response)),
    }


def session_sidechannel_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Session-level side-channels from the per-request rows:
    **latency variability**, **refusal rate**, and **formatting** profile."""
    def _col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if isinstance(r.get(name), (int, float, bool))]

    lat = _col("feature_latency_total_ms")
    ws = _col("feature_response_whitespace_ratio")
    n = len(rows)
    refusals = sum(1 for r in rows if r.get("feature_response_contains_refusal"))
    return {
        "feature_session_std_latency": pstdev(lat) if len(lat) > 1 else 0.0,
        "feature_session_refusal_rate": refusals / n if n else 0.0,
        "feature_session_mean_whitespace_ratio": mean(ws) if ws else 0.0,
        "feature_session_max_whitespace_ratio": max(ws) if ws else 0.0,
    }


__all__ = [
    "trailing_whitespace_len",
    "leading_whitespace_len",
    "whitespace_ratio",
    "repeated_newline_runs",
    "response_sidechannel_features",
    "session_sidechannel_features",
]
