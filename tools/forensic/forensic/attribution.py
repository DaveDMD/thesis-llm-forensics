"""Attribution heuristics: prudential correlation of sessions to a likely actor.

Implements attribution heuristics (user-agent patterns, IP/ASN hash, timing,
query fingerprints). Crucially, these are
*correlation* signals over pseudonymous hashes, NOT identification: the output
is a confidence-scored hypothesis that two sessions share an actor, with the
contributing signals listed for explainability. These are explicitly
**prudential, non-standardised heuristics** — never certain
attribution — consistent with the project's epistemic caution and with the fact
that we operate on pseudonymised hashes, not real identities.

Signals (all observable from Tier-1 artefacts)
----------------------------------------------
* shared ``asn_hash`` / ``ip_hash`` / ``user_agent_hash`` across sessions;
* prompt-fingerprint overlap (Jaccard over the sessions' sets of ``prompt_hash``);
* behavioural pattern similarity (e.g. both sessions score-exposing, same
  endpoint mix);
* temporal proximity (sessions close in time).

Each present signal contributes a documented weight to a confidence score in
[0, 1]; the score is a heuristic ranking aid, not a probability of identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

# Documented signal weights (thesis-defined; sum need not be 1 — score is
# clamped). Shared network/agent identifiers weigh most; fingerprint overlap and
# behaviour add corroboration; temporal proximity is the weakest signal.
SIGNAL_WEIGHTS = {
    "shared_asn_hash": 0.30,
    "shared_ip_hash": 0.35,
    "shared_user_agent_hash": 0.15,
    "prompt_fingerprint_overlap": 0.25,
    "behavioural_pattern_match": 0.15,
    "temporal_proximity": 0.10,
}

# Defaults for thresholds (thesis-defined, documented).
DEFAULT_FINGERPRINT_OVERLAP_MIN = 0.2
DEFAULT_TEMPORAL_PROXIMITY_SECONDS = 300.0


def _parse_ts(ts_iso: str | None) -> datetime | None:
    if not ts_iso:
        return None
    try:
        return datetime.fromisoformat(ts_iso)
    except ValueError:
        return None


@dataclass
class SessionProfile:
    """Per-session attribution-relevant profile, derived from prompt events."""

    session_id: str
    ip_hashes: set[str] = field(default_factory=set)
    asn_hashes: set[str] = field(default_factory=set)
    user_agent_hashes: set[str] = field(default_factory=set)
    prompt_hashes: set[str] = field(default_factory=set)
    endpoints: set[str] = field(default_factory=set)
    any_score_exposing: bool = False
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    n_requests: int = 0


def build_session_profiles(
    forensic_records: Sequence[Mapping[str, Any]],
) -> dict[str, SessionProfile]:
    """Build one attribution profile per session from the prompt events."""
    profiles: dict[str, SessionProfile] = {}
    for rec in forensic_records:
        if rec.get("event_type") != "prompt":
            continue
        payload = rec.get("payload", {})
        sid = rec.get("session_id") or "unknown"
        prof = profiles.setdefault(sid, SessionProfile(session_id=sid))
        if payload.get("ip_hash"):
            prof.ip_hashes.add(payload["ip_hash"])
        if payload.get("asn_hash"):
            prof.asn_hashes.add(payload["asn_hash"])
        if payload.get("user_agent_hash"):
            prof.user_agent_hashes.add(payload["user_agent_hash"])
        if payload.get("prompt_hash"):
            prof.prompt_hashes.add(payload["prompt_hash"])
        if payload.get("endpoint"):
            prof.endpoints.add(payload["endpoint"])
        # ``any_score_exposing`` is not derived from ``expose_logprobs``: that
        # flag, set only on attacks, would be a disguised label and is not written
        # to the forensic stream. The ``behavioural_pattern_match`` signal that
        # consumed it therefore does not fire; an honest symmetric replacement
        # (e.g. similarity of the logged logprob statistics) would require
        # consuming the LOGPROBS events, which is left as future work.
        ts = _parse_ts(rec.get("ts_iso"))
        if ts is not None:
            prof.first_ts = ts if prof.first_ts is None else min(prof.first_ts, ts)
            prof.last_ts = ts if prof.last_ts is None else max(prof.last_ts, ts)
        prof.n_requests += 1
    return profiles


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _temporal_proximity(a: SessionProfile, b: SessionProfile, *, window_s: float) -> bool:
    if a.last_ts is None or b.first_ts is None or a.first_ts is None or b.last_ts is None:
        return False
    # gap between the two sessions' time spans
    if a.last_ts < b.first_ts:
        gap = (b.first_ts - a.last_ts).total_seconds()
    elif b.last_ts < a.first_ts:
        gap = (a.first_ts - b.last_ts).total_seconds()
    else:
        gap = 0.0  # overlapping in time
    return gap <= window_s


@dataclass(frozen=True)
class AttributionLink:
    session_a: str
    session_b: str
    confidence: float
    signals: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_a": self.session_a,
            "session_b": self.session_b,
            "confidence": self.confidence,
            "signals": self.signals,
            "note": "prudential correlation heuristic; not identity attribution",
        }


def score_session_pair(
    a: SessionProfile,
    b: SessionProfile,
    *,
    fingerprint_overlap_min: float = DEFAULT_FINGERPRINT_OVERLAP_MIN,
    temporal_window_s: float = DEFAULT_TEMPORAL_PROXIMITY_SECONDS,
) -> AttributionLink:
    """Score the hypothesis that two sessions share an actor (explainable)."""
    signals: list[str] = []
    score = 0.0

    if a.asn_hashes & b.asn_hashes:
        signals.append("shared_asn_hash")
        score += SIGNAL_WEIGHTS["shared_asn_hash"]
    if a.ip_hashes & b.ip_hashes:
        signals.append("shared_ip_hash")
        score += SIGNAL_WEIGHTS["shared_ip_hash"]
    if a.user_agent_hashes & b.user_agent_hashes:
        signals.append("shared_user_agent_hash")
        score += SIGNAL_WEIGHTS["shared_user_agent_hash"]

    fp = _jaccard(a.prompt_hashes, b.prompt_hashes)
    if fp >= fingerprint_overlap_min:
        signals.append("prompt_fingerprint_overlap")
        # weight scaled by the overlap magnitude
        score += SIGNAL_WEIGHTS["prompt_fingerprint_overlap"] * fp

    # behavioural pattern: same score-exposing behaviour AND same endpoint mix.
    # ``any_score_exposing`` is not populated (the score-exposure flag would be a
    # disguised label), so this signal is currently INERT. It is left in place —
    # rather than substituted with a weak proxy such as bare endpoint-mix
    # equality, which would fire on unrelated sessions — until an honest symmetric
    # behavioural observable is defined (future work). The weight key is kept for
    # that future signal.
    if a.any_score_exposing and b.any_score_exposing and a.endpoints == b.endpoints:
        signals.append("behavioural_pattern_match")
        score += SIGNAL_WEIGHTS["behavioural_pattern_match"]

    if _temporal_proximity(a, b, window_s=temporal_window_s):
        signals.append("temporal_proximity")
        score += SIGNAL_WEIGHTS["temporal_proximity"]

    return AttributionLink(
        session_a=a.session_id,
        session_b=b.session_id,
        confidence=round(min(1.0, score), 6),
        signals=signals,
    )


def correlate_sessions(
    forensic_records: Sequence[Mapping[str, Any]],
    *,
    min_confidence: float = 0.3,
    fingerprint_overlap_min: float = DEFAULT_FINGERPRINT_OVERLAP_MIN,
    temporal_window_s: float = DEFAULT_TEMPORAL_PROXIMITY_SECONDS,
) -> list[AttributionLink]:
    """Return session pairs whose correlation confidence meets ``min_confidence``.

    Pairs are unordered (session_a < session_b) and sorted by descending
    confidence. The result is a ranked list of *hypotheses*, each with its
    contributing signals; it must be reported as prudential, not as proof.
    """
    profiles = build_session_profiles(forensic_records)
    sids = sorted(profiles)
    links: list[AttributionLink] = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            link = score_session_pair(
                profiles[sids[i]],
                profiles[sids[j]],
                fingerprint_overlap_min=fingerprint_overlap_min,
                temporal_window_s=temporal_window_s,
            )
            if link.confidence >= min_confidence and link.signals:
                links.append(link)
    links.sort(key=lambda l: l.confidence, reverse=True)
    return links


__all__ = [
    "SignalWeights",
    "SIGNAL_WEIGHTS",
    "SessionProfile",
    "build_session_profiles",
    "AttributionLink",
    "score_session_pair",
    "correlate_sessions",
]


# Backwards-friendly alias (kept simple); not a dataclass.
SignalWeights = SIGNAL_WEIGHTS
