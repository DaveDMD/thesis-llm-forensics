"""Timeline reconstruction: correlate a session's request series into phases.

This module implements timeline reconstruction: correlating a series of queries
with possible attack phases. It groups forensic records by session, orders them
temporally, and
derives, per session, a reconstructed sequence of phases plus the attribution
signals (actor hashes, timing).

Inputs
------
Forensic records as produced by the pipeline (the Tier-1 stream). Per record it
uses ``session_id`` and ``ts_iso`` (top level), and from the payload the
``sequence_number``, ``endpoint`` and the attribution hashes (``ip_hash`` /
``user_agent_hash`` / ``asn_hash``). Optionally a detector output and/or feature
rows can be supplied to label which requests are flagged; without them the
reconstruction is purely structural.

Phase taxonomy (declared as an original contribution of the thesis)
-------------------------------------------------------------------
The reconstructed phases are a thesis-side, defensive-analysis schema, not a
standard kill-chain. They are intentionally coarse and observable from Tier-1
artefacts alone:

* ``reconnaissance``      — availability/citation/benign-looking probing;
* ``membership_probing``  — defined but currently NOT assignable: its candidate
  triggers (the keyword Yes/No probe features and the ``expose_logprobs`` flag)
  would be circular, so they are not used; the phase is left to be rederived
  from real residuals. The constant is kept for the taxonomy and downstream
  consumers.
* ``extraction_attempt``  — response artefact: secret-like leakage in the output;
* ``injection_attempt``   — indirect-injection artefact: payload echo in response;
* ``benign``              — no attack-indicative artefact.

Phase assignment here is rule-light and explainable; it consumes only honest
observable columns (response-side leakage/echo and the structural retrieval
surface), never the removed keyword features, the score-exposure flag, or any
ground-truth label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

# Phase labels (thesis-original taxonomy).
PHASE_RECON = "reconnaissance"
PHASE_MEMBERSHIP = "membership_probing"
PHASE_EXTRACTION = "extraction_attempt"
PHASE_INJECTION = "injection_attempt"
PHASE_BENIGN = "benign"

PHASE_ORDER = (
    PHASE_RECON,
    PHASE_MEMBERSHIP,
    PHASE_EXTRACTION,
    PHASE_INJECTION,
)


def _parse_ts(ts_iso: str | None) -> datetime | None:
    if not ts_iso:
        return None
    try:
        return datetime.fromisoformat(ts_iso)
    except ValueError:
        return None


@dataclass
class TimelineEvent:
    session_id: str
    sequence_number: int
    ts_iso: str | None
    endpoint: str
    phase: str
    ip_hash: str | None
    user_agent_hash: str | None
    asn_hash: str | None
    expose_logprobs: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "ts_iso": self.ts_iso,
            "endpoint": self.endpoint,
            "phase": self.phase,
            "ip_hash": self.ip_hash,
            "user_agent_hash": self.user_agent_hash,
            "asn_hash": self.asn_hash,
            "expose_logprobs": self.expose_logprobs,
        }


@dataclass
class SessionTimeline:
    session_id: str
    events: list[TimelineEvent] = field(default_factory=list)

    @property
    def phases_observed(self) -> list[str]:
        seen: list[str] = []
        for ev in self.events:
            if ev.phase != PHASE_BENIGN and ev.phase not in seen:
                seen.append(ev.phase)
        return seen

    @property
    def is_multi_phase(self) -> bool:
        return len(self.phases_observed) >= 2

    @property
    def actor_hashes(self) -> dict[str, set[str]]:
        return {
            "ip_hash": {e.ip_hash for e in self.events if e.ip_hash},
            "user_agent_hash": {e.user_agent_hash for e in self.events if e.user_agent_hash},
            "asn_hash": {e.asn_hash for e in self.events if e.asn_hash},
        }

    @property
    def duration_seconds(self) -> float | None:
        ts = [_parse_ts(e.ts_iso) for e in self.events]
        ts = [t for t in ts if t is not None]
        if len(ts) < 2:
            return None
        return (max(ts) - min(ts)).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "n_events": len(self.events),
            "phases_observed": self.phases_observed,
            "is_multi_phase": self.is_multi_phase,
            "duration_seconds": self.duration_seconds,
            "actor_consistent": {
                k: (len(v) <= 1) for k, v in self.actor_hashes.items()
            },
            "events": [e.as_dict() for e in self.events],
        }


def _phase_from_features(features: Mapping[str, Any] | None, *, endpoint: str) -> str:
    """Assign a phase from observable feature columns (no ground-truth labels).

    Precedence (most specific first): injection echo -> extraction artefacts ->
    reconnaissance -> benign.

    This heuristic deliberately does NOT key on the prompt-side keyword features
    or on the ``expose_logprobs`` flag: both would be circular — the keyword
    features would match planted attack phrases and the flag would track attacks
    only. What it uses instead are honest observables: the response-side
    injection echo and secret-like leakage, and the structural reconnaissance
    surface (RAG retrieval).

    Consequence: the ``membership_probing`` phase has no honest observable
    once the keyword probes and the score-exposure flag are excluded, so it is
    not assigned here. Membership-phase detection is left to be rebuilt on the
    real residuals derived after the launches.
    """
    f = features or {}
    if f.get("diagnostic_response_echoes_injection_payload"):
        return PHASE_INJECTION
    if f.get("feature_response_contains_secret_like_pattern"):
        return PHASE_EXTRACTION
    if f.get("feature_has_rag_retrieval") or endpoint == "/v1/rag":
        # benign-looking retrieval/availability traffic, the recon surface
        return PHASE_RECON
    return PHASE_BENIGN


def reconstruct_timelines(
    forensic_records: Sequence[Mapping[str, Any]],
    *,
    feature_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, SessionTimeline]:
    """Reconstruct one timeline per session from the forensic stream.

    Uses the ``prompt`` events (one per request) as the timeline anchor and
    joins optional feature rows by ``(session_id, sequence_number)`` to derive
    phases. Records are ordered by sequence_number, then by ts_iso as tie-break.
    """
    # Feature rows are joined by (session_id, sequence_number) — case_id is no
    # longer in the forensic stream. The session_id is pseudonymized consistently
    # across forensic records and feature rows, so the key matches.
    feat_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    if feature_rows:
        for row in feature_rows:
            sid = row.get("session_id")
            seq = row.get("sequence_number")
            if sid is not None and seq is not None:
                feat_by_key[(sid, int(seq))] = row

    timelines: dict[str, SessionTimeline] = {}
    for rec in forensic_records:
        if rec.get("event_type") != "prompt":
            continue
        payload = rec.get("payload", {})
        session_id = rec.get("session_id") or payload.get("session_id") or "unknown"
        features = feat_by_key.get((session_id, int(payload.get("sequence_number", 0))))
        endpoint = payload.get("endpoint", "")
        phase = _phase_from_features(features, endpoint=endpoint)

        # expose_logprobs is not written to the prompt payload (logprobs are
        # logged uniformly for all traffic), so it is not a meaningful per-request
        # attribute. The field is kept on the event for serialization stability
        # but is always False.
        event = TimelineEvent(
            session_id=session_id,
            sequence_number=int(payload.get("sequence_number", 0)),
            ts_iso=rec.get("ts_iso"),
            endpoint=endpoint,
            phase=phase,
            ip_hash=payload.get("ip_hash"),
            user_agent_hash=payload.get("user_agent_hash"),
            asn_hash=payload.get("asn_hash"),
            expose_logprobs=False,
        )
        timelines.setdefault(session_id, SessionTimeline(session_id=session_id)).events.append(event)

    for tl in timelines.values():
        tl.events.sort(key=lambda e: (e.sequence_number, e.ts_iso or ""))
    return timelines


def multi_phase_sessions(timelines: Mapping[str, SessionTimeline]) -> list[str]:
    """Session ids whose reconstructed timeline spans >= 2 distinct attack phases."""
    return sorted(sid for sid, tl in timelines.items() if tl.is_multi_phase)


__all__ = [
    "TimelineEvent",
    "SessionTimeline",
    "reconstruct_timelines",
    "multi_phase_sessions",
    "PHASE_RECON",
    "PHASE_MEMBERSHIP",
    "PHASE_EXTRACTION",
    "PHASE_INJECTION",
    "PHASE_BENIGN",
    "PHASE_ORDER",
]
