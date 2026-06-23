"""Production countermeasures: an evaluable policy engine.

Implements policy/countermeasures to limit exfiltration in production. The
countermeasures are expressed as a *policy engine* that
evaluates the already-logged traffic and the detector/feature signals, and
emits a motivated decision per request and per session
(allow / throttle / suppress_logprobs / redact_output / block_session), with
the triggering signal recorded for explainability.

Scope and boundary (declared, consistent with the project framing)
------------------------------------------------------------------
This is an *evaluable and demonstrable* policy layer over the forensic data, not
an in-line enforcing WAF. It shows that the artefacts the pipeline detects
*enable* concrete countermeasures and quantifies how many attack requests each
policy would have affected. Wiring these policies into the live request path
(actual throttling/blocking at serving time) is an architectural extension
intentionally out of scope of the
logging prototype. Each countermeasure pairs a technical control with an
organisational note.

Policies
--------
* rate_limiting        — per-actor request rate over a window; throttle above a
                         threshold (covers low-and-slow probing that concentrates
                         on one pseudonymous actor).
* logprob_suppression  — deny score exposure on requests whose
                         pattern is score-based probing.
* output_filtering     — redact responses carrying secret-like patterns or an
                         injection-payload echo (reuses the response-side
                         features).
* session_circuit_breaker — recommend blocking a session that accumulates strong
                         attack-phase signals.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

# Decision vocabulary.
ACTION_ALLOW = "allow"
ACTION_THROTTLE = "throttle"
ACTION_SUPPRESS_LOGPROBS = "suppress_logprobs"
ACTION_REDACT_OUTPUT = "redact_output"
ACTION_BLOCK_SESSION = "block_session"

# Default thresholds (thesis-defined, documented).
DEFAULT_RATE_LIMIT_PER_WINDOW = 10
DEFAULT_RATE_WINDOW_SECONDS = 60.0
DEFAULT_BLOCK_STRONG_SIGNAL_COUNT = 2

# Organisational note paired with each technical control (for the thesis text).
ORGANISATIONAL_NOTES = {
    ACTION_THROTTLE: "Define per-tenant rate quotas and an appeals path; log throttling for audit.",
    ACTION_SUPPRESS_LOGPROBS: "Disable score exposure by default; allow it only for vetted internal use.",
    ACTION_REDACT_OUTPUT: "Define a data-classification policy and a redaction review workflow.",
    ACTION_BLOCK_SESSION: "Define who authorises session blocking and the user-notification duty (GDPR).",
}


def _parse_ts(ts_iso: str | None) -> datetime | None:
    if not ts_iso:
        return None
    try:
        return datetime.fromisoformat(ts_iso)
    except ValueError:
        return None


@dataclass(frozen=True)
class CountermeasureDecision:
    case_id: str | None
    session_id: str
    action: str
    policy: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "session_id": self.session_id,
            "action": self.action,
            "policy": self.policy,
            "reason": self.reason,
        }


# ── per-request policies ─────────────────────────────────────────────────────


def evaluate_logprob_suppression(feature_row: Mapping[str, Any]) -> CountermeasureDecision | None:
    """Score-exposure suppression policy — NEUTRALIZED.

    This policy would deny score exposure on requests that "ask for logprobs",
    triggering on ``feature_expose_logprobs`` / ``expose_logprobs``. Since that
    flag, set only on attack traffic, would be a disguised label — and the
    policy's whole premise ("clients rarely ask for scores, so asking is
    suspicious") collapses once logprobs are a uniform, legitimate observable
    logged for ALL traffic — there is no honest per-request signal left to trigger
    on, so the policy is neutralized (always returns None).

    Whether to remove this policy outright or to reframe it as a descriptive
    control is left as future work. The function and the
    ``ACTION_SUPPRESS_LOGPROBS`` vocabulary are kept for API stability until then.
    """
    return None


def evaluate_output_filtering(feature_row: Mapping[str, Any]) -> CountermeasureDecision | None:
    """Redact responses that leak secret-like patterns or echo an injection payload."""
    leak = bool(feature_row.get("feature_response_contains_secret_like_pattern"))
    echo = bool(feature_row.get("diagnostic_response_echoes_injection_payload"))
    if leak or echo:
        reason = "secret-like pattern in response" if leak else "response echoes injection payload"
        return CountermeasureDecision(
            case_id=feature_row.get("case_id"),
            session_id=str(feature_row.get("session_id")),
            action=ACTION_REDACT_OUTPUT,
            policy="output_filtering",
            reason=reason,
        )
    return None


# ── per-actor / per-session policies ─────────────────────────────────────────


def evaluate_rate_limiting(
    forensic_records: Sequence[Mapping[str, Any]],
    *,
    limit_per_window: int = DEFAULT_RATE_LIMIT_PER_WINDOW,
    window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
    actor_key: str = "ip_hash",
) -> list[CountermeasureDecision]:
    """Throttle actors whose request rate exceeds the window threshold.

    Groups prompt events by actor hash, and within each actor scans a sliding
    time window; actors exceeding ``limit_per_window`` requests in any window are
    recommended for throttling. Robust to missing timestamps (skips them).
    """
    events_by_actor: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for rec in forensic_records:
        if rec.get("event_type") != "prompt":
            continue
        payload = rec.get("payload", {})
        actor = payload.get(actor_key)
        ts = _parse_ts(rec.get("ts_iso"))
        sid = rec.get("session_id") or "unknown"
        if actor is None or ts is None:
            continue
        events_by_actor[actor].append((ts, sid))

    decisions: list[CountermeasureDecision] = []
    for actor, events in events_by_actor.items():
        events.sort(key=lambda e: e[0])
        # sliding window count
        start = 0
        flagged = False
        for end in range(len(events)):
            while (events[end][0] - events[start][0]).total_seconds() > window_seconds:
                start += 1
            count = end - start + 1
            if count > limit_per_window:
                flagged = True
                break
        if flagged:
            decisions.append(
                CountermeasureDecision(
                    case_id=None,
                    session_id=events[0][1],
                    action=ACTION_THROTTLE,
                    policy="rate_limiting",
                    reason=(
                        f"actor exceeded {limit_per_window} requests in "
                        f"{int(window_seconds)}s window"
                    ),
                )
            )
    return decisions


def evaluate_session_circuit_breaker(
    triage_result: Mapping[str, Any],
    *,
    block_if_multi_phase: bool = True,
) -> list[CountermeasureDecision]:
    """Recommend blocking sessions with strong/multi-phase attack signals.

    Consumes the playbook triage output (so the breaker is detector-/triage-
    triggered). Multi-phase attack sessions are recommended
    for blocking; single strong-phase sessions are left to throttling/redaction.
    """
    decisions: list[CountermeasureDecision] = []
    targets = triage_result.get("multi_phase_sessions", []) if block_if_multi_phase else []
    for sid in targets:
        decisions.append(
            CountermeasureDecision(
                case_id=None,
                session_id=sid,
                action=ACTION_BLOCK_SESSION,
                policy="session_circuit_breaker",
                reason="session spans multiple strong attack phases",
            )
        )
    return decisions


# ── orchestration + impact report ────────────────────────────────────────────


def evaluate_countermeasures(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    forensic_records: Sequence[Mapping[str, Any]],
    triage_result: Mapping[str, Any] | None = None,
    rate_limit_per_window: int = DEFAULT_RATE_LIMIT_PER_WINDOW,
    rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Evaluate all countermeasure policies and return decisions + an impact report.

    The impact report quantifies how many requests/sessions each policy would
    affect and, using the ground-truth labels in the feature rows (analysis-only,
    never sent to the server), how many of the affected requests are actually
    attacks — a precision-of-countermeasure view for the discussion.
    """
    per_request: list[CountermeasureDecision] = []
    for row in feature_rows:
        for evaluator in (evaluate_logprob_suppression, evaluate_output_filtering):
            decision = evaluator(row)
            if decision is not None:
                per_request.append(decision)

    rate_decisions = evaluate_rate_limiting(
        forensic_records,
        limit_per_window=rate_limit_per_window,
        window_seconds=rate_window_seconds,
    )
    breaker_decisions = (
        evaluate_session_circuit_breaker(triage_result) if triage_result else []
    )

    all_decisions = per_request + rate_decisions + breaker_decisions

    # Impact: per policy, count decisions and (label-based) attack hit-rate.
    label_by_case = {r.get("case_id"): bool(r.get("label_is_attack")) for r in feature_rows}
    by_policy: dict[str, dict[str, Any]] = {}
    for d in all_decisions:
        slot = by_policy.setdefault(
            d.policy, {"action": d.action, "n_decisions": 0, "n_on_attack": 0, "n_with_label": 0}
        )
        slot["n_decisions"] += 1
        if d.case_id in label_by_case:
            slot["n_with_label"] += 1
            if label_by_case[d.case_id]:
                slot["n_on_attack"] += 1
    for slot in by_policy.values():
        slot["attack_hit_rate"] = (
            round(slot["n_on_attack"] / slot["n_with_label"], 6) if slot["n_with_label"] else None
        )
        slot["organisational_note"] = ORGANISATIONAL_NOTES.get(slot["action"], "")

    return {
        "decisions": [d.as_dict() for d in all_decisions],
        "n_decisions": len(all_decisions),
        "by_policy": by_policy,
        "boundary_note": (
            "Evaluable policy layer over forensic data; not an in-line enforcing "
            "WAF. In-line enforcement at serving time is an architectural "
            "extension out of scope of this prototype."
        ),
    }


__all__ = [
    "CountermeasureDecision",
    "evaluate_logprob_suppression",
    "evaluate_output_filtering",
    "evaluate_rate_limiting",
    "evaluate_session_circuit_breaker",
    "evaluate_countermeasures",
    "ACTION_ALLOW",
    "ACTION_THROTTLE",
    "ACTION_SUPPRESS_LOGPROBS",
    "ACTION_REDACT_OUTPUT",
    "ACTION_BLOCK_SESSION",
]
