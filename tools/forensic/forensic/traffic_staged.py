"""Multi-turn staged traffic plan: low-and-slow escalation campaigns.

Contribution of the thesis (declared as such, not as a standard). This builder
composes the *existing* single-request scenarios---harvested from the public
plan builders of :mod:`forensic.traffic`, :mod:`forensic.traffic_mia` and
:mod:`forensic.traffic_rag`---into multi-request sessions that

1. open with benign-appearing reconnaissance and escalate, under a single actor
   (shared ``session_id`` and ``user_id``), through membership probing to
   extraction or indirect injection, so that the first detector-flagged request
   occurs at a position > 1 and time-to-detection becomes informative; and
2. reuse the same actor (``user_id``) across several distinct sessions, so that
   cross-session attribution has genuine identity reuse to recover.

It adds new *sequencing* of existing attack semantics, not new attack semantics,
consistently with the dataset-construction posture of the scaled plan
(``traffic_volume.scale_plan``). It is a separate, additive demonstration set:
it does not modify or extend the canonical integrated dataset.

Benign multi-turn sessions are included as negatives so that "multi-request
session" does not coincide with "attack" (which would re-introduce trivial
separability at the session level).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from forensic.traffic import TrafficCase, _mock_hash, build_m3_traffic_plan
from forensic.traffic_mia import build_mia_probing_plan
from forensic.traffic_rag import build_rag_attack_plan

# --- Session templates (deterministic) -------------------------------------
# Each template is (session_id, user_id, [scenario, ...]). The scenarios are
# played in order with sequence_number 1..k under the one actor. Reused user_ids
# (A1, A2, A3) span more than one session to exercise cross-session attribution.

_RECON = "benign_rag_availability_query"
_CITE = "benign_rag_citation"
_RECALL = "benign_recall_request"
_SCORE = "mia_score_probing"
_DSTORE = "rag_datastore_probing"
# The F-A secret-extraction scenario uses the camouflaged system-prompt form;
# the staged campaigns escalate to that scenario (its first representative case
# is picked by ``_source_index``).
_SECRET = "secret_extraction_system_prompt"
_INJECT = "rag_indirect_prompt_injection"
_BENIGN_COMPLETE = "benign_complete"

ATTACK_TEMPLATES: list[tuple[str, str, list[str]]] = [
    ("staged-atk-01", "staged-actor-A1", [_RECON, _CITE, _SCORE, _SECRET]),          # first attack @3
    ("staged-atk-02", "staged-actor-A1", [_RECON, _INJECT]),                          # @2
    ("staged-atk-03", "staged-actor-A2", [_RECON, _CITE, _RECALL, _DSTORE, _INJECT]), # @4
    ("staged-atk-04", "staged-actor-A2", [_RECON, _CITE, _CITE, _SECRET]),            # @4
    ("staged-atk-05", "staged-actor-A3", [_SCORE, _SECRET]),                          # @1 (no recon)
    ("staged-atk-06", "staged-actor-A3", [_RECON, _CITE, _RECALL, _RECON, _DSTORE, _INJECT]),  # @5
    ("staged-atk-07", "staged-actor-A4", [_RECON, _CITE, _SCORE, _INJECT]),           # @3
    ("staged-atk-08", "staged-actor-A5", [_RECON, _SECRET]),                          # @2
    ("staged-atk-09", "staged-actor-A6", [_RECON, _CITE, _DSTORE]),                   # @3
    ("staged-atk-10", "staged-actor-A7", [_RECON, _CITE, _RECALL, _SCORE, _SECRET]),  # @4
]

BENIGN_TEMPLATES: list[tuple[str, str, list[str]]] = [
    (f"staged-ben-{i:02d}", f"staged-benign-B{i}",
     [_RECON, _CITE, _RECALL, _BENIGN_COMPLETE][: 3 + (i % 2)])
    for i in range(1, 9)
]


def _source_index() -> dict[str, TrafficCase]:
    """First representative case of each scenario, across all public builders."""
    plan: list[TrafficCase] = []
    plan += build_m3_traffic_plan()
    plan += build_rag_attack_plan()
    plan += build_mia_probing_plan()
    idx: dict[str, TrafficCase] = {}
    for case in plan:
        idx.setdefault(case.groundtruth.get("scenario", ""), case)
    return idx


def _restitch(
    source: TrafficCase, *, session_id: str, user_id: str, sequence_number: int, case_id: str
) -> TrafficCase:
    """Copy a source case into a staged session under a given actor and position."""
    body = deepcopy(source.body)
    body["session_id"] = session_id
    body["user_id"] = user_id
    body["sequence_number"] = sequence_number
    body["ip_hash"] = _mock_hash("ip", user_id)
    body["user_agent_hash"] = _mock_hash("user_agent", user_id)
    body["asn_hash"] = _mock_hash("asn", user_id)
    rm = dict(body.get("request_metadata") or {})
    rm["case_id"] = case_id
    body["request_metadata"] = rm
    gt = deepcopy(source.groundtruth)
    gt["case_id"] = case_id
    gt["session_id"] = session_id
    gt["sequence_number"] = sequence_number
    return replace(source, case_id=case_id, body=body, groundtruth=gt)


def _expand(templates: list[tuple[str, str, list[str]]], idx: dict[str, TrafficCase]) -> list[TrafficCase]:
    out: list[TrafficCase] = []
    for session_id, user_id, scenarios in templates:
        for seq, scenario in enumerate(scenarios, start=1):
            if scenario not in idx:
                raise KeyError(
                    f"scenario {scenario!r} not produced by any public builder; "
                    f"available: {sorted(idx)}"
                )
            out.append(_restitch(
                idx[scenario],
                session_id=session_id,
                user_id=user_id,
                sequence_number=seq,
                case_id=f"{session_id}-s{seq}",
            ))
    return out


def build_staged_plan() -> list[TrafficCase]:
    """Return the deterministic multi-turn staged plan (attack + benign sessions)."""
    idx = _source_index()
    plan = _expand(ATTACK_TEMPLATES, idx) + _expand(BENIGN_TEMPLATES, idx)
    # Defensive: the offline feature join relies on a unique (session, seq, endpoint).
    seen: set[tuple[str, int, str]] = set()
    for case in plan:
        key = (case.body["session_id"], int(case.body["sequence_number"]), case.endpoint)
        if key in seen:
            raise ValueError(f"join-key collision in staged plan: {key} (case {case.case_id})")
        seen.add(key)
    return plan


__all__ = ["build_staged_plan", "ATTACK_TEMPLATES", "BENIGN_TEMPLATES"]
