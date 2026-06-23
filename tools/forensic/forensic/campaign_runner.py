"""F-MT runner: execute multi-turn campaigns against the E0/E1 server.

Drives a :class:`~forensic.campaign.CampaignPlan` turn by turn
through the SAME FastAPI server used by every other request (``create_app``), so
each turn is a normal request logged in the forensic stream with a shared
``session_id`` and an increasing ``sequence_number``. The adaptive loop is:

    move = entry move
    while not terminated:
        POST the move to E0/E1
        if blocked (E1 defence): outcome = BLOCKED, no completion
        else: attacker EXACT-MATCH win-check on the response (hit-and-run)
              if recognised -> terminate (success recognised)
        outcome -> branch signal -> next move (press/pivot); None -> terminate (failed)

Symmetry: the SAME campaigns run in E0 and E1 (one code path). In E1 a
blocked turn logs a ``prompt`` with no ``completion`` (consistent with the
level-2 defences); the campaign continues/terminates by the branching logic and the block is a
PRESERVED residue, never discarded.

Two-stream separation: the forensic stream carries only the
per-turn requests/responses/defence-outcomes; the oracle's campaign label
(outcome, leak turn, divergence, length marker) is GROUNDTRUTH, returned
separately and never written to the log.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from .application import _FORBIDDEN_RETRIEVAL_CONTENT_KEYS, _reject_groundtruth_keys
from .campaign import (
    BranchConfig,
    CampaignLabel,
    CampaignPlan,
    CampaignState,
    Move,
    TurnRecord,
    attacker_recognizes_success,
    build_campaign_plans,
    choose_next_move,
    classify_outcome,
    label_campaign,
    observe_outcome,
)
from .schema import DefenseOutcome
from .secret_oracle import SecretRecord, build_secret_groundtruth
from .server import create_app
from .traffic import _mock_hash
from .verifier import EvidenceVerifier

DEFAULT_RUN_ID = "fmt-campaigns"
DEFAULT_SALT = b"fmt-campaigns-synthetic-salt-bytes!!"

# Technical cap: a purely defensive ceiling on turns, NOT a
# behavioural choice. Set HIGH — the plans terminate far sooner (plan exhaustion or
# recognised success). It only guards against an unforeseen branching cycle.
DEFAULT_CAP_MAX_TURNS = 32


@dataclass(frozen=True)
class CampaignRunResult:
    """In-memory outcome of running one or more campaigns in one environment."""

    environment: str
    forensic_records: list[dict[str, Any]]
    states: list[CampaignState]
    labels: list[CampaignLabel]
    summary: dict[str, Any]
    verification_ok: bool


def _target_values(
    plan: CampaignPlan, secrets: list[SecretRecord]
) -> list[str]:
    """Resolve the attacker's KNOWN target values (a SUBSET of the manifest).

    The attacker knows only the value(s) it hunts; the oracle uses the whole
    manifest. Keeping these distinct is what lets attacker and oracle diverge.
    """
    by_id = {s.secret_id: s.value for s in secrets}
    return [by_id[sid] for sid in plan.target_secret_ids if sid in by_id]


def _turn_body(
    plan: CampaignPlan, move: Move, *, sequence_number: int
) -> dict[str, Any]:
    """Build the HTTP request body for one campaign turn (no ground-truth keys)."""
    user_id = f"{plan.campaign_id}-actor"
    body: dict[str, Any] = {
        "session_id": plan.campaign_id,
        "user_id": user_id,
        "sequence_number": sequence_number,
        "actor_type": "external_user",
        "ip_hash": _mock_hash("ip", user_id),
        "user_agent_hash": _mock_hash("user_agent", user_id),
        "asn_hash": _mock_hash("asn", user_id),
        "request_metadata": {
            "simulator": "fmt_campaign_runner",
            "campaign_id": plan.campaign_id,
            "move_id": move.move_id,
            "client_surface": "local_fastapi_mock",
        },
        "sampling_params": {"temperature": 0.0},
        "prompt": move.prompt,
    }
    if move.endpoint == "/v1/rag":
        body.update({
            "retrieval_query": move.retrieval_query or move.prompt,
            "max_tokens": 128,
            "top_k": 3,
            "embedding_model_id": "deterministic-mock-embedding",
            "vector_store_id": "deterministic-mock-vector-store",
        })
    else:
        body["max_tokens"] = 64
    return body


def run_campaign(
    client: TestClient,
    plan: CampaignPlan,
    *,
    secrets: list[SecretRecord],
    config: BranchConfig | None = None,
    cap_max_turns: int = DEFAULT_CAP_MAX_TURNS,
) -> tuple[CampaignState, CampaignLabel]:
    """Execute one campaign turn-by-turn and return its state + oracle label."""
    config = config or BranchConfig()
    targets = _target_values(plan, secrets)

    state = CampaignState(
        campaign_id=plan.campaign_id,
        session_id=plan.campaign_id,
        user_id=f"{plan.campaign_id}-actor",
        family=plan.family,
        cadence=plan.cadence,
    )

    move: Move | None = plan.get(plan.entry_move)
    seq = 0
    while move is not None:
        seq += 1
        if seq > cap_max_turns:  # technical safety net only
            state.terminated = True
            state.termination_reason = "cap_reached"
            break

        resp = client.post(move.endpoint, json=_turn_body(plan, move, sequence_number=seq))

        if resp.status_code == 200:
            body = resp.json()
            response_raw = body.get("response")
            finish_reason = body.get("finish_reason")
            token_count = body.get("response_token_count")
            defense_outcome = DefenseOutcome.ACCEPTED.value
            outcome = observe_outcome(
                response_raw=response_raw,
                finish_reason=finish_reason,
                response_token_count=token_count,
                blocked=False,
                config=config,
            )
            recognised = attacker_recognizes_success(response_raw, targets)
        elif resp.status_code in (400, 429):
            # Blocked by an E1 level-2 defence: no completion. The block is a
            # residue, not a reason to drop the turn.
            detail = resp.json().get("detail", {})
            defense_outcome = (
                detail.get("defense_outcome")
                if isinstance(detail, dict)
                else None
            ) or "blocked"
            response_raw = None
            outcome = observe_outcome(
                response_raw=None, finish_reason=None, blocked=True, config=config
            )
            recognised = False
        else:  # pragma: no cover - unexpected server error
            resp.raise_for_status()
            raise RuntimeError(f"unexpected status {resp.status_code}")

        signal = classify_outcome(outcome, config=config)
        state.turns.append(TurnRecord(
            sequence_number=seq,
            move_id=move.move_id,
            scenario=move.scenario,
            endpoint=move.endpoint,
            prompt=move.prompt,
            response_raw=response_raw,
            structural_outcome=outcome,
            branch_signal=signal.value,
            defense_outcome=defense_outcome,
            attacker_recognized_success=recognised,
        ))

        if recognised:
            state.terminated = True
            state.termination_reason = "attacker_recognized_success"
            break

        nxt = choose_next_move(plan, move, signal)
        if nxt is None:
            state.terminated = True
            state.termination_reason = "plan_exhausted"
            break
        move = nxt

    label = label_campaign(state, secrets=secrets)
    return state, label


def _structural_anti_leak(records: list[dict[str, Any]]) -> None:
    for rec in records:
        _reject_groundtruth_keys(
            rec.get("payload", {}), path=f"{rec.get('event_type', 'record')}.payload"
        )
        for hit in (rec.get("payload", {}).get("retriever_hits") or []):
            forbidden = set(hit.keys()) & _FORBIDDEN_RETRIEVAL_CONTENT_KEYS
            if forbidden:
                raise ValueError(
                    f"raw retrieval content keys in forensic log: {sorted(forbidden)}"
                )


def run_campaigns(
    *,
    log_path: str,
    environment: str = "E0",
    salt: bytes = DEFAULT_SALT,
    run_id: str = DEFAULT_RUN_ID,
    repo_path: str = ".",
    plans: list[CampaignPlan] | None = None,
    secrets: list[SecretRecord] | None = None,
    backend: Any | None = None,
    retriever: Any | None = None,
    expose_logprobs: bool = True,
    config: BranchConfig | None = None,
    cap_max_turns: int = DEFAULT_CAP_MAX_TURNS,
    read_records: Callable[[str], list[dict[str, Any]]] | None = None,
    verify: bool = True,
) -> CampaignRunResult:
    """Run all F-MT campaigns in ONE environment (E0 or E1) and return the dataset.

    The SAME plans are sent to whichever environment is requested (symmetry: call
    twice, once per environment). Defaults to the deterministic mocks; a real
    backend/retriever can be injected for model-dependent runs.
    """
    import json

    plans = plans if plans is not None else build_campaign_plans()
    secrets = list(secrets if secrets is not None else build_secret_groundtruth())

    if read_records is None:
        def read_records(path: str) -> list[dict[str, Any]]:  # type: ignore[misc]
            return [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    app = create_app(
        log_path=log_path,
        salt=salt,
        run_id=run_id,
        experiment_phase="fmt_campaigns",
        repo_path=repo_path,
        experiment_config={
            "mock_mode": backend is None and retriever is None,
            "simulation": "fmt_multi_turn_campaigns",
            "groundtruth_separate": True,
            "plans": [p.campaign_id for p in plans],
        },
        backend=backend,
        retriever=retriever,
        environment=environment,
        expose_logprobs=expose_logprobs,
    )

    states: list[CampaignState] = []
    labels: list[CampaignLabel] = []
    with TestClient(app) as client:
        for plan in plans:
            state, label = run_campaign(
                client, plan, secrets=secrets, config=config, cap_max_turns=cap_max_turns
            )
            states.append(state)
            labels.append(label)

    verification_ok = True
    if verify:
        verification = EvidenceVerifier(log_path).verify()
        verification_ok = verification.ok
        if not verification_ok:
            raise ValueError(f"hash-chain verification failed: {verification.to_dict()}")

    records = read_records(log_path)
    _structural_anti_leak(records)

    event_types = [r["event_type"] for r in records]
    summary = {
        "status": "ok",
        "environment": environment,
        "run_id": run_id,
        "campaign_count": len(plans),
        "turn_count": sum(s.length for s in states),
        "blocked_turn_count": sum(len(l.blocked_turns) for l in labels),
        "record_count": len(records),
        "event_type_counts": {et: event_types.count(et) for et in sorted(set(event_types))},
        "campaign_lengths": {s.campaign_id: s.length for s in states},
        "campaign_outcomes": {l.campaign_id: l.campaign_outcome for l in labels},
        "defense_outcomes": sorted({
            t.defense_outcome for s in states for t in s.turns
        }),
        "verification_ok": verification_ok,
    }

    return CampaignRunResult(
        environment=environment,
        forensic_records=records,
        states=states,
        labels=labels,
        summary=summary,
        verification_ok=verification_ok,
    )


__all__ = [
    "CampaignRunResult",
    "run_campaign",
    "run_campaigns",
    "DEFAULT_RUN_ID",
    "DEFAULT_SALT",
    "DEFAULT_CAP_MAX_TURNS",
]
