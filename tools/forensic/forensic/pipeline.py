"""Integrated execution of the attack/benign traffic plans (testable library).

This module is the runner that turns the simulators into a single forensic
dataset: it executes the combined traffic plan (M3.5 baseline + RAG attacks +
application-layer MIA probing) against the FastAPI app, collects the Tier-1
forensic stream and the separate ground truth, and runs the standard integrity
checks (hash-chain verification + structural anti-leak). It is written as a
library function returning in-memory results so it can be unit-tested without
touching disk; the operator script ``runners/run_integrated_pipeline_real.py`` wraps it
to persist evidence/groundtruth/summary files.

Design points
-------------
* The combined plan reuses the existing per-simulator builders unchanged. To
  avoid (session_id, sequence_number, endpoint) key collisions across plans —
  the join key used by ``features.build_features`` — each source plan keeps its
  own session_ids (already distinct by prefix), so no renumbering is required;
  this is asserted explicitly.
* Ground-truth labels are never sent to the server (``assert_no_groundtruth_in_request``)
  and never written to the forensic stream (structural ``_reject_groundtruth_keys``).
* The expected event sequence includes a ``logprobs`` event after every
  completion: logprobs are logged symmetrically for all traffic,
  not only for score-exposing attack cases.
* The backend/retriever default to the deterministic mocks; a real backend and a
  Chroma retriever can be injected for model-dependent runs (after rebuild).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi.testclient import TestClient

from .application import _FORBIDDEN_RETRIEVAL_CONTENT_KEYS, _reject_groundtruth_keys
from .hashing import pseudonymize
from .server import create_app
from .traffic import (
    TrafficCase,
    assert_no_groundtruth_in_request,
    build_m3_traffic_plan,
)
from .traffic_mia import build_mia_probing_plan
from .traffic_rag import build_rag_attack_plan
from .traffic_volume import scale_plan
from .verifier import EvidenceVerifier

DEFAULT_RUN_ID = "integrated-pipeline"
DEFAULT_SALT = b"integrated-pipeline-synthetic-salt!!"


@dataclass(frozen=True)
class PipelineResult:
    """In-memory outcome of an integrated run."""

    forensic_records: list[dict[str, Any]]
    groundtruth_records: list[dict[str, Any]]
    summary: dict[str, Any]
    verification_ok: bool


def build_combined_plan(
    *, rag_variants: int = 1, mia_variants: int = 1, benign_variants: int = 1
) -> list[TrafficCase]:
    """Concatenate the three source plans into one combined traffic plan.

    Order: baseline (M3.5) -> RAG attacks -> application MIA probing. The RAG and
    MIA attack plans can be scaled into multiple deterministic session variants
    (``rag_variants`` / ``mia_variants``); the benign cases of the M3.5 baseline
    can be scaled independently (``benign_variants``) to keep the negative class
    large enough for an honest false-positive-rate estimate. Session ids are
    distinct across sources and across variants by construction; this function
    asserts that no (session_id, sequence_number, endpoint) join key collides,
    since that key is what the offline feature join relies on.
    """
    baseline = build_m3_traffic_plan()
    baseline_attack = [c for c in baseline if c.groundtruth["is_attack"]]
    baseline_benign = [c for c in baseline if not c.groundtruth["is_attack"]]

    plan: list[TrafficCase] = []
    plan.extend(baseline_attack)
    plan.extend(scale_plan(baseline_benign, variants=benign_variants))
    plan.extend(scale_plan(build_rag_attack_plan(), variants=rag_variants))
    plan.extend(scale_plan(build_mia_probing_plan(), variants=mia_variants))

    seen: set[tuple[str, int, str]] = set()
    for case in plan:
        key = (case.body["session_id"], int(case.body["sequence_number"]), case.endpoint)
        if key in seen:
            raise ValueError(f"join-key collision across plans: {key} (case {case.case_id})")
        seen.add(key)
    return plan


def _expected_event_sequence(
    plan: list[TrafficCase], *, expose_logprobs: bool = True
) -> list[str]:
    """Compute the expected Tier-1 event sequence for the plan (manifest first).

    Assumes every request is ACCEPTED (E0, or E1 without any defence trip): the
    runner sends the same plan to either environment (symmetry). Blocked
    requests (E1) would log a ``prompt`` without a following ``completion`` —
    out of scope here (the benign traffic that trips defences is a separate study).
    """
    expected = ["manifest"]
    for case in plan:
        expected.append("prompt")
        if case.endpoint == "/v1/rag":
            expected.append("rag_retrieval")
        elif case.endpoint != "/v1/complete":
            raise ValueError(f"unexpected endpoint: {case.endpoint}")
        expected.append("completion")
        # When the score switch is ON (the default), the server logs
        # logprobs symmetrically for every request (the mock backend always
        # produces them), so a logprobs event follows every completion regardless
        # of the case's attack/benign nature. When OFF, no logprobs are emitted.
        if expose_logprobs:
            expected.append("logprobs")
    return expected


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


def run_integrated_pipeline(
    *,
    log_path: str,
    salt: bytes = DEFAULT_SALT,
    run_id: str = DEFAULT_RUN_ID,
    repo_path: str = ".",
    model_id: str = "deterministic-mock-model",
    model_revision: str = "integrated-pipeline-mock",
    model_hash: str = "deterministic-mock-model",
    backend: Any | None = None,
    retriever: Any | None = None,
    notes: str = "",
    model_artifacts: Mapping[str, Path] | None = None,
    dataset_paths: Mapping[str, Path] | None = None,
    plan_factory: Callable[[], list[TrafficCase]] | None = None,
    rag_variants: int = 1,
    mia_variants: int = 1,
    benign_variants: int = 1,
    read_records: Callable[[str], list[dict[str, Any]]] | None = None,
    verify: bool = True,
    environment: str = "E0",
    expose_logprobs: bool = True,
) -> PipelineResult:
    """Execute the combined plan against the app and return the dataset + summary.

    ``read_records`` reads back the written JSONL (injected so the caller decides
    the IO); if omitted, a default UTF-8 JSONL reader is used. ``backend`` and
    ``retriever`` default to the mocks inside ``create_app``. ``rag_variants`` /
    ``mia_variants`` / ``benign_variants`` scale the plans for volume (see
    build_combined_plan); a custom ``plan_factory`` overrides them entirely.
    """
    import json

    if plan_factory is None:
        def plan_factory() -> list[TrafficCase]:  # type: ignore[misc]
            return build_combined_plan(
                rag_variants=rag_variants,
                mia_variants=mia_variants,
                benign_variants=benign_variants,
            )

    if read_records is None:
        def read_records(path: str) -> list[dict[str, Any]]:  # type: ignore[misc]
            from pathlib import Path

            return [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    plan = plan_factory()
    groundtruth_records: list[dict[str, Any]] = []

    app = create_app(
        log_path=log_path,
        salt=salt,
        run_id=run_id,
        experiment_phase="integrated_pipeline",
        model_id=model_id,
        model_revision=model_revision,
        model_hash=model_hash,
        repo_path=repo_path,
        experiment_config={
            "mock_mode": backend is None and retriever is None,
            "simulation": "integrated_pipeline",
            "groundtruth_separate": True,
            "plans": ["m3_traffic", "rag_attacks", "mia_probing"],
        },
        backend=backend,
        retriever=retriever,
        notes=notes,
        model_artifacts=model_artifacts,
        dataset_paths=dataset_paths,
        environment=environment,
        expose_logprobs=expose_logprobs,
    )

    with TestClient(app) as client:
        for case in plan:
            assert_no_groundtruth_in_request(case)
            resp = client.post(case.endpoint, json=case.request_json())
            resp.raise_for_status()
            body = resp.json()
            gt = case.groundtruth_json()
            # Pseudonymize the groundtruth session_id with the SAME salt as the
            # server, so the (session_id, seq, endpoint) key matches the forensic
            # one (pseudonymized in _append). case_id stays raw in the groundtruth
            # as a human-readable handle for inspection.
            gt["session_id"] = pseudonymize(gt["session_id"], salt)
            gt.update(
                {
                    "prompt_record_hash": body.get("prompt_record_hash"),
                    "retrieval_record_hash": body.get("retrieval_record_hash"),
                    "completion_record_hash": body.get("completion_record_hash"),
                    "response_hash": body.get("response_hash"),
                    "http_status": resp.status_code,
                }
            )
            groundtruth_records.append(gt)

    verification_ok = True
    if verify:
        verification = EvidenceVerifier(log_path).verify()
        verification_ok = verification.ok
        if not verification_ok:
            raise ValueError(f"hash-chain verification failed: {verification.to_dict()}")

    records = read_records(log_path)
    event_types = [r["event_type"] for r in records]
    expected = _expected_event_sequence(plan, expose_logprobs=expose_logprobs)
    if event_types != expected:
        raise ValueError(
            f"unexpected event sequence: expected {len(expected)} events, observed {len(event_types)}"
        )
    _structural_anti_leak(records)

    # Aggregates for the summary
    scenario_counts: dict[str, int] = {}
    attack_counts = {"benign": 0, "attack": 0}
    family_counts: dict[str, int] = {}
    for gt in groundtruth_records:
        scenario_counts[gt["scenario"]] = scenario_counts.get(gt["scenario"], 0) + 1
        attack_counts["attack" if gt["is_attack"] else "benign"] += 1
        fam = gt.get("attack_family")
        if fam:
            family_counts[fam] = family_counts.get(fam, 0) + 1

    summary = {
        "status": "ok",
        "run_id": run_id,
        "case_count": len(plan),
        "groundtruth_count": len(groundtruth_records),
        "record_count": len(records),
        "event_type_counts": {et: event_types.count(et) for et in sorted(set(event_types))},
        "scenario_counts": scenario_counts,
        "attack_counts": attack_counts,
        "attack_family_counts": family_counts,
        "verification_ok": verification_ok,
    }

    return PipelineResult(
        forensic_records=records,
        groundtruth_records=groundtruth_records,
        summary=summary,
        verification_ok=verification_ok,
    )


__all__ = [
    "PipelineResult",
    "build_combined_plan",
    "run_integrated_pipeline",
    "DEFAULT_RUN_ID",
    "DEFAULT_SALT",
]
