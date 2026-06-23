"""Runner for the query-based MIA / extraction plan on a Pile-trained model.

Drives ``build_mia_pile_plan`` turn-by-turn against the FastAPI app and collects
the Tier-1 forensic stream (the query residues the detector consumes) plus the
separate ground truth, and evaluates extraction success against the held-out
suffix (Carlini 2022 discoverable extraction). It is the ``mia_pile`` analogue of
``pipeline.run_integrated_pipeline``: a library function returning in-memory
results, testable on the deterministic mock (no GPU).

Real runs host **Pythia** by passing a transformers backend, e.g.::

    from forensic.backends_transformers import TransformersBackend
    backend = TransformersBackend(model_id="EleutherAI/pythia-1.4b",
                                  revision="step99000", load_in_4bit=False)
    run_mia_pile(log_path=..., targets=targets, backend=backend)

The membership signal is in the model's continuation; the secret is in Pythia's
training (the Pile), so the attack has a real referent — unlike a frozen Mistral.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi.testclient import TestClient

from .hashing import pseudonymize
from .mia_pile import (
    MiaTarget,
    build_divergence_plan,
    build_mia_pile_plan,
    evaluate_divergence,
    evaluate_extraction,
)
from .pipeline import _structural_anti_leak
from .server import create_app
from .traffic import assert_no_groundtruth_in_request
from .verifier import EvidenceVerifier

DEFAULT_RUN_ID = "mia-pile"
DEFAULT_SALT = b"mia-pile-synthetic-salt-32bytes!!"


@dataclass(frozen=True)
class MiaPileResult:
    """In-memory outcome of a query-based MIA/extraction run."""

    forensic_records: list[dict[str, Any]]
    groundtruth_records: list[dict[str, Any]]
    extraction_results: list[dict[str, Any]]
    summary: dict[str, Any]
    verification_ok: bool


def run_mia_pile(
    *,
    log_path: str,
    targets: list[MiaTarget],
    salt: bytes = DEFAULT_SALT,
    run_id: str = DEFAULT_RUN_ID,
    repo_path: str = ".",
    model_id: str = "deterministic-mock-model",
    model_revision: str = "mia-pile-mock",
    model_hash: str = "deterministic-mock-model",
    backend: Any | None = None,
    environment: str = "E0",
    session_prefix: str = "miapile",
    max_tokens: int = 64,
    system_prompt: str = "",
    expose_logprobs: bool = True,
    read_records: Callable[[str], list[dict[str, Any]]] | None = None,
    verify: bool = True,
) -> MiaPileResult:
    """Execute the MIA/extraction plan against the app and return dataset + summary.

    ``backend`` defaults to the deterministic mock (which never regenerates a
    suffix, so extraction is always False on the mock — real residuals require a
    real model). The ground-truth ``session_id`` is pseudonymised with the same
    ``salt`` the server uses (reconciliation via the pipeline, so the offline join
    matches without passing the salt to ``build_features``).
    """
    import json

    if read_records is None:
        def read_records(path: str) -> list[dict[str, Any]]:  # type: ignore[misc]
            from pathlib import Path

            return [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    plan = build_mia_pile_plan(targets, session_prefix=session_prefix, max_tokens=max_tokens)
    targets_by_id = {t.target_id: t for t in targets}

    app = create_app(
        log_path=log_path,
        salt=salt,
        run_id=run_id,
        experiment_phase="mia_pile_probing",
        model_id=model_id,
        model_revision=model_revision,
        model_hash=model_hash,
        repo_path=repo_path,
        experiment_config={
            "mock_mode": backend is None,
            "simulation": "mia_pile",
            "world": "pythia_pile",
            "groundtruth_separate": True,
        },
        backend=backend,
        environment=environment,
        # Pythia is a BASE model probed by raw prefix-continuation: no system
        # prompt (the Mistral level-1 defence prompt belongs to the app world).
        system_prompt=system_prompt,
        expose_logprobs=expose_logprobs,
    )

    groundtruth_records: list[dict[str, Any]] = []
    extraction_results: list[dict[str, Any]] = []

    with TestClient(app) as client:
        for case in plan:
            assert_no_groundtruth_in_request(case)
            resp = client.post(case.endpoint, json=case.request_json())
            resp.raise_for_status()
            body = resp.json()
            gt = case.groundtruth_json()
            # via (ii): pseudonymise the gt session_id with the server's salt so
            # (session_id, seq, endpoint) matches the (pseudonymised) forensic key.
            gt["session_id"] = pseudonymize(gt["session_id"], salt)
            gt.update(
                {
                    "prompt_record_hash": body.get("prompt_record_hash"),
                    "completion_record_hash": body.get("completion_record_hash"),
                    "response_hash": body.get("response_hash"),
                    "http_status": resp.status_code,
                }
            )
            groundtruth_records.append(gt)

            tid = gt.get("target_id")
            if gt.get("is_attack") and tid in targets_by_id:
                ev = evaluate_extraction(targets_by_id[tid], body.get("response", ""))
                ev["session_id"] = gt["session_id"]
                ev["sequence_number"] = gt["sequence_number"]
                extraction_results.append(ev)

    verification_ok = True
    if verify:
        verification = EvidenceVerifier(log_path).verify()
        verification_ok = verification.ok
        if not verification_ok:
            raise ValueError(f"hash-chain verification failed: {verification.to_dict()}")

    records = read_records(log_path)
    _structural_anti_leak(records)

    n_attack = sum(1 for g in groundtruth_records if g.get("is_attack"))
    summary = {
        "status": "ok",
        "run_id": run_id,
        "world": "pythia_pile",
        "model_id": model_id,
        "case_count": len(plan),
        "attack_count": n_attack,
        "benign_count": len(groundtruth_records) - n_attack,
        "member_count": sum(1 for g in groundtruth_records if g.get("membership_truth") is True),
        "nonmember_count": sum(1 for g in groundtruth_records if g.get("membership_truth") is False),
        "secret_bearing_count": sum(1 for g in groundtruth_records if g.get("is_secret_bearing")),
        "extraction_attempts": len(extraction_results),
        "extraction_succeeded": sum(1 for e in extraction_results if e["extracted"]),
        "record_count": len(records),
        "verification_ok": verification_ok,
    }

    return MiaPileResult(
        forensic_records=records,
        groundtruth_records=groundtruth_records,
        extraction_results=extraction_results,
        summary=summary,
        verification_ok=verification_ok,
    )


@dataclass(frozen=True)
class DivergenceResult:
    forensic_records: list[dict[str, Any]]
    groundtruth_records: list[dict[str, Any]]
    divergence_results: list[dict[str, Any]]
    summary: dict[str, Any]
    verification_ok: bool


def run_divergence(
    *,
    log_path: str,
    salt: bytes = DEFAULT_SALT,
    run_id: str = "divergence",
    repo_path: str = ".",
    model_id: str = "deterministic-mock-model",
    model_revision: str = "divergence-mock",
    model_hash: str = "deterministic-mock-model",
    backend: Any | None = None,
    known_members: list | None = None,
    repeat: int = 50,
    max_tokens: int = 200,
    read_records: Callable[[str], list[dict[str, Any]]] | None = None,
    verify: bool = True,
) -> DivergenceResult:
    """Run the divergence extraction plan; harvest each response and evaluate
    whether it surfaced secret-like content (and, optionally, a known training
    member). The mock never leaks, so on the mock both counts are 0."""
    import json

    if read_records is None:
        def read_records(path: str) -> list[dict[str, Any]]:  # type: ignore[misc]
            from pathlib import Path

            return [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    plan = build_divergence_plan(repeat=repeat, max_tokens=max_tokens)
    app = create_app(
        log_path=log_path, salt=salt, run_id=run_id, experiment_phase="divergence_extraction",
        model_id=model_id, model_revision=model_revision, model_hash=model_hash, repo_path=repo_path,
        experiment_config={"mock_mode": backend is None, "simulation": "divergence",
                           "world": "pythia_pile", "groundtruth_separate": True},
        backend=backend, environment="E0", system_prompt="", expose_logprobs=True,
    )

    gt_records: list[dict[str, Any]] = []
    div_results: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for case in plan:
            assert_no_groundtruth_in_request(case)
            resp = client.post(case.endpoint, json=case.request_json())
            resp.raise_for_status()
            body = resp.json()
            gt = case.groundtruth_json()
            gt["session_id"] = pseudonymize(gt["session_id"], salt)
            gt.update(
                {
                    "prompt_record_hash": body.get("prompt_record_hash"),
                    "completion_record_hash": body.get("completion_record_hash"),
                    "response_hash": body.get("response_hash"),
                    "http_status": resp.status_code,
                }
            )
            gt_records.append(gt)
            if gt.get("is_attack"):
                ev = evaluate_divergence(body.get("response", ""), known_members)
                ev["case_id"] = gt["case_id"]
                ev["divergence_seed"] = gt.get("divergence_seed")
                div_results.append(ev)

    verification_ok = True
    if verify:
        verification = EvidenceVerifier(log_path).verify()
        verification_ok = verification.ok
        if not verification_ok:
            raise ValueError(f"hash-chain verification failed: {verification.to_dict()}")

    records = read_records(log_path)
    _structural_anti_leak(records)
    n_attack = sum(1 for g in gt_records if g.get("is_attack"))
    summary = {
        "status": "ok",
        "run_id": run_id,
        "world": "pythia_pile",
        "model_id": model_id,
        "case_count": len(plan),
        "attack_count": n_attack,
        "benign_count": len(gt_records) - n_attack,
        "secret_like_count": sum(1 for e in div_results if e["secret_like"]),
        "member_match_count": sum(1 for e in div_results if e.get("member_match")),
        "record_count": len(records),
        "verification_ok": verification_ok,
    }
    return DivergenceResult(records, gt_records, div_results, summary, verification_ok)


__all__ = [
    "MiaPileResult", "run_mia_pile", "DivergenceResult", "run_divergence",
    "DEFAULT_RUN_ID", "DEFAULT_SALT",
]
