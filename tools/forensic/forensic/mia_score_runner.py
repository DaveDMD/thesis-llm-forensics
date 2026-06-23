"""Runner for the score-based MIA on a Pile-trained model.

Submits each candidate to the server (the forensic query residue) and reads its
per-token log-probs via ``backend.score_sequence`` (white-box) to compute the
membership scores, reporting **ROC-AUC per scorer** (LOSS/Min-K%/Min-K%++/zlib).
Testable on the deterministic mock (no GPU); real runs pass a Pythia backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi.testclient import TestClient

from .hashing import pseudonymize
from .mia_score import SCORERS, build_mia_score_plan, mia_ne, mia_ref, roc_auc
from .pipeline import _structural_anti_leak
from .server import create_app
from .traffic import assert_no_groundtruth_in_request
from .verifier import EvidenceVerifier

DEFAULT_RUN_ID = "mia-score"
DEFAULT_SALT = b"mia-score-synthetic-salt-32byte!!"


@dataclass(frozen=True)
class MiaScoreResult:
    forensic_records: list[dict[str, Any]]
    groundtruth_records: list[dict[str, Any]]
    auc: dict[str, float]
    summary: dict[str, Any]
    verification_ok: bool


def run_mia_score(
    *,
    log_path: str,
    targets: list,
    backend: Any | None = None,
    reference_backend: Any | None = None,
    salt: bytes = DEFAULT_SALT,
    run_id: str = DEFAULT_RUN_ID,
    repo_path: str = ".",
    model_id: str = "deterministic-mock-model",
    model_revision: str = "mia-score-mock",
    model_hash: str = "deterministic-mock-model",
    environment: str = "E0",
    session_prefix: str = "miascore",
    read_records: Callable[[str], list[dict[str, Any]]] | None = None,
    verify: bool = True,
) -> MiaScoreResult:
    import json

    if read_records is None:
        def read_records(path: str) -> list[dict[str, Any]]:  # type: ignore[misc]
            from pathlib import Path

            return [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    if backend is None:
        from .backends import DeterministicMockBackend

        backend = DeterministicMockBackend()

    plan = build_mia_score_plan(targets, session_prefix=session_prefix)
    targets_by_id = {t.target_id: t for t in targets}

    app = create_app(
        log_path=log_path, salt=salt, run_id=run_id, experiment_phase="mia_score_probing",
        model_id=model_id, model_revision=model_revision, model_hash=model_hash, repo_path=repo_path,
        experiment_config={"simulation": "mia_score", "world": "pythia_pile", "groundtruth_separate": True},
        backend=backend, environment=environment, system_prompt="", expose_logprobs=True,
    )

    gt_records: list[dict[str, Any]] = []
    per_scorer: dict[str, dict[str, list]] = {
        name: {"scores": [], "labels": []} for name in list(SCORERS) + ["ref", "ne"]
    }

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
                    "http_status": resp.status_code,
                }
            )
            gt_records.append(gt)
            tid = gt.get("target_id")
            if gt.get("is_attack") and tid in targets_by_id:
                t = targets_by_id[tid]
                label = gt["membership_truth"]
                s = backend.score_sequence(t.full_text)
                for name, fn in SCORERS.items():
                    per_scorer[name]["scores"].append(fn(s))
                    per_scorer[name]["labels"].append(label)
                if reference_backend is not None:  # reference-based MIA
                    rs = reference_backend.score_sequence(t.full_text)
                    per_scorer["ref"]["scores"].append(mia_ref(s, rs))
                    per_scorer["ref"]["labels"].append(label)
                if t.neighbors:  # neighbourhood MIA
                    nss = [backend.score_sequence(n) for n in t.neighbors]
                    per_scorer["ne"]["scores"].append(mia_ne(s, nss))
                    per_scorer["ne"]["labels"].append(label)

    verification_ok = True
    if verify:
        verification = EvidenceVerifier(log_path).verify()
        verification_ok = verification.ok
        if not verification_ok:
            raise ValueError(f"hash-chain verification failed: {verification.to_dict()}")

    records = read_records(log_path)
    _structural_anti_leak(records)

    auc = {name: roc_auc(d["scores"], d["labels"]) for name, d in per_scorer.items() if d["scores"]}
    n_attack = sum(1 for g in gt_records if g.get("is_attack"))
    summary = {
        "status": "ok",
        "run_id": run_id,
        "world": "pythia_pile",
        "model_id": model_id,
        "case_count": len(plan),
        "attack_count": n_attack,
        "benign_count": len(gt_records) - n_attack,
        "member_count": sum(1 for g in gt_records if g.get("membership_truth") is True),
        "nonmember_count": sum(1 for g in gt_records if g.get("membership_truth") is False),
        "auc": auc,
        "record_count": len(records),
        "verification_ok": verification_ok,
    }
    return MiaScoreResult(records, gt_records, auc, summary, verification_ok)


__all__ = ["MiaScoreResult", "run_mia_score", "DEFAULT_RUN_ID", "DEFAULT_SALT"]
