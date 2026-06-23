"""Session-level detector for the Pythia-world attacks (the thesis core: residues
→ distinguish attacks from benign).

A single attack probe ~ a benign request, so per-request detection is weak; what
is detectable is the **campaign** — a session that systematically probes/harvests.
This module therefore (1) generates realistic BENIGN traffic (varied session
sizes, including a few large "power-user" sessions so volume alone is not a
setup-artifact proxy), (2) lays the attack campaigns as many sessions (chunked),
(3) aggregates the per-request forensic features into **per-session** features
(volume, prompt/response profile, latency, secret-like-response rate, prompt
diversity), and (4) trains a grouped-CV classifier, reporting FP/FN/TPR/AUC.

Anti-circularity: features are derived from the OBSERVED residuals; labels
never enter the forensic stream; benign is realistic, not engineered to
mirror the attack — the detector earns its keep by finding the real difference.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from fastapi.testclient import TestClient

from .detector_ml import build_xy, cross_validate_grouped
from .features import build_features
from .hashing import pseudonymize
from .mia_pile import build_divergence_plan, build_mia_pile_plan
from .mia_score import build_mia_score_plan, roc_auc
from .pipeline import _structural_anti_leak
from .response_sidechannel import session_sidechannel_features
from .server import create_app
from .text_features import session_chaining_features, session_text_features
from .traffic import TrafficCase, _base_body, assert_no_groundtruth_in_request
from .verifier import EvidenceVerifier

DEFAULT_RUN_ID = "pile-detector"
DEFAULT_SALT = b"pile-detector-synthetic-salt-32b!"

_BENIGN_PROMPTS = (
    "Write a short function that sums a list of integers.",
    "Explain what a hash function is in two sentences.",
    "How do I reverse a string in Python?",
    "Summarise the steps of incident response.",
    "What is the difference between TCP and UDP?",
    "Give an example of a SQL SELECT with a WHERE clause.",
    "Describe how a binary search works.",
    "What are good practices for password storage?",
    "Write a regex that matches an email address.",
    "Explain what a deadlock is.",
)


def _benign_case(case_id, session_id, seq, prompt, *, max_tokens):
    body = _base_body(case_id=case_id, session_id=session_id, user_id=session_id, sequence_number=seq)
    rm = dict(body.get("request_metadata", {}))
    rm["simulator"] = "benign_simulator"
    body["request_metadata"] = rm
    body.update({"prompt": prompt, "max_tokens": max_tokens})
    return TrafficCase(
        case_id=case_id, scenario="benign_completion", endpoint="/v1/complete", body=body,
        groundtruth={
            "case_id": case_id, "scenario": "benign_completion", "endpoint": "/v1/complete",
            "is_attack": False, "attack_family": None, "objective": "legitimate_use",
            "session_id": session_id, "sequence_number": seq,
        },
    )


def build_benign_sessions(
    *, corpus_texts: list | None = None, session_prefix: str = "ben",
    n_short: int = 18, n_multi: int = 8, multi_size: int = 5,
    n_codecomplete: int = 16, n_check: int = 14, n_coverage: int = 6, coverage_size: int = 20,
) -> list[TrafficCase]:
    """MIRRORED benign traffic — overlaps the attack's observable channels so the
    detector cannot separate on a trivial channel:
    * short Q&A (single) — ordinary use;
    * multi-request Q&A sessions;
    * LONG code-completion sessions (legit held-out code) — overlaps the attack
      prompt-length channel;
    * HIGH-VOLUME coverage sessions — overlaps the attack volume+diversity channel.
    What still separates (on a real model) is the attack's response-side leakage
    and systematic-probing profile, not length/volume."""
    texts = corpus_texts or []
    cases: list[TrafficCase] = []
    for i in range(n_short):
        sid = f"{session_prefix}-s-{i:03d}"
        cases.append(_benign_case(sid, sid, 1, _BENIGN_PROMPTS[i % len(_BENIGN_PROMPTS)], max_tokens=32))
    for j in range(n_multi):
        sid = f"{session_prefix}-m-{j:03d}"
        for t in range(multi_size):
            cases.append(_benign_case(f"{sid}-{t}", sid, t + 1,
                                      _BENIGN_PROMPTS[(j * 7 + t) % len(_BENIGN_PROMPTS)], max_tokens=32))
    # benign content is the FULL held-out text (same length distribution as the
    # attack candidates) — NOT artificially shortened, else prompt length becomes
    # a setup artefact instead of a residual.
    for c in range(n_codecomplete):
        sid = f"{session_prefix}-code-{c:03d}"
        body = texts[c % len(texts)] if texts else _BENIGN_PROMPTS[c % len(_BENIGN_PROMPTS)]
        cases.append(_benign_case(sid, sid, 1, body, max_tokens=48))
    for q in range(n_check):  # "quick check": long content + minimal response — overlaps the MIA-score profile
        sid = f"{session_prefix}-chk-{q:03d}"
        body = texts[(q + 3) % len(texts)] if texts else _BENIGN_PROMPTS[q % len(_BENIGN_PROMPTS)]
        cases.append(_benign_case(sid, sid, 1, body, max_tokens=2))
    for v in range(n_coverage):  # high-volume coverage of varied full-length content
        sid = f"{session_prefix}-cov-{v:03d}"
        for t in range(coverage_size):
            body = texts[(v * coverage_size + t) % len(texts)] if texts else _BENIGN_PROMPTS[t % len(_BENIGN_PROMPTS)]
            cases.append(_benign_case(f"{sid}-{t}", sid, t + 1, body, max_tokens=24))
    return cases


def build_attack_sessions(targets: list, *, chunk_size: int = 20, session_prefix: str = "atk") -> list[TrafficCase]:
    """Lay the attack campaign as MANY sessions of MIXED type (so the positive
    class is varied, not uniform): MIA-score and extraction campaigns over
    candidate chunks, plus a couple of divergence-harvest sessions."""
    cases: list[TrafficCase] = []
    for idx, k in enumerate(range(0, len(targets), chunk_size)):
        chunk = targets[k:k + chunk_size]
        sp = f"{session_prefix}-{idx:03d}"
        if idx % 2 == 0:
            plan = build_mia_score_plan(chunk, session_prefix=sp)                 # long prompt, 1-token probe
        else:
            plan = build_mia_pile_plan(chunk, session_prefix=sp, max_tokens=48)   # prefix, long harvest
        cases.extend(c for c in plan if c.groundtruth["is_attack"])
    for d in range(2):
        plan = build_divergence_plan(session_prefix=f"{session_prefix}-div-{d:03d}", repeat=40, max_tokens=64)
        cases.extend(c for c in plan if c.groundtruth["is_attack"])
    return cases


def aggregate_sessions(feature_rows: list) -> list[dict]:
    """Per-request feature rows → one row per session with aggregated features."""
    by: dict[str, list] = defaultdict(list)
    for r in feature_rows:
        by[str(r.get("session_id"))].append(r)

    def _col(rows, name):
        return [float(r[name]) for r in rows if isinstance(r.get(name), (int, float, bool))]

    out: list[dict] = []
    for sid, rows in by.items():
        n = len(rows)
        plen = _col(rows, "feature_prompt_length_chars")
        rlen = _col(rows, "feature_response_length_chars")
        lat = _col(rows, "feature_latency_total_ms")
        secret = sum(1 for r in rows if r.get("feature_response_contains_secret_like_pattern"))
        phash = {r.get("prompt_hash") for r in rows}
        row = {
            "session_id": sid,
            "label_is_attack": 1 if any(r.get("label_is_attack") for r in rows) else 0,
            "feature_session_n_requests": float(n),
            "feature_session_mean_prompt_len": st.mean(plen) if plen else 0.0,
            "feature_session_max_prompt_len": float(max(plen)) if plen else 0.0,
            "feature_session_std_prompt_len": st.pstdev(plen) if len(plen) > 1 else 0.0,
            "feature_session_mean_response_len": st.mean(rlen) if rlen else 0.0,
            "feature_session_std_response_len": st.pstdev(rlen) if len(rlen) > 1 else 0.0,
            "feature_session_mean_latency": st.mean(lat) if lat else 0.0,
            "feature_session_secret_like_rate": secret / n if n else 0.0,
            "feature_session_prompt_diversity": len(phash) / n if n else 0.0,
        }
        # anti-circular textual session features (degeneracy / self-similarity /
        # incompleteness) — recognise extraction/probing prompt patterns, no keywords.
        row.update(session_text_features([str(r.get("prompt_norm", "")) for r in rows]))
        # response-side-channel session features (latency variability, refusal rate,
        # formatting profile) — anti-circular, derived from response characteristics.
        row.update(session_sidechannel_features(rows))
        # TARGET C: multi-turn fingerprints (chaining response[N]->prompt[N+1] + response
        # novelty), from the ORDERED (prompt, response) pairs — candidate stealth signals,
        # evaluated OOD. Anti-circular (logged text only); response_norm is not a feature_.
        _ordered = sorted(rows, key=lambda r: int(r.get("sequence_number") or 0))
        row.update(session_chaining_features(
            [(str(r.get("prompt_norm", "")), str(r.get("response_norm", ""))) for r in _ordered]))
        out.append(row)
    return out


def _binary_metrics(y_true, y_pred, y_score) -> dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    def _s(a, b):
        return a / b if b else 0.0

    return {
        "n_sessions": len(y_true), "n_attack": tp + fn, "n_benign": tn + fp,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr_recall": _s(tp, tp + fn), "fpr": _s(fp, fp + tn),
        "precision": _s(tp, tp + fp), "accuracy": _s(tp + tn, len(y_true)),
        "roc_auc": roc_auc(list(y_score), list(y_true)),
    }


@dataclass(frozen=True)
class DetectorResult:
    session_rows: list[dict]
    metrics: dict[str, Any]
    summary: dict[str, Any]
    verification_ok: bool


def run_pile_detector(
    *,
    attack_targets: list,
    log_path: str,
    salt: bytes = DEFAULT_SALT,
    run_id: str = DEFAULT_RUN_ID,
    repo_path: str = ".",
    backend: Any | None = None,
    benign_corpus: list | None = None,
    chunk_size: int = 20,
    model_name: str = "logistic",
    n_splits: int = 5,
    read_records: Callable[[str], list[dict[str, Any]]] | None = None,
    verify: bool = True,
) -> DetectorResult:
    import json

    if read_records is None:
        def read_records(path: str) -> list[dict[str, Any]]:  # type: ignore[misc]
            from pathlib import Path

            return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

    if benign_corpus is None:  # mirrored benign content = held-out (non-member) code
        benign_corpus = [t.full_text for t in attack_targets if not t.is_member]
    plan = build_attack_sessions(attack_targets, chunk_size=chunk_size) + build_benign_sessions(corpus_texts=benign_corpus)
    app = create_app(
        log_path=log_path, salt=salt, run_id=run_id, experiment_phase="pile_detector",
        model_id="deterministic-mock-model" if backend is None else "pile-detector-model",
        repo_path=repo_path,
        experiment_config={"mock_mode": backend is None, "simulation": "pile_detector",
                           "world": "pythia_pile", "groundtruth_separate": True},
        backend=backend, environment="E0", system_prompt="", expose_logprobs=True,
    )

    gt_records: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for case in plan:
            assert_no_groundtruth_in_request(case)
            resp = client.post(case.endpoint, json=case.request_json())
            resp.raise_for_status()
            body = resp.json()
            gt = case.groundtruth_json()
            gt["session_id"] = pseudonymize(gt["session_id"], salt)
            gt.update({
                "prompt_record_hash": body.get("prompt_record_hash"),
                "completion_record_hash": body.get("completion_record_hash"),
                "response_hash": body.get("response_hash"),
                "http_status": resp.status_code,
            })
            gt_records.append(gt)

    verification_ok = True
    if verify:
        verification = EvidenceVerifier(log_path).verify()
        verification_ok = verification.ok
        if not verification_ok:
            raise ValueError(f"hash-chain verification failed: {verification.to_dict()}")

    records = read_records(log_path)
    _structural_anti_leak(records)
    feature_rows = build_features(records, gt_records)
    session_rows = aggregate_sessions(feature_rows)

    ds = build_xy(session_rows)
    cv = cross_validate_grouped(ds, model_name=model_name, n_splits=n_splits)
    metrics = _binary_metrics(cv.oof_y_true, cv.oof_y_pred, cv.oof_y_score)

    # univariate AUC per session feature — shows WHICH residual drives detection
    labels = [s["label_is_attack"] for s in session_rows]
    feat_names = sorted({k for s in session_rows for k in s if k.startswith("feature_")})
    feature_aucs = {
        name: round(roc_auc([float(s.get(name) or 0.0) for s in session_rows], labels), 3)
        for name in feat_names
    }
    feature_aucs = dict(sorted(feature_aucs.items(), key=lambda kv: -abs(kv[1] - 0.5)))

    n_attack_sessions = sum(1 for s in session_rows if s["label_is_attack"])
    summary = {
        "status": "ok", "run_id": run_id, "world": "pythia_pile", "model": model_name,
        "n_sessions": len(session_rows), "n_attack_sessions": n_attack_sessions,
        "n_benign_sessions": len(session_rows) - n_attack_sessions,
        "n_requests": len(feature_rows), "metrics": metrics,
        "feature_aucs": feature_aucs, "verification_ok": verification_ok,
    }
    return DetectorResult(session_rows, metrics, summary, verification_ok)


__all__ = [
    "build_benign_sessions", "build_attack_sessions", "aggregate_sessions",
    "run_pile_detector", "DetectorResult", "DEFAULT_RUN_ID", "DEFAULT_SALT",
]
