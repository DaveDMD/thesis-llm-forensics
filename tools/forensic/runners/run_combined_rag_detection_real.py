#!/usr/bin/env python3
"""Combined RAG detection on a unified pool (serious detection).

Runs ``traffic_combined_rag.build_combined_rag_plan`` (benign easy+HARD + all attack
families) against real Mistral-7B-Instruct-v0.2 over ``internal-handbook-realistic-bd``,
E0 and E1. Scores detection at a realistic operating point — AUC AND TPR@FPR — plus
per-family detection, for frozen v1/v2/v3 (OOD), v4-RAG, and grouped-CV. The HARD
negatives blunt the length artefact so the AUC is a genuine detection signal.

Reproduce (needs GPU; build the backdoor env first):
    python3 tools/forensic/runners/run_rag_realistic_env_real.py --include-backdoor
    python3 tools/forensic/runners/run_combined_rag_detection_real.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import run_rag_realistic_env_real as env


class _PacedClock:
    def __init__(self, step: float = 10.0) -> None:
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _build_retriever():
    from chromadb.utils import embedding_functions
    from forensic.rag_chroma import ChromaRetriever

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=env.EMBEDDING_MODEL_ID)
    return ChromaRetriever(
        persist_directory=str(env.CHROMA_DIR_BD), collection_name=env.COLLECTION_BD,
        embedding_model_id=env.EMBEDDING_MODEL_ID, vector_store_id=env.COLLECTION_BD,
        embedding_function=ef,
    )


def _rank1_by_session(records):
    out = defaultdict(set)
    for r in records:
        if r.get("event_type") != "rag_retrieval":
            continue
        for hit in r["payload"].get("retriever_hits", []):
            if int(hit.get("rank", 0)) == 1 and hit.get("chunk_hash"):
                out[r["session_id"]].add(hit["chunk_hash"])
    return {k: sorted(v) for k, v in out.items()}


def _tpr_at_fpr(scores, y, fam, fpr_target=0.1):
    """TPR at FPR<=target, plus per-family detection at that operating point."""
    neg = sorted((s for s, t in zip(scores, y) if t == 0), reverse=True)
    pos = [(s, f) for s, t, f in zip(scores, y, fam) if t == 1]
    if not neg or not pos:
        return {"tpr": None, "fpr": None, "threshold": None, "per_family": {}}
    k = int(math.floor(fpr_target * len(neg)))
    thr = (neg[0] + 1e-9) if k == 0 else neg[k]
    tp = sum(1 for s, _ in pos if s > thr)
    fp = sum(1 for s in neg if s > thr)
    per_family = {}
    by_fam = defaultdict(list)
    for s, f in pos:
        by_fam[f].append(s)
    for f, ss in by_fam.items():
        per_family[f] = round(sum(1 for s in ss if s > thr) / len(ss), 3)
    return {"tpr": round(tp / len(pos), 3), "fpr": round(fp / len(neg), 3),
            "threshold": round(float(thr), 4), "per_family": per_family}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(env.ROOT))
    ap.add_argument("--n-attack-covers", type=int, default=3)
    ap.add_argument("--fpr", type=float, default=0.1)
    args = ap.parse_args()

    if not (env.CHROMA_DIR_BD.exists() and any(env.CHROMA_DIR_BD.iterdir())):
        print(f"[!] backdoor env missing at {env.CHROMA_DIR_BD}. Build it first:")
        print("    python3 tools/forensic/runners/run_rag_realistic_env_real.py --include-backdoor")
        return 2

    from fastapi.testclient import TestClient
    from forensic.backends_transformers import TransformersBackend
    from forensic.defenses import DefenseConfig
    from forensic.detector_adaptive_rag import AdaptiveDetectorRAG
    from forensic.detector_ml import build_xy, cross_validate_grouped
    from forensic.detector_store import load_scorer
    from forensic.features import build_features
    from forensic.hashing import pseudonymize
    from forensic.mia_score import roc_auc
    from forensic.pile_detector import aggregate_sessions
    from forensic.server import create_app
    from forensic.traffic import assert_no_groundtruth_in_request
    from forensic.traffic_combined_rag import build_combined_rag_plan

    print(f"loading {env.MODEL_ID} (4-bit NF4) ...")
    backend = TransformersBackend(
        model_id=env.MODEL_ID, model_revision=env.MODEL_REVISION, quantization="4bit",
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="float16",
    ).load()
    retriever = _build_retriever()
    plan = build_combined_rag_plan(n_attack_covers=args.n_attack_covers)
    n_atk = sum(1 for c in plan if c.groundtruth["is_attack"])
    print(f"[i] combined campaign: {len(plan)} cases ({n_atk} attack) — E0, E1")

    salt = b"combined-rag-salt-0123456789abc!"
    ev_dir = Path(args.repo_root) / "evidence" / "combined_rag"
    res_dir = Path(args.repo_root) / "results" / "combined_rag"
    ev_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    detectors_dir = Path(args.repo_root) / "results" / "detectors"
    frozen = {n: load_scorer(detectors_dir / f"{n}.joblib")[0]
              for n in ("v1", "v2", "v3") if (detectors_dir / f"{n}.joblib").exists()}

    def run_env(environment):
        run_id = f"cmb-rag-{environment}-{uuid.uuid4().hex[:8]}"
        log_path = str(ev_dir / f"{environment}_{run_id}.jsonl")
        is_e1 = environment == "E1"
        app = create_app(
            log_path=log_path, salt=salt, run_id=run_id, experiment_phase=f"combined_rag_{environment}",
            model_id=env.MODEL_ID, model_revision=env.MODEL_REVISION, model_hash=env.MODEL_ID,
            repo_path=args.repo_root,
            experiment_config={"simulation": "combined_rag", "world": "rag_mistral_realistic_bd",
                               "environment": environment, "groundtruth_separate": True},
            backend=backend, retriever=retriever, environment=environment, system_prompt=None,
            expose_logprobs=not is_e1, output_filtering=is_e1,
            defense_config=DefenseConfig() if is_e1 else None, clock=_PacedClock() if is_e1 else None,
        )
        gt_records = []
        print(f"[i] running combined in {environment} ({len(plan)} cases)…")
        with TestClient(app) as client:
            for case in plan:
                assert_no_groundtruth_in_request(case)
                resp = client.post(case.endpoint, json=case.request_json())
                g = case.groundtruth_json()
                g["session_id"] = pseudonymize(g["session_id"], salt)
                gt_records.append(g)
        return _read(log_path), gt_records

    def analyze(environment, records, gt_records):
        # session -> family label (action family; stealth marked), benign -> "benign"
        fam_by_session = {}
        for g in gt_records:
            sid = g["session_id"]
            if not g.get("is_attack"):
                fam_by_session.setdefault(sid, "benign")
            elif g.get("objective") != "reconnaissance":
                lab = g.get("attack_family", "?")
                if g.get("variant") == "stealth":
                    lab += "/stealth"
                fam_by_session[sid] = lab
        completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                     for r in records if r["event_type"] == "completion"}
        feat_records = [r for r in records if not (
            r["event_type"] == "prompt"
            and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
        srows = aggregate_sessions(build_features(feat_records, gt_records))
        rank1 = _rank1_by_session(records)
        for row in srows:
            row["session_rank1_chunk_hashes"] = rank1.get(str(row.get("session_id")), [])
        y = [int(r.get("label_is_attack")) for r in srows]
        fam = [fam_by_session.get(str(r.get("session_id")), "benign") for r in srows]

        def _detector(scores, name):
            try:
                auc = round(roc_auc(scores, y), 3)
            except Exception as exc:
                return {"auc": f"err:{str(exc)[:50]}"}
            op = _tpr_at_fpr(scores, y, fam, fpr_target=args.fpr)
            return {"auc": auc, **op}

        det = {}
        for name, fn in frozen.items():
            det[f"frozen_{name}"] = _detector([fn(r) for r in srows], name)
        base = frozen.get("v3") or (lambda r: 0.5)
        v4 = AdaptiveDetectorRAG(base)
        v4_scores = [v4.observe_and_score(r)["score"] for r in srows]
        det["v4rag"] = {**_detector(v4_scores, "v4rag"), "refits": v4.n_refits}
        try:
            cv = cross_validate_grouped(build_xy(srows), model_name="logistic", n_splits=5)
            det["grouped_cv"] = _detector(list(cv.oof_y_score), "cv")
        except Exception as exc:
            det["grouped_cv"] = {"auc": f"err:{str(exc)[:50]}"}

        return {"environment": environment, "n_sessions": len(srows), "n_attack": sum(y),
                "n_benign": len(y) - sum(y), "family_counts": dict(Counter(fam)), "detectors": det}

    e0 = analyze("E0", *run_env("E0"))
    e1 = analyze("E1", *run_env("E1"))
    summary = {"status": "ok", "backend": "mistral-7b-instruct-v0.2-4bit",
               "world": "rag_mistral_realistic_bd", "collection": env.COLLECTION_BD,
               "fpr_target": args.fpr, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "campaign_cases": len(plan), "E0": e0, "E1": e1}
    out = res_dir / "combined_detection_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== Combined RAG detection (Mistral, hard negatives) ==========")
    for envn, a in (("E0", e0), ("E1", e1)):
        print(f"[{envn}] sessions={a['n_sessions']} (attack {a['n_attack']}/benign {a['n_benign']}) "
              f"families={a['family_counts']}")
        for dn, d in a["detectors"].items():
            print(f"    {dn}: AUC={d.get('auc')} TPR@FPR{args.fpr}={d.get('tpr')} (fpr={d.get('fpr')}) "
                  f"per_family={d.get('per_family')}")
    print(f"\n[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
