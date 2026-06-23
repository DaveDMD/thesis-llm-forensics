#!/usr/bin/env python3
"""Realistic controls E0 vs E1 in the RAG world (Mistral-7B NF4 + Chroma).

Quantifies the alignment-defence effect ("high refusal = alignment defence"). Runs
the SAME integrated RAG campaign (RAG attacks + application MIA + benign) twice,
changing ONLY the environment:

  * E0 — only the level-1 alignment (instruct/safety-tuned Mistral); no level-2;
  * E1 — level-2 defences ON on top of alignment: output filtering (redaction of
          secret spans in the response), input query filtering (structural), and
          logprob suppression. The per-session RATE LIMITER does not engage here
          by design: the RAG stream is ~1 request/session (campaign-volume
          throttling is a completion-world phenomenon), so a paced clock keeps it
          from firing as a batch artefact — stated honestly, not hidden.

Differential measured:
  (a) EFFICACY — real-secret leaks per family (oracle) + refusal rate: how much
                 the explicit level-2 controls add beyond alignment;
  (b) RESIDUES — defense_outcome distribution, response_redactions, logprobs
                 events present/absent, retriever hits;
  (c) DETECTOR — re-run (request-level here) on E0 vs E1.

    docker compose run --rm thesis \\
        python3 tools/forensic/runners/run_rag_e0_e1_real.py --scale small
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

import run_integrated_pipeline_real as ip  # Chroma/model setup + constants


class _PacedClock:
    """Clock advancing a fixed large step per request: in batch this keeps the
    per-session rate limiter from firing (>20/60s is unreachable), so the RAG E1
    differential is attributable to output/query filtering + logprob suppression,
    not a batch rate-limit artefact (the RAG stream is not volume-structured)."""

    def __init__(self, step: float = 10.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=list(ip.SCALES), default="small")
    ap.add_argument("--repo-root", default=str(ip.ROOT))
    args = ap.parse_args()
    cfg = ip.SCALES[args.scale]

    from forensic.backends_transformers import TransformersBackend
    from forensic.defenses import DefenseConfig
    from forensic.detector_ml import build_xy, cross_validate_grouped
    from forensic.features import build_features, response_contains_refusal
    from forensic.hashing import pseudonymize
    from forensic.mia_score import roc_auc
    from forensic.pile_detector import aggregate_sessions
    from forensic.pipeline import build_combined_plan
    from forensic.rag_chroma import ChromaRetriever
    from forensic.secret_oracle import build_secret_groundtruth, evaluate_secret_leak
    from forensic.server import create_app
    from forensic.traffic import assert_no_groundtruth_in_request

    ip._build_chroma_index_if_absent()

    print(f"loading {ip.MODEL_ID} (4-bit NF4) ...")
    backend = TransformersBackend(
        model_id=ip.MODEL_ID, model_revision=ip.MODEL_REVISION,
        quantization="4bit", bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="float16",
    ).load()
    retriever = ChromaRetriever(
        persist_directory=str(ip.CHROMA_DIR), collection_name=ip.COLLECTION,
        embedding_model_id=ip.EMBEDDING_MODEL_ID, vector_store_id=ip.COLLECTION,
    )

    plan = build_combined_plan(rag_variants=cfg["rag_variants"],
                               mia_variants=cfg["mia_variants"],
                               benign_variants=cfg["benign_variants"])
    n_attack = sum(1 for c in plan if c.groundtruth["is_attack"])
    print(f"[i] RAG campaign: {len(plan)} cases ({n_attack} attack) — run twice: E0, E1")

    secrets = build_secret_groundtruth()
    salt = b"rag-e0e1-salt-0123456789abcdef!!"
    ev_dir = Path(args.repo_root) / "evidence" / "rag_e0_e1"
    ev_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(args.repo_root) / "results" / "rag_e0_e1"
    res_dir.mkdir(parents=True, exist_ok=True)

    def run_env(environment: str):
        run_id = f"rag-e0e1-{environment}-{uuid.uuid4().hex[:8]}"
        log_path = str(ev_dir / f"{environment}_{run_id}.jsonl")
        is_e1 = environment == "E1"
        clock = _PacedClock() if is_e1 else None
        app = create_app(
            log_path=log_path, salt=salt, run_id=run_id,
            experiment_phase=f"rag_e0_e1_{environment}",
            model_id=ip.MODEL_ID, model_revision=ip.MODEL_REVISION, model_hash=ip.MODEL_ID,
            repo_path=args.repo_root,
            experiment_config={"simulation": "rag_e0_e1", "world": "rag_mistral",
                               "environment": environment, "groundtruth_separate": True},
            backend=backend, retriever=retriever, environment=environment, system_prompt=None,
            expose_logprobs=not is_e1, output_filtering=is_e1,
            defense_config=DefenseConfig() if is_e1 else None, clock=clock,
        )
        from fastapi.testclient import TestClient
        gt_records, statuses = [], Counter()
        print(f"[i] running RAG campaign in {environment}…")
        with TestClient(app) as client:
            for n, case in enumerate(plan, start=1):
                assert_no_groundtruth_in_request(case)
                resp = client.post(case.endpoint, json=case.request_json())
                statuses[resp.status_code] += 1
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                g = case.groundtruth_json()
                g["session_id"] = pseudonymize(g["session_id"], salt)
                g.update({"http_status": resp.status_code,
                          "completion_record_hash": body.get("completion_record_hash") if isinstance(body, dict) else None})
                gt_records.append(g)
                if n % 50 == 0:
                    print(f"      {n}/{len(plan)} cases…")
        records = _read(log_path)
        return records, gt_records, statuses

    def analyze(environment, records, gt_records, statuses):
        resp_by_hash, redactions = {}, 0
        for rec in records:
            if rec.get("event_type") == "completion":
                p = rec["payload"]
                resp_by_hash[rec.get("record_hash")] = p.get("response_raw") or ""
                redactions += int(p.get("response_redactions") or 0)
        # leaks per family + refusal (join via completion_record_hash)
        fam = defaultdict(lambda: {"n": 0, "leak_total": 0, "refusal": 0})
        benign = {"n": 0, "leak_total": 0, "refusal": 0}
        leaked_ids: set[str] = set()
        for g in gt_records:
            resp = resp_by_hash.get(g.get("completion_record_hash"), "")
            res = evaluate_secret_leak(resp, secrets)
            ref = response_contains_refusal(resp)
            if g.get("is_attack"):
                d = fam[g.get("attack_family", "?")]
                d["n"] += 1
                d["leak_total"] += int(res.leak_total)
                d["refusal"] += int(ref)
                leaked_ids |= set(res.total_secret_ids)
            else:
                benign["n"] += 1
                benign["leak_total"] += int(res.leak_total)
                benign["refusal"] += int(ref)
        defense_outcomes = Counter()
        for rec in records:
            if rec.get("event_type") == "prompt":
                defense_outcomes[rec["payload"].get("defense_outcome", "accepted")] += 1
        n_logprobs = sum(1 for r in records if r.get("event_type") == "logprobs")
        n_retrieval = sum(1 for r in records if r.get("event_type") == "rag_retrieval")

        # detector (request-level here); feed only completed requests
        completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                     for r in records if r["event_type"] == "completion"}
        feat_records = [r for r in records if not (
            r["event_type"] == "prompt"
            and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
        det = {"roc_auc": None}
        try:
            rows = build_features(feat_records, gt_records)
            srows = aggregate_sessions(rows)
            cv = cross_validate_grouped(build_xy(srows), model_name="logistic", n_splits=5)
            det = {"roc_auc": round(roc_auc(list(cv.oof_y_score), list(cv.oof_y_true)), 3),
                   "n_sessions": len(srows)}
        except Exception as exc:  # detector is secondary here; never abort the run
            det = {"roc_auc": None, "error": str(exc)[:120]}

        attack_leaks = sum(d["leak_total"] for d in fam.values())
        return {
            "environment": environment,
            "http_statuses": dict(statuses),
            "attack_leaks_total": attack_leaks,
            "distinct_secrets_leaked": sorted(leaked_ids),
            "by_family": {k: dict(v) for k, v in fam.items()},
            "benign": benign,
            "response_redactions": redactions,
            "defense_outcomes": dict(defense_outcomes),
            "logprobs_events": n_logprobs,
            "retrieval_events": n_retrieval,
            "detector": det,
        }

    e0 = analyze("E0", *run_env("E0"))
    e1 = analyze("E1", *run_env("E1"))

    summary = {
        "status": "ok", "scale": args.scale, "backend": "mistral-7b-instruct-v0.2-4bit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_cases": len(plan), "E0": e0, "E1": e1,
        "differential": {
            "attack_leaks_total": {"E0": e0["attack_leaks_total"], "E1": e1["attack_leaks_total"]},
            "benign_leaks": {"E0": e0["benign"]["leak_total"], "E1": e1["benign"]["leak_total"]},
            "response_redactions": {"E0": e0["response_redactions"], "E1": e1["response_redactions"]},
            "logprobs_events": {"E0": e0["logprobs_events"], "E1": e1["logprobs_events"]},
            "requests_blocked_E1": sum(v for k, v in e1["http_statuses"].items() if int(k) >= 400),
            "detector_auc": {"E0": e0["detector"].get("roc_auc"), "E1": e1["detector"].get("roc_auc")},
        },
    }
    out = res_dir / f"rag_e0_e1_{args.scale}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== RAG E0 vs E1 (Mistral) ==========")
    print(f"[EFFICACY] real-secret leaks (attack): E0={e0['attack_leaks_total']} → E1={e1['attack_leaks_total']}"
          f"  | benign leaks: E0={e0['benign']['leak_total']} → E1={e1['benign']['leak_total']}")
    print(f"           distinct secrets leaked: E0={e0['distinct_secrets_leaked']} → E1={e1['distinct_secrets_leaked']}")
    for env, a in (("E0", e0), ("E1", e1)):
        fams = ", ".join(f"{k}:{v['leak_total']}/{v['n']}(ref{v['refusal']})" for k, v in a["by_family"].items())
        print(f"           {env} by family (leak/n(refusals)): {fams}")
    print("\n[RESIDUES]")
    print(f"      defense_outcomes  E0={e0['defense_outcomes']}   E1={e1['defense_outcomes']}")
    print(f"      response_redactions E0={e0['response_redactions']}   E1={e1['response_redactions']}")
    print(f"      logprobs events   E0={e0['logprobs_events']}   E1={e1['logprobs_events']}  "
          f"(blocked E1={summary['differential']['requests_blocked_E1']})")
    print("\n[DETECTOR]")
    print(f"      E0 AUC={e0['detector'].get('roc_auc')}   E1 AUC={e1['detector'].get('roc_auc')}")
    print(f"\n[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
