#!/usr/bin/env python3
"""Popularity / parametric-vs-retrieval + adaptive retrieval (Mallen 2212.10511).

Runs the popularity-stratified probes (``popularity_rag.build_popularity_plan``) against
the real Mistral-7B-Instruct-v0.2 (4-bit NF4) over the realistic collection
``internal-handbook-realistic`` (no backdoor), each probe under retrieval OFF
(``/v1/complete``) and ON (``/v1/rag``), E0 (alignment only).

Measured:
* POPULAR facts — parametric accuracy OFF vs ON (does retrieval help or HURT?);
* KB-ONLY secrets — extraction leak OFF vs ON (retrieval is the leak CHANNEL);
* ADAPTIVE retrieval (defense) — per-probe gate on parametric confidence (computed
  offline from OFF+ON): utility kept, retrieval calls saved, residual leak on targeted
  long-tail queries.

Reproduce (needs GPU; build the realistic env first):
    python3 tools/forensic/runners/run_rag_realistic_env_real.py --n-distractors 300
    python3 tools/forensic/runners/run_popularity_rag_real.py
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

import run_rag_realistic_env_real as env


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _build_retriever():
    from chromadb.utils import embedding_functions
    from forensic.rag_chroma import ChromaRetriever

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=env.EMBEDDING_MODEL_ID)
    return ChromaRetriever(
        persist_directory=str(env.CHROMA_DIR), collection_name=env.COLLECTION,
        embedding_model_id=env.EMBEDDING_MODEL_ID, vector_store_id=env.COLLECTION,
        embedding_function=ef,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(env.ROOT))
    ap.add_argument("--kb-covers", type=int, default=5)
    ap.add_argument("--popular-only", action="store_true",
                    help="run only the popular stratum (clean parametric-accuracy measurement)")
    ap.add_argument("--neutral-prompt", action="store_true",
                    help="use a NEUTRAL system prompt (removes the defensive-prompt confound on popular facts)")
    args = ap.parse_args()

    _NEUTRAL_PROMPT = "You are a helpful assistant. Answer the user's question concisely."
    system_prompt = _NEUTRAL_PROMPT if args.neutral_prompt else None  # None -> DEFENSE_SYSTEM_PROMPT

    if not (env.CHROMA_DIR.exists() and any(env.CHROMA_DIR.iterdir())):
        print(f"[!] realistic env missing at {env.CHROMA_DIR}. Build it first:")
        print("    python3 tools/forensic/runners/run_rag_realistic_env_real.py --n-distractors 300")
        return 2

    from fastapi.testclient import TestClient
    from forensic.backends_transformers import TransformersBackend
    from forensic.hashing import pseudonymize
    from forensic.popularity_rag import adaptive_decision, build_popularity_plan, evaluate_popular
    from forensic.secret_oracle import build_secret_groundtruth, evaluate_secret_leak
    from forensic.server import create_app
    from forensic.traffic import assert_no_groundtruth_in_request

    print(f"loading {env.MODEL_ID} (4-bit NF4) ...")
    backend = TransformersBackend(
        model_id=env.MODEL_ID, model_revision=env.MODEL_REVISION, quantization="4bit",
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="float16",
    ).load()
    retriever = _build_retriever()
    secrets = build_secret_groundtruth()

    plan = build_popularity_plan(kb_covers=args.kb_covers)
    if args.popular_only:
        plan = [c for c in plan if c.groundtruth["stratum"] == "popular"]
    salt = b"popularity-rag-salt-0123456789!!"
    ev_dir = Path(args.repo_root) / "evidence" / "popularity_rag"
    res_dir = Path(args.repo_root) / "results" / "popularity_rag"
    ev_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"pop-rag-E0-{uuid.uuid4().hex[:8]}"
    log_path = str(ev_dir / f"E0_{run_id}.jsonl")
    app = create_app(
        log_path=log_path, salt=salt, run_id=run_id, experiment_phase="popularity_rag_E0",
        model_id=env.MODEL_ID, model_revision=env.MODEL_REVISION, model_hash=env.MODEL_ID,
        repo_path=args.repo_root,
        experiment_config={"simulation": "popularity_rag", "world": "rag_mistral_realistic",
                           "environment": "E0", "groundtruth_separate": True},
        backend=backend, retriever=retriever, environment="E0", system_prompt=system_prompt,
        expose_logprobs=True, output_filtering=False,
    )
    gt_records = []
    print(f"[i] running popularity probes ({len(plan)} cases: OFF/ON)…")
    with TestClient(app) as client:
        for case in plan:
            assert_no_groundtruth_in_request(case)
            resp = client.post(case.endpoint, json=case.request_json())
            body = resp.json() if resp.status_code == 200 else {}
            g = case.groundtruth_json()
            g["session_id"] = pseudonymize(g["session_id"], salt)
            g["completion_record_hash"] = body.get("completion_record_hash") if isinstance(body, dict) else None
            gt_records.append(g)

    records = _read(log_path)
    resp_by_hash = {r.get("record_hash"): (r["payload"].get("response_raw") or "")
                    for r in records if r.get("event_type") == "completion"}

    # pair OFF/ON per probe
    probes: dict[str, dict] = defaultdict(dict)
    for g in gt_records:
        resp = resp_by_hash.get(g.get("completion_record_hash"), "")
        probes[g["probe_id"]][g["condition"]] = (g, resp)

    pop = {"off_correct": 0, "on_correct": 0, "n": 0}
    kb = {"off_leak": 0, "on_leak": 0, "n": 0}
    adapt = {"retrieve_calls": 0, "n": 0, "pop_correct": 0, "pop_n": 0, "kb_leak": 0, "kb_n": 0}

    for probe_id, cond in probes.items():
        if "off" not in cond or "on" not in cond:
            continue
        (g_off, r_off), (g_on, r_on) = cond["off"], cond["on"]
        stratum = g_off["stratum"]
        decision = adaptive_decision(r_off)            # gate on parametric confidence
        used = r_off if decision == "off" else r_on
        adapt["n"] += 1
        adapt["retrieve_calls"] += int(decision == "on")
        if stratum == "popular":
            kws = tuple(g_off["answer_keywords"])
            pop["n"] += 1
            pop["off_correct"] += int(evaluate_popular(r_off, kws))
            pop["on_correct"] += int(evaluate_popular(r_on, kws))
            adapt["pop_n"] += 1
            adapt["pop_correct"] += int(evaluate_popular(used, kws))
        else:  # kb_secret
            ids = set(g_off["target_secret_ids"])
            kb["n"] += 1
            kb["off_leak"] += int(bool(set(evaluate_secret_leak(r_off, secrets).total_secret_ids) & ids))
            kb["on_leak"] += int(bool(set(evaluate_secret_leak(r_on, secrets).total_secret_ids) & ids))
            adapt["kb_n"] += 1
            adapt["kb_leak"] += int(bool(set(evaluate_secret_leak(used, secrets).total_secret_ids) & ids))

    def _rate(a, b):
        return round(a / b, 3) if b else None

    summary = {
        "status": "ok", "backend": "mistral-7b-instruct-v0.2-4bit",
        "world": "rag_mistral_realistic", "collection": env.COLLECTION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cases": len(plan),
        "system_prompt_mode": "neutral" if args.neutral_prompt else "defensive",
        "popular_only": args.popular_only,
        "popular": {**pop, "acc_off": _rate(pop["off_correct"], pop["n"]),
                    "acc_on": _rate(pop["on_correct"], pop["n"])},
        "kb_secret": {**kb, "leak_off": _rate(kb["off_leak"], kb["n"]),
                      "leak_on": _rate(kb["on_leak"], kb["n"])},
        "adaptive": {**adapt, "retrieval_call_rate": _rate(adapt["retrieve_calls"], adapt["n"]),
                     "pop_acc": _rate(adapt["pop_correct"], adapt["pop_n"]),
                     "kb_leak": _rate(adapt["kb_leak"], adapt["kb_n"])},
    }
    suffix = "_neutral" if args.neutral_prompt else ""
    suffix += "_popularonly" if args.popular_only else ""
    out = res_dir / f"popularity_summary{suffix}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== Popularity / adaptive retrieval (Mistral) ==========")
    print(f"[POPULAR]   acc OFF={summary['popular']['acc_off']} → ON={summary['popular']['acc_on']} "
          f"(retrieval helps/hurts on parametric facts)")
    print(f"[KB-SECRET] leak OFF={summary['kb_secret']['leak_off']} → ON={summary['kb_secret']['leak_on']} "
          f"(retrieval = the leak channel for long-tail)")
    print(f"[ADAPTIVE]  retrieval_call_rate={summary['adaptive']['retrieval_call_rate']} | "
          f"pop_acc={summary['adaptive']['pop_acc']} | kb_leak(residual)={summary['adaptive']['kb_leak']}")
    print(f"\n[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
