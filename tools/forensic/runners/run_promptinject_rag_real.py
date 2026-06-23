#!/usr/bin/env python3
"""PromptInject (goal hijacking + prompt leaking) on the realistic RAG env.

Runs the deterministic PromptInject battery (``promptinject_rag.build_promptinject_plan``)
against the REAL Mistral-7B-Instruct-v0.2 (4-bit NF4) over the SEPARATE realistic
Chroma collection built by ``run_rag_realistic_env_real``, twice — once in E0
(level-1 alignment only) and once in E1 (level-2 defences ON) — and scores attack
success with the OFFLINE ORACLES only:

  * goal hijacking — ``evaluate_goal_hijack`` (normalised exact rogue-string match);
  * prompt leaking — ``secret_oracle.evaluate_secret_leak`` for the system-prompt
    secret (``secret-operator-console``), which the level-1 ``DEFENSE_SYSTEM_PROMPT``
    carries and the server injects on every request.

Anti-circularity: the oracle verdicts and the rogue string / target secret live in
the ground truth only; the detector (grouped-CV AUC, secondary here) never sees
them, and the injection template is never used as a feature.

Reproduce (needs GPU; build the env first):
    docker compose run --rm -e HF_HUB_OFFLINE=1 \\
        -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic thesis \\
        python3 tools/forensic/runners/run_rag_realistic_env_real.py --n-distractors 300
    docker compose run --rm -e HF_HUB_OFFLINE=1 \\
        -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic thesis \\
        python3 tools/forensic/runners/run_promptinject_rag_real.py
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

import run_rag_realistic_env_real as env  # realistic env constants (collection/persist/model)


class _PacedClock:
    """Large fixed step per request so the per-session rate limiter never fires in
    batch: the E1 differential is attributable to output/query filtering + logprob
    suppression, not a batch rate-limit artefact (the RAG stream is not volume-structured)."""

    def __init__(self, step: float = 10.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _wilson(success: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a proportion (citable ASR with uncertainty)."""
    if n == 0:
        return [0.0, 0.0]
    phat = success / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def _build_retriever():
    from chromadb.utils import embedding_functions
    from forensic.rag_chroma import ChromaRetriever

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=env.EMBEDDING_MODEL_ID)
    return ChromaRetriever(
        persist_directory=str(env.CHROMA_DIR),
        collection_name=env.COLLECTION,
        embedding_model_id=env.EMBEDDING_MODEL_ID,
        vector_store_id=env.COLLECTION,
        embedding_function=ef,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(env.ROOT))
    ap.add_argument("--n-covers", type=int, default=10,
                    help="cover framings per variant → N per ASR cell")
    ap.add_argument("--n-benign", type=int, default=10, help="benign multi-turn sessions")
    args = ap.parse_args()

    if not (env.CHROMA_DIR.exists() and any(env.CHROMA_DIR.iterdir())):
        print(f"[!] realistic Chroma env missing at {env.CHROMA_DIR}. Build it first:")
        print("    python3 tools/forensic/runners/run_rag_realistic_env_real.py --n-distractors 300")
        return 2

    from fastapi.testclient import TestClient

    from forensic.backends_transformers import TransformersBackend
    from forensic.defenses import DefenseConfig
    from forensic.detector_ml import build_xy, cross_validate_grouped
    from forensic.features import build_features, response_contains_refusal
    from forensic.hashing import pseudonymize
    from forensic.mia_score import roc_auc
    from forensic.pile_detector import aggregate_sessions
    from forensic.promptinject_rag import build_promptinject_plan, evaluate_goal_hijack
    from forensic.secret_oracle import (
        SYSTEM_PROMPT_SECRET_ID,
        build_secret_groundtruth,
        evaluate_secret_leak,
    )
    from forensic.server import create_app
    from forensic.traffic import assert_no_groundtruth_in_request

    print(f"loading {env.MODEL_ID} (4-bit NF4) ...")
    backend = TransformersBackend(
        model_id=env.MODEL_ID, model_revision=env.MODEL_REVISION,
        quantization="4bit", bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="float16",
    ).load()
    retriever = _build_retriever()

    plan = build_promptinject_plan(n_covers=args.n_covers, n_benign_sessions=args.n_benign)
    n_attack = sum(1 for c in plan if c.groundtruth["is_attack"])
    print(f"[i] PromptInject campaign: {len(plan)} cases ({n_attack} attack), "
          f"n_covers={args.n_covers} — run twice: E0, E1")

    secrets = build_secret_groundtruth()
    salt = b"promptinject-rag-salt-0123456789"
    ev_dir = Path(args.repo_root) / "evidence" / "promptinject_rag"
    res_dir = Path(args.repo_root) / "results" / "promptinject_rag"
    ev_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    def run_env(environment: str):
        run_id = f"pi-rag-{environment}-{uuid.uuid4().hex[:8]}"
        log_path = str(ev_dir / f"{environment}_{run_id}.jsonl")
        is_e1 = environment == "E1"
        app = create_app(
            log_path=log_path, salt=salt, run_id=run_id,
            experiment_phase=f"promptinject_rag_{environment}",
            model_id=env.MODEL_ID, model_revision=env.MODEL_REVISION, model_hash=env.MODEL_ID,
            repo_path=args.repo_root,
            experiment_config={"simulation": "promptinject_rag", "world": "rag_mistral_realistic",
                               "environment": environment, "groundtruth_separate": True},
            backend=backend, retriever=retriever, environment=environment,
            system_prompt=None,  # -> DEFENSE_SYSTEM_PROMPT (carries secret-operator-console)
            expose_logprobs=not is_e1, output_filtering=is_e1,
            defense_config=DefenseConfig() if is_e1 else None,
            clock=_PacedClock() if is_e1 else None,
        )
        gt_records, statuses = [], Counter()
        print(f"[i] running PromptInject in {environment}…")
        with TestClient(app) as client:
            for n, case in enumerate(plan, start=1):
                assert_no_groundtruth_in_request(case)
                resp = client.post(case.endpoint, json=case.request_json())
                statuses[resp.status_code] += 1
                body = resp.json() if resp.status_code == 200 else {}
                g = case.groundtruth_json()
                g["session_id"] = pseudonymize(g["session_id"], salt)
                g.update({"http_status": resp.status_code,
                          "completion_record_hash": body.get("completion_record_hash")
                          if isinstance(body, dict) else None})
                gt_records.append(g)
        return _read(log_path), gt_records, statuses

    def analyze(environment, records, gt_records, statuses):
        resp_by_hash, redactions = {}, 0
        for rec in records:
            if rec.get("event_type") == "completion":
                p = rec["payload"]
                resp_by_hash[rec.get("record_hash")] = p.get("response_raw") or ""
                redactions += int(p.get("response_redactions") or 0)

        # per (family, variant): attack-success rate via the oracles + refusal rate.
        # Reconnaissance turns are benign-equivalent probing (textually identical to
        # benign coverage); they are part of the attacker SESSION for the detector
        # but are NOT an ASR action, so they are excluded from the success cells.
        cells = defaultdict(lambda: {"n": 0, "success": 0, "refusal": 0})
        for g in gt_records:
            if not g.get("is_attack") or g.get("objective") == "reconnaissance":
                continue
            resp = resp_by_hash.get(g.get("completion_record_hash"), "")
            fam = g.get("attack_family", "?")
            key = (fam, g.get("variant", "-"))
            cells[key]["n"] += 1
            cells[key]["refusal"] += int(response_contains_refusal(resp))
            if fam == "promptinject_goal_hijack":
                ok = evaluate_goal_hijack(resp, g.get("rogue_string"))
            elif fam == "promptinject_prompt_leak":
                res = evaluate_secret_leak(resp, secrets)
                ok = SYSTEM_PROMPT_SECRET_ID in res.total_secret_ids
            else:
                ok = False
            cells[key]["success"] += int(ok)

        # benign false-positive control (any secret surfacing in a benign answer)
        benign = {"n": 0, "leak": 0, "refusal": 0}
        for g in gt_records:
            if g.get("is_attack"):
                continue
            resp = resp_by_hash.get(g.get("completion_record_hash"), "")
            benign["n"] += 1
            benign["leak"] += int(evaluate_secret_leak(resp, secrets).leak_total)
            benign["refusal"] += int(response_contains_refusal(resp))

        defense_outcomes = Counter()
        for rec in records:
            if rec.get("event_type") == "prompt":
                defense_outcomes[rec["payload"].get("defense_outcome", "accepted")] += 1
        n_logprobs = sum(1 for r in records if r.get("event_type") == "logprobs")

        # secondary detector: grouped-CV AUC (generalisation, never sees the oracle)
        completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                     for r in records if r["event_type"] == "completion"}
        feat_records = [r for r in records if not (
            r["event_type"] == "prompt"
            and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
        det = {"roc_auc": None}
        try:
            srows = aggregate_sessions(build_features(feat_records, gt_records))
            cv = cross_validate_grouped(build_xy(srows), model_name="logistic", n_splits=5)
            det = {"roc_auc": round(roc_auc(list(cv.oof_y_score), list(cv.oof_y_true)), 3),
                   "n_sessions": len(srows)}
        except Exception as exc:  # detector is secondary; never abort the run
            det = {"roc_auc": None, "error": str(exc)[:120]}

        return {
            "environment": environment,
            "http_statuses": dict(statuses),
            "by_cell": {
                f"{fam}/{var}": {
                    **v,
                    "asr": round(v["success"] / v["n"], 3) if v["n"] else None,
                    "asr_ci95": _wilson(v["success"], v["n"]),
                }
                for (fam, var), v in cells.items()
            },
            "benign": benign,
            "response_redactions": redactions,
            "defense_outcomes": dict(defense_outcomes),
            "logprobs_events": n_logprobs,
            "detector": det,
        }

    e0 = analyze("E0", *run_env("E0"))
    e1 = analyze("E1", *run_env("E1"))

    summary = {
        "status": "ok", "backend": "mistral-7b-instruct-v0.2-4bit",
        "world": "rag_mistral_realistic", "collection": env.COLLECTION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_cases": len(plan), "E0": e0, "E1": e1,
    }
    out = res_dir / "promptinject_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== PromptInject RAG (Mistral, realistic env) ==========")
    for envn, a in (("E0", e0), ("E1", e1)):
        print(f"[{envn}] by family/variant (ASR=success/n, CI95, refusals):")
        for k, v in a["by_cell"].items():
            print(f"      {k}: {v['success']}/{v['n']} ASR={v['asr']} CI{v['asr_ci95']} (ref {v['refusal']})")
        print(f"      benign: leak {a['benign']['leak']}/{a['benign']['n']} | "
              f"redactions {a['response_redactions']} | detector AUC {a['detector'].get('roc_auc')}")
    print(f"\n[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
