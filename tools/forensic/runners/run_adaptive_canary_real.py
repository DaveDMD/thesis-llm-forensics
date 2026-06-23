#!/usr/bin/env python3
"""Adaptive multi-turn extraction on the CONTROLLED target (canary).

Compares three arms on the SAME 48 canaries, same fine-tuned Pythia-410m, through
the forensic server (perfect ground truth):
  * naive       — 1 query/target, M tokens (the naive baseline, ~23/48);
  * naive_long  — 1 query/target, budget*M tokens (controls for "just more tokens");
  * adaptive    — multi-turn policy (chain/refine/stop), <=budget queries/target.

If adaptive > naive_long, the value is the ADAPTIVITY (reading turn N-1 to build
turn N: chaining past the truncation, refining a sterile prefix), not the token
count. Anti-circular: the attacker adapts only on observable signals; the oracle
(true canary value, case-insensitive) judges success separately.

    docker compose run --rm thesis python3 tools/forensic/runners/run_adaptive_canary_real.py \\
        --registry results/canary/<run_id>_registry.json
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_CACHE", "/workspace/data/mimir-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path


def _body(session_id, seq, prompt, max_tokens):
    return {
        "session_id": session_id, "user_id": f"attacker-{session_id}", "prompt": prompt,
        "sequence_number": seq, "actor_type": "external_user",
        "ip_hash": "ip-adapt-aa", "user_agent_hash": "ua-adapt-aa", "asn_hash": "asn-adapt-aa",
        "request_metadata": {"client_surface": "adaptive_probe"},
        "sampling_params": {"temperature": 0.0}, "max_tokens": max_tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--environment", choices=["E0", "E1"], default="E0",
                    help="E1 turns on output filtering + defenses (blocks secret delivery)")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from forensic.adaptive_attack import run_adaptive_extraction
    from forensic.backends_transformers import TransformersBackend
    from forensic.defenses import DefenseConfig
    from forensic.features import normalize_text
    from forensic.server import create_app

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ckpt = reg["checkpoint"]
    canaries = reg["canaries"]
    reg8 = Path(args.registry).stem.replace("_registry", "")[:8]  # per-campaign output namespacing
    salt = b"adaptive-canary-salt-0123456789!"
    is_e1 = args.environment == "E1"
    print(f"[i] target {ckpt}; {len(canaries)} canaries; budget={args.budget}, "
          f"M={args.max_tokens}, env={args.environment}")
    print("[i] loading fine-tuned checkpoint…")
    backend = TransformersBackend(model_id=ckpt, model_revision=None, torch_dtype=args.dtype).load()

    ev_dir = Path(args.repo_root) / "evidence" / "adaptive_canary"
    ev_dir.mkdir(parents=True, exist_ok=True)
    long_tokens = args.budget * args.max_tokens

    def run_arm(arm_name: str):
        run_id = f"adaptive-canary-{reg8}-{arm_name}-{uuid.uuid4().hex[:8]}"
        log_path = str(ev_dir / f"{arm_name}_{run_id}.jsonl")
        app = create_app(
            log_path=log_path, salt=salt, run_id=run_id, experiment_phase=f"adaptive_canary_{arm_name}",
            model_id=ckpt, model_revision="finetuned", model_hash=backend.model_hash,
            repo_path=args.repo_root,
            experiment_config={"simulation": "adaptive_canary", "arm": arm_name,
                               "budget": args.budget, "environment": args.environment},
            backend=backend, environment=args.environment, system_prompt="",
            expose_logprobs=not is_e1, output_filtering=is_e1,
            defense_config=DefenseConfig() if is_e1 else None,
        )
        by_rep = defaultdict(lambda: [0, 0])
        by_kind = defaultdict(lambda: [0, 0])
        extracted, total_queries = 0, 0
        print(f"[i] arm '{arm_name}'…")
        with TestClient(app) as client:
            for c in canaries:
                sid = f"{arm_name}-{c['canary_id']}"
                prefix, value, kind, rep = c["prefix"], c["value"], c["kind"], c["repetition"]
                if arm_name == "adaptive":
                    seq = {"n": 0}
                    def q(context, mtok, _sid=sid, _seq=seq):
                        _seq["n"] += 1
                        r = client.post("/v1/complete", json=_body(_sid, _seq["n"], context, mtok))
                        r.raise_for_status()
                        return r.json().get("response", "") or "", None
                    res = run_adaptive_extraction(prefix, q, budget=args.budget, max_tokens=args.max_tokens)
                    captured, nq = res.captured, res.n_queries
                else:
                    mtok = long_tokens if arm_name == "naive_long" else args.max_tokens
                    r = client.post("/v1/complete", json=_body(sid, 1, prefix, mtok))
                    r.raise_for_status()
                    captured, nq = r.json().get("response", "") or "", 1
                total_queries += nq
                ok = normalize_text(value) in normalize_text(captured)
                by_rep[rep][1] += 1
                by_kind[kind][1] += 1
                if ok:
                    by_rep[rep][0] += 1
                    by_kind[kind][0] += 1
                    extracted += 1
        return {
            "arm": arm_name, "run_id": run_id, "extracted": extracted, "total": len(canaries),
            "total_queries": total_queries,
            "queries_per_secret": round(total_queries / extracted, 2) if extracted else None,
            "by_repetition": {r: {"extracted": v[0], "total": v[1]} for r, v in sorted(by_rep.items())},
            "by_kind": {k: {"extracted": v[0], "total": v[1]} for k, v in sorted(by_kind.items())},
        }

    arms = {a: run_arm(a) for a in ("naive", "naive_long", "adaptive")}

    summary = {
        "status": "ok", "world": "pythia_canary", "checkpoint": ckpt,
        "environment": args.environment,
        "budget": args.budget, "max_tokens": args.max_tokens, "arms": arms,
        "differential": {a: {"extracted": arms[a]["extracted"], "queries": arms[a]["total_queries"]}
                         for a in arms},
    }
    res_dir = Path(args.repo_root) / "results" / "adaptive_canary"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / f"adaptive_canary_{reg8}_{args.environment}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== ADATTIVO vs NAIVE — Pythia/canary (estrazione) ==========")
    for a, r in arms.items():
        cells = ", ".join(f"N={rp}:{v['extracted']}/{v['total']}" for rp, v in r["by_repetition"].items())
        kinds = ", ".join(f"{k}:{v['extracted']}/{v['total']}" for k, v in r["by_kind"].items())
        print(f"\n[{a}]  extracted={r['extracted']}/{r['total']}  queries={r['total_queries']}  "
              f"queries/secret={r['queries_per_secret']}")
        print(f"     by repetition: {cells}")
        print(f"     by kind:       {kinds}")
    n, nl, ad = arms['naive']['extracted'], arms['naive_long']['extracted'], arms['adaptive']['extracted']
    print(f"\n[Δ] naive {n}  →  naive_long {nl}  →  adaptive {ad}   "
          f"(adaptive vs naive_long = {ad - nl:+d}: the gain from adaptivity beyond tokens)")
    print(f"[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
