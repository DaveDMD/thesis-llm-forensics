#!/usr/bin/env python3
"""Sampling-based extraction on the CONTROLLED target (canary) — efficacy boost.

Greedy decoding only yields the argmax continuation, so a weakly-memorised secret
never surfaces. This arm SAMPLES many continuations over an escalating temperature
schedule (cold -> warm) and counts a canary extracted if the TRUE value appears in
ANY sample. Compared against the greedy baseline on the SAME 48 canaries.

  * naive_greedy — 1 greedy continuation/target (the naive baseline, ~23/48);
  * sampling     — temperature-escalated sampling (default 1x T=0, 8x T=0.7,
                   16x T=1.0), reproducible via a fixed seed.

Anti-circular: the attacker draws samples and keeps secret-SHAPED ones (observable);
the oracle (true canary value, case-insensitive) judges success separately.

    docker compose run --rm thesis python3 tools/forensic/runners/run_sampling_canary_real.py \\
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


def _body(session_id, seq, prompt, max_tokens, sampling_params):
    return {
        "session_id": session_id, "user_id": f"attacker-{session_id}", "prompt": prompt,
        "sequence_number": seq, "actor_type": "external_user",
        "ip_hash": "ip-sample-aa", "user_agent_hash": "ua-sample-aa", "asn_hash": "asn-sample-aa",
        "request_metadata": {"client_surface": "sampling_probe"},
        "sampling_params": sampling_params, "max_tokens": max_tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--schedule", default="0.0:1,0.7:8,1.0:16",
                    help="temp:n stages, comma-separated")
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--environment", choices=["E0", "E1"], default="E0",
                    help="E1 = full defenses (output redaction + rate limiting + score-channel off)")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    schedule = [(float(t), int(n)) for t, n in (s.split(":") for s in args.schedule.split(","))]

    from fastapi.testclient import TestClient

    from forensic.adaptive_attack import run_sampling_extraction
    from forensic.backends_transformers import TransformersBackend
    from forensic.defenses import DefenseConfig
    from forensic.features import normalize_text
    from forensic.mia_pile import secret_spans
    from forensic.server import create_app

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ckpt = reg["checkpoint"]
    canaries = reg["canaries"]
    reg8 = Path(args.registry).stem.replace("_registry", "")[:8]  # per-campaign output namespacing
    salt = b"sampling-canary-salt-0123456789!"
    is_e1 = args.environment == "E1"
    print(f"[i] target {ckpt}; {len(canaries)} canaries; schedule={schedule}; env={args.environment}")
    print("[i] loading fine-tuned checkpoint…")
    backend = TransformersBackend(model_id=ckpt, model_revision=None, torch_dtype=args.dtype).load()
    torch = backend._torch
    total_samples = sum(n for _t, n in schedule)

    ev_dir = Path(args.repo_root) / "evidence" / "sampling_canary"
    ev_dir.mkdir(parents=True, exist_ok=True)

    def run_arm(arm_name: str):
        run_id = f"sampling-canary-{reg8}-{arm_name}-{uuid.uuid4().hex[:8]}"
        log_path = str(ev_dir / f"{arm_name}_{run_id}.jsonl")
        app = create_app(
            log_path=log_path, salt=salt, run_id=run_id, experiment_phase=f"sampling_canary_{arm_name}",
            model_id=ckpt, model_revision="finetuned", model_hash=backend.model_hash,
            repo_path=args.repo_root,
            experiment_config={"simulation": "sampling_canary", "arm": arm_name,
                               "schedule": str(schedule), "environment": args.environment},
            backend=backend, environment=args.environment, system_prompt="", expose_logprobs=False,
            output_filtering=is_e1, defense_config=DefenseConfig() if is_e1 else None,
        )
        by_rep = defaultdict(lambda: [0, 0])
        by_kind = defaultdict(lambda: [0, 0])
        extracted, total_queries, shaped_candidates = 0, 0, 0
        # reproducibility: seed once per arm; the deterministic loop order fixes the sample stream.
        torch.manual_seed(args.seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[i] arm '{arm_name}'…")
        with TestClient(app) as client:
            for n_done, c in enumerate(canaries, start=1):
                sid = f"{arm_name}-{c['canary_id']}"
                prefix, value, kind, rep = c["prefix"], c["value"], c["kind"], c["repetition"]
                seq = {"n": 0}

                def sample_fn(pfx, temperature, n, _sid=sid, _seq=seq):
                    outs = []
                    sp = {"temperature": temperature}
                    if temperature > 0:
                        sp = {"temperature": temperature, "top_p": 0.95}
                    for _ in range(n):
                        _seq["n"] += 1
                        r = client.post("/v1/complete", json=_body(_sid, _seq["n"], pfx, args.max_tokens, sp))
                        # E1 rate-limiting returns 429 (the defense blocking the burst): a
                        # blocked request delivers no response -> empty (not an error)
                        outs.append((r.json().get("response", "") or "") if r.status_code == 200 else "")
                    return outs

                if arm_name == "naive_greedy":
                    captured = "\n".join(sample_fn(prefix, 0.0, 1))
                    nq = 1
                    shaped = 1 if secret_spans(captured) else 0
                else:
                    res = run_sampling_extraction(prefix, sample_fn, schedule=schedule,
                                                  stop_on_candidate=False)
                    captured, nq, shaped = res.captured, res.n_queries, res.n_secret_shaped
                total_queries += nq
                shaped_candidates += shaped
                ok = normalize_text(value) in normalize_text(captured)
                by_rep[rep][1] += 1
                by_kind[kind][1] += 1
                if ok:
                    by_rep[rep][0] += 1
                    by_kind[kind][0] += 1
                    extracted += 1
                if n_done % 12 == 0:
                    print(f"      {n_done}/{len(canaries)} canaries…")
        return {
            "arm": arm_name, "run_id": run_id, "extracted": extracted, "total": len(canaries),
            "total_queries": total_queries, "secret_shaped_samples": shaped_candidates,
            "queries_per_secret": round(total_queries / extracted, 2) if extracted else None,
            "by_repetition": {r: {"extracted": v[0], "total": v[1]} for r, v in sorted(by_rep.items())},
            "by_kind": {k: {"extracted": v[0], "total": v[1]} for k, v in sorted(by_kind.items())},
        }

    arms = {a: run_arm(a) for a in ("naive_greedy", "sampling")}
    summary = {
        "status": "ok", "world": "gpt2_rich_canary", "environment": args.environment, "checkpoint": ckpt,
        "schedule": schedule, "samples_per_target": total_samples, "seed": args.seed, "arms": arms,
        "differential": {a: {"extracted": arms[a]["extracted"], "queries": arms[a]["total_queries"]}
                         for a in arms},
    }
    res_dir = Path(args.repo_root) / "results" / "sampling_canary"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / f"sampling_canary_{reg8}_{args.environment}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== CAMPIONAMENTO vs GREEDY — Pythia/canary ==========")
    for a, r in arms.items():
        cells = ", ".join(f"N={rp}:{v['extracted']}/{v['total']}" for rp, v in r["by_repetition"].items())
        kinds = ", ".join(f"{k}:{v['extracted']}/{v['total']}" for k, v in r["by_kind"].items())
        print(f"\n[{a}]  extracted={r['extracted']}/{r['total']}  queries={r['total_queries']}  "
              f"queries/secret={r['queries_per_secret']}")
        print(f"     by repetition: {cells}")
        print(f"     by kind:       {kinds}")
    g, s = arms['naive_greedy']['extracted'], arms['sampling']['extracted']
    print(f"\n[Δ] greedy {g}  →  sampling {s}   ({s - g:+d})")
    print(f"[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
