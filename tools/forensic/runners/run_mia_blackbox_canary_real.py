#!/usr/bin/env python3
"""BLACK-BOX (API-only) membership inference vs E0/E1 — honesty add-on.

The white-box MIA (run_mia_comparison) assumes the attacker holds the weights and
is defence-independent (AUC stays at its ceiling under E1). This runner models the
REALISTIC API attacker: it submits each candidate document to the forensic server
and reads back the per-token logprobs of the prompt via the OpenAI-style
``echo_logprobs`` channel, then runs the standard loss-family MIA on them.

The channel is gated by the deployment's score switch:
  * E0 — score channel OPEN: the API echoes prompt logprobs -> loss/zlib/Min-K% MIA
          is feasible (no reference model black-box, so no ``ref``/``min_k_pp``);
  * E1 — score channel CLOSED: logprobs are withheld -> the loss-family MIA has no
          input -> INFEASIBLE (collapses to chance). E1 also rate-limits, but here
          each doc uses its own session (1 request) to ISOLATE the score-channel
          effect from throttling.

    docker compose run --rm -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_CACHE=/workspace/data/mimir-cache \\
        thesis python3 tools/forensic/runners/run_mia_blackbox_canary_real.py \\
        --registry results/canary/<run_id>_registry.json --environment E0
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_CACHE", "/workspace/data/mimir-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

# methods an API attacker can run from echoed per-token logprobs (no 2nd model, no
# per-token distribution stats -> no ref / min_k_pp, which stay white-box only)
BLACKBOX_METHODS = ("loss", "min_k", "zlib")


def _body(session_id, prompt, max_tokens):
    return {
        "session_id": session_id, "user_id": f"miabb-{session_id}", "prompt": prompt,
        "sequence_number": 1, "actor_type": "external_user",
        "ip_hash": "ip-miabb-aa", "user_agent_hash": "ua-miabb-aa", "asn_hash": "asn-miabb-aa",
        "request_metadata": {"client_surface": "mia_probe"},
        "sampling_params": {"temperature": 0.0}, "max_tokens": max_tokens,
        "echo_logprobs": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--environment", choices=["E0", "E1"], default="E0")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from forensic.backends_transformers import TransformersBackend
    from forensic.canary_dataset import mia_texts_from_registry
    from forensic.defenses import DefenseConfig
    from forensic.mia_score import mia_loss, mia_min_k, mia_zlib, roc_auc
    from forensic.mia_strata import _quantile
    from forensic.server import create_app

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ckpt = reg["checkpoint"]
    reg8 = Path(args.registry).stem.replace("_registry", "")[:8]  # per-campaign output namespacing
    mem_texts, non_texts = mia_texts_from_registry(reg)
    docs = [(t, 1) for t in mem_texts] + [(t, 0) for t in non_texts]
    is_e1 = args.environment == "E1"
    print(f"[i] target {ckpt}; env={args.environment}; MIA {len(mem_texts)} member / {len(non_texts)} non-member")

    backend = TransformersBackend(model_id=ckpt, model_revision=None, torch_dtype=args.dtype).load()
    app = create_app(
        log_path=str(Path(args.repo_root) / "evidence" / "mia_blackbox" / f"{reg8}_{args.environment}.jsonl"),
        salt=b"mia-blackbox-salt-0123456789abc!", run_id=f"mia-bb-{reg8}-{args.environment}",
        experiment_phase=f"mia_blackbox_{args.environment}",
        model_id=ckpt, model_revision="finetuned", model_hash=backend.model_hash,
        repo_path=args.repo_root,
        experiment_config={"simulation": "mia_blackbox", "environment": args.environment},
        backend=backend, environment=args.environment, system_prompt="",
        expose_logprobs=not is_e1,                       # E0 opens the score channel, E1 closes it
        output_filtering=is_e1, defense_config=DefenseConfig() if is_e1 else None,
    )
    Path(args.repo_root, "evidence", "mia_blackbox").mkdir(parents=True, exist_ok=True)

    scores = {m: [] for m in BLACKBOX_METHODS}
    labels: list[int] = []
    n_with_logprobs, n_blocked = 0, 0
    print(f"[i] probing {len(docs)} docs through the API (echo_logprobs)…")
    with TestClient(app) as client:
        for i, (text, lab) in enumerate(docs):
            r = client.post("/v1/complete", json=_body(f"miabb-{args.environment}-{i:04d}", text, args.max_tokens))
            if r.status_code != 200:
                n_blocked += 1
                continue
            lp = r.json().get("prompt_token_logprobs")
            if not lp:
                continue
            n_with_logprobs += 1
            s = SimpleNamespace(text=text, token_logprobs=lp)
            scores["loss"].append(mia_loss(s))
            scores["min_k"].append(mia_min_k(s))
            scores["zlib"].append(mia_zlib(s))
            labels.append(lab)
            if (i + 1) % 100 == 0:
                print(f"      {i + 1}/{len(docs)} docs…")

    feasible = n_with_logprobs > 0 and len(set(labels)) == 2
    method_metrics = {}
    if feasible:
        for m in BLACKBOX_METHODS:
            auc = roc_auc(scores[m], labels)
            nonm = [s for s, l in zip(scores[m], labels) if l == 0]
            memb = [s for s, l in zip(scores[m], labels) if l == 1]
            thr = _quantile(nonm, 1.0 - args.fpr) if nonm else float("inf")
            method_metrics[m] = {"auc": round(auc, 4),
                                 "members_confirmed_at_fpr": sum(1 for s in memb if s > thr),
                                 "members": len(memb)}

    summary = {
        "status": "ok", "environment": args.environment, "checkpoint": ckpt,
        "attacker": "black_box_api", "channel": "echo_logprobs",
        "docs_total": len(docs), "docs_with_logprobs": n_with_logprobs, "docs_blocked": n_blocked,
        "feasible": feasible,
        "methods": method_metrics,
        "note": ("E0: API echoes prompt logprobs -> loss-family MIA feasible. "
                 "E1: score channel closed -> no logprobs -> MIA INFEASIBLE (chance). "
                 "ref/min_k_pp are white-box only (need a 2nd model / per-token stats)."),
    }
    res_dir = Path(args.repo_root) / "results" / "mia_blackbox"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / f"mia_blackbox_{reg8}_{args.environment}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n========== MIA BLACK-BOX (API-only) — {args.environment} ==========")
    print(f"  docs con logprob: {n_with_logprobs}/{len(docs)}  (bloccati: {n_blocked})")
    if feasible:
        for m in BLACKBOX_METHODS:
            mm = method_metrics[m]
            print(f"  {m:8} AUC={mm['auc']:.4f}  membri@FPR{int(args.fpr*100)}%={mm['members_confirmed_at_fpr']}/{mm['members']}")
    else:
        print("  INFEASIBLE — canale score chiuso (E1): nessun logprob restituito -> MIA = caso (0.5)")
    print(f"\n[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
