#!/usr/bin/env python3
"""Systematic MIA method comparison on the controlled target — honesty add-on.

Membership inference is FAR easier than verbatim extraction (a diffuse loss signal
vs peaked token-exact memorisation), and the white-box AUC is an UPPER BOUND: it
assumes the attacker has the weights (exact per-token loss). This runner makes the
picture defensible by:

  * comparing EVERY score-based MIA on the SAME member/non-member docs, so we see
    which discriminates best:
      - ``loss``     mean token log-prob (raw NLL)                  [Yeom 2018]
      - ``min_k``    mean of the K% least-likely tokens (K=20%)     [Shi 2023]
      - ``min_k_pp`` Min-K%++ (z-scored per-token)                  [Zhang 2024]
      - ``zlib``     loss calibrated by zlib-compressed length      [Carlini 2021]
      - ``ref``      loss calibrated by a REFERENCE model's loss    [Carlini 2021]
                     (reference = the *base* gpt2-medium, pre-fine-tune)
  * separating WHITE-BOX (model access -> all methods) from BLACK-BOX (API only):
    the black-box attacker can use only logprob-derived scores (loss/min_k/zlib),
    has NO reference model, and is DEFEATED when the score channel is closed (E1:
    logprob events 952 -> 0). The realistic black-box best is therefore the best
    *uncalibrated* method, which is strictly <= the white-box best (which can
    calibrate with the reference). Run it on both the 12-epoch and 4-epoch
    checkpoints to expose the over-training gradient (more epochs -> inflated MIA).

    docker compose run --rm -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_CACHE=/workspace/data/mimir-cache \\
        thesis python3 tools/forensic/runners/run_mia_comparison_canary_real.py \\
        --registry results/canary/<run_id>_registry.json --label 12ep
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_CACHE", "/workspace/data/mimir-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
from pathlib import Path

# which methods an API-only (black-box) attacker can still run: logprob-derived,
# no second model. ``ref`` needs the reference model -> white-box only.
BLACKBOX_METHODS = ("loss", "min_k", "min_k_pp", "zlib")
WHITEBOX_ONLY = ("ref",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="results/canary/<run_id>_registry.json")
    ap.add_argument("--ref-model", default="gpt2-medium", help="reference (base) model id")
    ap.add_argument("--ref-revision", default="main")
    ap.add_argument("--label", default="", help="tag for the output (e.g. 12ep / 4ep)")
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from forensic.backends_transformers import TransformersBackend
    from forensic.canary_dataset import mia_texts_from_registry
    from forensic.mia_score import (
        mia_loss, mia_min_k, mia_min_k_pp, mia_ref, mia_zlib, roc_auc,
    )
    from forensic.mia_strata import _quantile

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ckpt = reg["checkpoint"]
    label = args.label or Path(args.registry).stem
    mem_texts, non_texts = mia_texts_from_registry(reg)
    docs = [(t, 1) for t in mem_texts] + [(t, 0) for t in non_texts]
    labels = [lab for _t, lab in docs]
    print(f"[i] target {ckpt}  (epochs={reg.get('hyperparams', {}).get('epochs')}, label={label})")
    print(f"[i] MIA docs: {len(mem_texts)} member / {len(non_texts)} non-member")

    print("[i] loading fine-tuned target…")
    target = TransformersBackend(model_id=ckpt, model_revision=None, torch_dtype=args.dtype).load()
    print(f"[i] loading reference base {args.ref_model}@{args.ref_revision}…")
    ref = TransformersBackend(model_id=args.ref_model, model_revision=args.ref_revision,
                              torch_dtype=args.dtype).load()

    # per-doc scores under target + reference (white-box: exact per-token loss)
    print("[i] scoring (target + reference)…")
    per_method: dict[str, list[float]] = {m: [] for m in (*BLACKBOX_METHODS, *WHITEBOX_ONLY)}
    for i, (text, _lab) in enumerate(docs, start=1):
        st = target.score_sequence(text)
        sr = ref.score_sequence(text)
        per_method["loss"].append(mia_loss(st))
        per_method["min_k"].append(mia_min_k(st))
        per_method["min_k_pp"].append(mia_min_k_pp(st))
        per_method["zlib"].append(mia_zlib(st))
        per_method["ref"].append(mia_ref(st, sr))
        if i % 100 == 0:
            print(f"      {i}/{len(docs)} docs…")

    def _metrics(scores: list[float]) -> dict:
        auc = roc_auc(scores, labels)
        nonm = [s for s, l in zip(scores, labels) if l == 0]
        memb = [s for s, l in zip(scores, labels) if l == 1]
        thr = _quantile(nonm, 1.0 - args.fpr) if nonm else float("inf")
        confirmed = sum(1 for s in memb if s > thr)
        return {"auc": round(auc, 4), "members_confirmed_at_fpr": confirmed,
                "members": len(memb), "non_members": len(nonm)}

    method_metrics = {m: _metrics(per_method[m]) for m in per_method}

    wb_best = max(method_metrics, key=lambda m: method_metrics[m]["auc"])
    bb_best = max(BLACKBOX_METHODS, key=lambda m: method_metrics[m]["auc"])

    summary = {
        "status": "ok", "label": label, "checkpoint": ckpt,
        "epochs": reg.get("hyperparams", {}).get("epochs"),
        "reference_model": f"{args.ref_model}@{args.ref_revision}",
        "fpr": args.fpr,
        "methods": method_metrics,
        "white_box": {
            "available_methods": list(per_method.keys()),
            "best_method": wb_best, "best_auc": method_metrics[wb_best]["auc"],
            "note": "attacker has the weights -> exact per-token loss; all methods incl. reference-calibrated",
        },
        "black_box": {
            "available_methods_E0": list(BLACKBOX_METHODS),
            "best_method_E0": bb_best, "best_auc_E0": method_metrics[bb_best]["auc"],
            "E1_feasible": False,
            "note": ("API-only: logprob-derived scores only (no reference model). Feasible only while "
                     "the score channel is open (E0); E1 suppresses logprobs (952->0 events) -> MIA "
                     "infeasible (AUC->chance). black-box best <= white-box best (no calibration)."),
        },
    }

    res_dir = Path(args.repo_root) / "results" / "mia_comparison"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / f"mia_comparison_{label}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n========== MIA — confronto metodi ({label}, epochs={summary['epochs']}) ==========")
    print(f"  {'metodo':10} {'AUC':>8} {'membri@FPR'+str(int(args.fpr*100))+'%':>14}   accesso")
    for m in (*BLACKBOX_METHODS, *WHITEBOX_ONLY):
        mm = method_metrics[m]
        access = "white+black(E0)" if m in BLACKBOX_METHODS else "white-box only"
        print(f"  {m:10} {mm['auc']:>8.4f} {str(mm['members_confirmed_at_fpr'])+'/'+str(mm['members']):>14}   {access}")
    print(f"\n  WHITE-BOX best: {wb_best} AUC={method_metrics[wb_best]['auc']:.4f}  (upper bound, model access)")
    print(f"  BLACK-BOX best (E0, no ref): {bb_best} AUC={method_metrics[bb_best]['auc']:.4f}")
    print(f"  BLACK-BOX under E1: INFEASIBLE (score channel closed: logprobs 952->0)")
    print(f"\n[✓] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
