#!/usr/bin/env python3
"""Detector generalization test on the accumulated residue pool.

Realistic premise: in production the detector runs alongside the LLM, trained on
GENERIC residues, and does NOT know which attacks arrive. We test how well a
detector trained on some campaigns generalizes to a held-out, OUT-OF-DISTRIBUTION
scenario it never saw (a different domain or model size).

This is a pure analysis over the persisted residue pool
(``results/residue_pool/sessions.jsonl``, written by run_attack_harvest_real.py):
no GPU / no model load.

  * IN-DISTRIBUTION = pool sessions NOT in the held-out cell;
  * OOD            = pool sessions in the held-out cell (e.g. domain=arxiv, or
                     model=...-1.4b).

A frozen detector is fit on a train split of the in-distribution sessions and
applied (post-hoc, complete-session) to (a) a held-out in-distribution test split
and (b) the OOD sessions. We report detection rate / false-alarm rate for both,
so the generalization gap (in-dist vs OOD) is explicit.

Example::

    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
        thesis python3 tools/forensic/runners/run_detector_generalization_real.py \\
        --holdout-key domain --holdout-value arxiv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="/workspace/results/residue_pool/sessions.jsonl")
    ap.add_argument("--holdout-key", default="domain", choices=["domain", "model"],
                    help="provenance axis held out as OOD (prov_domain / prov_model)")
    ap.add_argument("--holdout-value", required=True, help="value of that axis to hold out as OOD")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from forensic.harvest import load_jsonl
    from forensic.online_detector import (
        detection_metrics,
        fit_session_scorer,
        posthoc_detect,
        split_sessions,
    )

    rows = load_jsonl(args.pool)
    if not rows:
        print(f"[!] empty/missing residue pool: {args.pool}")
        return 2

    prov = f"prov_{args.holdout_key}"
    if prov not in rows[0]:
        print(f"[!] pool rows have no '{prov}' field (provenance missing)")
        return 2

    ood = [r for r in rows if str(r.get(prov)) == args.holdout_value]
    indist = [r for r in rows if str(r.get(prov)) != args.holdout_value]
    if not ood:
        cells = sorted({str(r.get(prov)) for r in rows})
        print(f"[!] no OOD sessions for {prov}={args.holdout_value!r}; available: {cells}")
        return 2
    if not indist:
        print("[!] no in-distribution sessions left after holdout")
        return 2

    # train / in-dist-test split (stratified by label, deterministic)
    train_ids, indist_test_ids = split_sessions(indist, train_frac=args.train_frac)
    train_rows = [r for r in indist if str(r["session_id"]) in train_ids]
    indist_test = [r for r in indist if str(r["session_id"]) in indist_test_ids]

    score_fn, _names = fit_session_scorer(train_rows)

    indist_res = posthoc_detect(indist_test, score_fn, threshold=args.threshold)
    ood_res = posthoc_detect(ood, score_fn, threshold=args.threshold)
    m_indist = detection_metrics(indist_res)
    m_ood = detection_metrics(ood_res)

    def _cells(rs):
        return sorted({(str(r.get("prov_domain")), str(r.get("prov_model"))) for r in rs})

    summary = {
        "status": "ok",
        "pool": args.pool,
        "holdout": {"key": args.holdout_key, "value": args.holdout_value},
        "n_pool_sessions": len(rows),
        "train": {"n_sessions": len(train_rows), "cells": _cells(train_rows)},
        "in_distribution_test": {"n_sessions": len(indist_test), "metrics": m_indist},
        "ood_test": {"n_sessions": len(ood), "cells": _cells(ood), "metrics": m_ood},
        "generalization_gap": {
            "detection_rate_drop": round(m_indist["detection_rate"] - m_ood["detection_rate"], 3),
            "false_alarm_delta": round(m_ood["false_alarm_rate"] - m_indist["false_alarm_rate"], 3),
        },
    }

    res_dir = Path(args.repo_root) / "results" / "generalization"
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / f"holdout_{args.holdout_key}_{args.holdout_value}.json".replace("/", "-")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[✓] detector trained on {len(train_rows)} in-distribution sessions "
          f"(cells={_cells(train_rows)})")
    print(f"    held out OOD: {args.holdout_key}={args.holdout_value} "
          f"({len(ood)} sessions, cells={_cells(ood)})\n")
    print(f"    {'split':18} {'sessions':>8} {'detect':>8} {'false-alarm':>12}")
    print(f"    {'in-distribution':18} {m_indist['n_attack'] + m_indist['n_benign']:>8} "
          f"{m_indist['detection_rate']:>8.3f} {m_indist['false_alarm_rate']:>12.3f}")
    print(f"    {'OUT-of-distribution':18} {m_ood['n_attack'] + m_ood['n_benign']:>8} "
          f"{m_ood['detection_rate']:>8.3f} {m_ood['false_alarm_rate']:>12.3f}")
    print(f"\n    generalization gap: detection drop = "
          f"{summary['generalization_gap']['detection_rate_drop']:+.3f}, "
          f"false-alarm delta = {summary['generalization_gap']['false_alarm_delta']:+.3f}")
    print(f"\n[✓] report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
