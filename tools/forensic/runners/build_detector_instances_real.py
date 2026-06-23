#!/usr/bin/env python3
"""Build the two FROZEN detector instances ONCE — v1 and v2 — and save them.

Train-once: after this, every experiment LOADS these instances (never re-fits).

  * v1 — Passata-1 frozen detector: fit on the residue pool EXCLUDING the gpt2
          cell (knows nothing of the new environment). This is the same detector
          that produced the 0.667 OOD generalisation.
  * v2 — specialised detector: fit on the pool INCLUDING the gpt2 residues (the
          defender that has collected this environment). v2 additionally uses the
          extended secret recogniser (forensic.detector_v2) AT INFERENCE time on
          new traffic; the training-side effect of the extended recogniser on the
          pool is negligible (the PAN/CONF formats appear in ~5 gpt2 sessions and
          not at all in the Pythia cells), so v2 trains on the committed pool.

Pure analysis over the committed residue pool — no GPU, no model load.

    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
        thesis python3 tools/forensic/runners/build_detector_instances_real.py
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

GPT2_CELL = "gpt2-medium-finetuned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="/workspace/results/residue_pool/sessions.jsonl")
    ap.add_argument("--out-dir", default="/workspace/results/detectors")
    args = ap.parse_args()

    from forensic.detector_store import fit_estimator, save_scorer
    from forensic.harvest import load_jsonl

    pool = load_jsonl(args.pool)
    if not pool:
        print(f"[!] empty/missing residue pool: {args.pool}")
        return 2

    def _cells(rows):
        return dict(Counter(f"{r.get('prov_domain')}/{r.get('prov_model')}" for r in rows))

    v1_train = [r for r in pool if str(r.get("prov_model")) != GPT2_CELL]
    v2_train = list(pool)

    specs = [
        ("v1", v1_train, "frozen Passata-1 detector (pool excluding the gpt2 cell)", "v1_regex"),
        ("v2", v2_train, "specialised detector (pool including gpt2 residues)", "v2_extended_PAN_CONF"),
    ]
    out_dir = Path(args.out_dir)
    for name, train, desc, recogniser in specs:
        est, names = fit_estimator(train)
        prov = {
            "instance": name,
            "description": desc,
            "secret_recogniser": recogniser,
            "n_train_sessions": len(train),
            "n_attack": sum(1 for r in train if r.get("label_is_attack")),
            "train_cells": _cells(train),
            "includes_gpt2_cell": name == "v2",
        }
        path = save_scorer(out_dir / f"{name}.joblib", est, names, provenance=prov)
        print(f"[✓] {name}: {len(train)} sessions ({prov['n_attack']} attack), "
              f"{len(names)} features → {path}")
        print(f"       cells: {prov['train_cells']}")
    print("\n[i] v1 and v2 are now frozen instances; experiments LOAD them (never re-fit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
