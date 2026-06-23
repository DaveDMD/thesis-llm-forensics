#!/usr/bin/env python3
"""v5 re-analysis on the combined hard-negative RAG pool (NO GPU).

Reconstructs the combined-pool ground truth deterministically from the plan (no model call),
reads the COMMITTED forensic streams (``evidence/combined_rag/E{0,1}_*.jsonl``), and
evaluates the v5 adaptive detector in three honest views:

  * GATE — reproduce the committed numbers (frozen v1/v2/v3, v4-RAG, grouped-CV) to
    prove the ground-truth reconstruction is correct before trusting v5;
  * Motore 1 — behavioural-feature specialisation in-distribution (grouped CV):
    AUC + TPR@FPR + per-family, vs the length-only / all-feature ablation;
  * Motore 2 — the DEFENSE-ACTION online loop (v5, base = frozen v3): AUC + TPR@FPR
    + n_refits + pseudo-label precision, for the naive and behaviourally-gated
    anchors, E0 vs E1 (the defense fires only in E1);
  * LOFO — leave-one-family-out: does the behavioural base catch a family it never
    saw (closed-world limit), and does the defense loop rescue it?

This reads only committed artefacts and writes a NEW file
``results/b7_metrics/v5_rag.json`` — no existing campaign data is touched.

Reproduce (no GPU):
    python3 tools/forensic/runners/run_b7_v5_rag.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent.parent.parent
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

SALT = b"combined-rag-salt-0123456789abc!"
BIG_FAMILIES = ("promptinject_goal_hijack", "promptinject_prompt_leak", "rag_kb_extraction")


def _read(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _latest(glob_pat: str, directory: Path) -> Path:
    cands = sorted(directory.glob(glob_pat))
    if not cands:
        raise FileNotFoundError(f"no stream matching {glob_pat} in {directory}")
    return cands[-1]


def _rank1_by_session(records: list[dict]) -> dict[str, list[str]]:
    out: dict[str, set] = defaultdict(set)
    for r in records:
        if r.get("event_type") != "rag_retrieval":
            continue
        for hit in r["payload"].get("retriever_hits", []):
            if int(hit.get("rank", 0)) == 1 and hit.get("chunk_hash"):
                out[r["session_id"]].add(hit["chunk_hash"])
    return {k: sorted(v) for k, v in out.items()}


def _defense_rate_by_session(records: list[dict]) -> dict[str, float]:
    """Observable defense action per session = share of completions redacted."""
    tot: dict[str, int] = defaultdict(int)
    red: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("event_type") != "completion":
            continue
        sid = r["session_id"]
        tot[sid] += 1
        if r["payload"].get("response_redactions"):
            red[sid] += 1
    return {sid: (red[sid] / tot[sid] if tot[sid] else 0.0) for sid in tot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(_ROOT))
    ap.add_argument("--fpr", type=float, default=0.1)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    from forensic.b7_operating_point import ablation_grouped_cv, auc_and_op, subset_dataset
    from forensic.detector_adaptive_rag import AdaptiveDetectorRAG
    from forensic.detector_ml import build_xy, cross_validate_grouped
    from forensic.detector_store import load_scorer
    from forensic.detector_v5 import (
        AdaptiveDetectorV5, behavioral_feature_names, fit_behavioral_base, pseudo_label_precision,
    )
    from forensic.features import build_features
    from forensic.hashing import pseudonymize
    from forensic.pile_detector import aggregate_sessions
    from forensic.pipeline import _structural_anti_leak
    from forensic.traffic_combined_rag import build_combined_rag_plan

    repo = Path(args.repo_root)
    ev_dir = repo / "evidence" / "combined_rag"
    res_dir = repo / "results" / "combined_rag"
    out_dir = repo / "results" / "b7_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    committed = json.loads((res_dir / "combined_detection_summary.json").read_text(encoding="utf-8"))

    # ---- ground truth from the plan (deterministic, no GPU) ----
    plan = build_combined_rag_plan(n_attack_covers=3)
    gt_records: list[dict] = []
    fam_by_session: dict[str, str] = {}
    for case in plan:
        g = case.groundtruth_json()
        g["session_id"] = pseudonymize(g["session_id"], SALT)
        gt_records.append(g)
        sid = g["session_id"]
        if not g.get("is_attack"):
            fam_by_session.setdefault(sid, "benign")
        elif g.get("objective") != "reconnaissance":
            lab = g.get("attack_family", "?")
            if g.get("variant") == "stealth":
                lab += "/stealth"
            fam_by_session[sid] = lab

    frozen = {n: load_scorer(repo / "results" / "detectors" / f"{n}.joblib")[0]
              for n in ("v1", "v2", "v3")}

    def build_srows(records: list[dict]) -> list[dict]:
        _structural_anti_leak(records)
        completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                     for r in records if r["event_type"] == "completion"}
        feat_records = [r for r in records if not (
            r["event_type"] == "prompt"
            and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
        srows = aggregate_sessions(build_features(feat_records, gt_records))
        rank1 = _rank1_by_session(records)
        defense = _defense_rate_by_session(records)
        for row in srows:
            sid = str(row.get("session_id"))
            row["session_rank1_chunk_hashes"] = rank1.get(sid, [])
            row["session_defense_action_rate"] = float(defense.get(sid, 0.0))
            row["_family"] = fam_by_session.get(sid, "benign")
        return srows

    def gate(env: str, srows: list[dict]) -> dict:
        """Reproduce committed frozen / v4rag / grouped_cv AUCs as a correctness proof."""
        c = committed[env]["detectors"]
        y = [int(r["label_is_attack"]) for r in srows]
        checks, ok = {}, True
        for n in ("v1", "v2", "v3"):
            auc = auc_and_op([frozen[n](r) for r in srows], y, fpr_target=args.fpr)["auc"]
            exp = c[f"frozen_{n}"]["auc"]
            match = abs(auc - exp) <= args.tol
            ok &= match
            checks[f"frozen_{n}"] = {"auc": auc, "committed": exp, "match": match}
        v4 = AdaptiveDetectorRAG(frozen["v3"])
        v4_auc = auc_and_op([v4.observe_and_score(r)["score"] for r in srows], y, fpr_target=args.fpr)["auc"]
        match = abs(v4_auc - c["v4rag"]["auc"]) <= args.tol and v4.n_refits == c["v4rag"]["refits"]
        ok &= match
        checks["v4rag"] = {"auc": v4_auc, "committed": c["v4rag"]["auc"],
                           "n_refits": v4.n_refits, "committed_refits": c["v4rag"]["refits"], "match": match}
        cv = cross_validate_grouped(build_xy(srows), n_splits=args.n_splits)
        cv_auc = auc_and_op(list(cv.oof_y_score), list(cv.oof_y_true), fpr_target=args.fpr)["auc"]
        match = abs(cv_auc - c["grouped_cv"]["auc"]) <= max(args.tol, 0.02)
        ok &= match
        checks["grouped_cv"] = {"auc": cv_auc, "committed": c["grouped_cv"]["auc"], "match": match}
        return {"gate_ok": ok, "checks": checks}

    def motore1(srows: list[dict]) -> dict:
        """Behavioural specialisation in-distribution + length/all ablation + per-family."""
        abl = ablation_grouped_cv(srows, fpr_target=args.fpr, n_splits=args.n_splits)
        beh = behavioral_feature_names(srows)
        cv = cross_validate_grouped(subset_dataset(build_xy(srows), beh), n_splits=args.n_splits)
        fams = [srows[i]["_family"] for i in cv.oof_index]
        per = auc_and_op(list(cv.oof_y_score), list(cv.oof_y_true), fams, fpr_target=args.fpr)
        return {"ablation": abl, "behavioral_with_per_family": per}

    def motore2(srows: list[dict]) -> dict:
        beh = behavioral_feature_names(srows)
        y = [int(r["label_is_attack"]) for r in srows]
        fams = [r["_family"] for r in srows]
        out = {}
        for anchor in ("defense", "defense_gated"):
            det = AdaptiveDetectorV5(frozen["v3"], anchor=anchor, restrict_features=beh)
            res = [det.observe_and_score(r) for r in srows]
            op = auc_and_op([x["score"] for x in res], y, fams, fpr_target=args.fpr)
            prec = pseudo_label_precision([x["pseudo_label"] for x in res], y)
            out[anchor] = {**op, "n_refits": det.n_refits, "pseudo_label": prec}
        return out

    def lofo(srows: list[dict]) -> dict:
        beh = behavioral_feature_names(srows)
        out = {}
        for fam in BIG_FAMILIES:
            train = [r for r in srows if r["_family"] != fam]
            testF = [r for r in srows if r["_family"] == fam]
            if not testF or len({r["label_is_attack"] for r in train}) < 2:
                out[fam] = {"note": "insufficient data"}
                continue
            base = fit_behavioral_base(train)
            base_tpr = sum(1 for r in testF if base(r) >= 0.5) / len(testF)
            det = AdaptiveDetectorV5(base, anchor="defense", restrict_features=beh)
            online_F = {}
            for r in srows:
                o = det.observe_and_score(r)
                if r["_family"] == fam:
                    online_F[str(r["session_id"])] = o["score"]
            online_tpr = sum(1 for s in online_F.values() if s >= 0.5) / len(online_F) if online_F else None
            out[fam] = {
                "n_family_sessions": len(testF),
                "base_tpr_at_0.5": round(base_tpr, 3),
                "online_tpr_at_0.5": round(online_tpr, 3) if online_tpr is not None else None,
                "n_refits": det.n_refits,
            }
        return out

    results = {}
    for env in ("E0", "E1"):
        stream = _latest(f"{env}_*.jsonl", ev_dir)
        srows = build_srows(_read(stream))
        n_def = sum(1 for r in srows if r["session_defense_action_rate"] > 0.0)
        results[env] = {
            "stream": stream.name,
            "n_sessions": len(srows),
            "n_attack": sum(int(r["label_is_attack"]) for r in srows),
            "n_sessions_with_defense_action": n_def,
            "gate": gate(env, srows),
            "motore1_specialised_base": motore1(srows),
            "motore2_defense_loop": motore2(srows),
            "lofo_novelty": lofo(srows),
        }

    summary = {
        "status": "ok",
        "analysis": "b7_v5_rag",
        "world": committed.get("world"),
        "fpr_target": args.fpr,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "salt_id": "combined-rag-salt",
        "note": ("v5 = behavioural-specialised base (Motore 1) + defense-action online "
                 "anchor (Motore 2); LOFO probes the closed-world limit. Reads committed "
                 "streams only; gate reproduces the committed frozen/v4rag/grouped_cv AUCs."),
        "E0": results["E0"],
        "E1": results["E1"],
    }
    out_path = out_dir / "v5_rag.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- console report ----
    print("\n========== v5 RAG re-analysis (no GPU) ==========")
    for env in ("E0", "E1"):
        r = results[env]
        g = r["gate"]
        print(f"\n[{env}] sessions={r['n_sessions']} attack={r['n_attack']} "
              f"defense-action sessions={r['n_sessions_with_defense_action']}  GATE_OK={g['gate_ok']}")
        for k, v in g["checks"].items():
            print(f"    gate {k}: auc={v['auc']} (committed {v['committed']}) match={v['match']}")
        m1 = r["motore1_specialised_base"]["ablation"]
        print(f"    Motore1 ablation: all={m1['all'].get('auc')} "
              f"behavioral_only={m1['behavioral_only'].get('auc')} "
              f"length_only={m1['length_only'].get('auc')}")
        beh = r["motore1_specialised_base"]["behavioral_with_per_family"]
        print(f"    Motore1 behavioural: AUC={beh['auc']} TPR@FPR={beh['tpr']} per_family={beh['per_family']}")
        for anchor, d in r["motore2_defense_loop"].items():
            print(f"    Motore2[{anchor}]: AUC={d['auc']} TPR@FPR={d['tpr']} refits={d['n_refits']} "
                  f"pseudo+={d['pseudo_label'].get('n_pos_anchors')} "
                  f"prec={d['pseudo_label'].get('pos_anchor_precision')}")
        for fam, d in r["lofo_novelty"].items():
            print(f"    LOFO[{fam}]: base_TPR={d.get('base_tpr_at_0.5')} "
                  f"online_TPR={d.get('online_tpr_at_0.5')} refits={d.get('n_refits')}")
    print(f"\n[✓] summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
