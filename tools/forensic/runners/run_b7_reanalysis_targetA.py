#!/usr/bin/env python3
"""Operating-point RE-ANALYSIS of the COMMITTED-but-UNTRACKED TARGET A run (no GPU).

Twin of ``run_b7_reanalysis_target{B,C}.py`` for TARGET A (gpt2-medium fine-tuned on
WikiText, 12 epochs, heavy OVERFIT — registry ``c630d00e``). TARGET A's runtime
streams SURVIVED on disk under ``evidence/runtime_gpt2/`` with unique UUID names; only
its fixed-path summary ``runtime_battery.json`` was overwritten by the later TARGET B
run (shared fixed path), so A was mistakenly thought lost. This script LOCATES A's
streams by matching the per-prompt ``model_id`` to A's registry id, reads them
READ-ONLY, and recomputes ROC-AUC + TPR@FPR + per-family at a realistic operating
point plus the length-vs-behavioural ablation, for the two frozen detectors this
battery carried (v1, v2). No model is loaded; the target is never re-run.

The forensic stream carries no labels (two-tier), so the session-level ground truth
is RECONSTRUCTED from A's registry with the same plan builders and salt as
``run_runtime_detection_gpt2_real.py`` (identical to the B/C re-analysis, which were
self-validated against their committed summaries). A's own committed summary was
overwritten, so A cannot be cross-checked against it; instead the reconstruction is
validated by (a) MODEL BINDING — every recovered stream's ``model_id`` resolves to
registry ``c630d00e``; (b) full label COVERAGE (no stream session left unlabelled);
(c) observed family counts vs the planned counts. Output goes to a NEW file
``results/b7_metrics/targetA.json``; nothing existing is modified.

NOTE (provenance): A's streams are UNTRACKED — force-add them at commit time to stop
them from being lost for real.

Reproduce (CPU only; needs the recovered A streams + the A registry):
    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
      thesis python3 tools/forensic/runners/run_b7_reanalysis_targetA.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Must match run_runtime_detection_gpt2_real.py exactly.
SALT = b"runtime-gpt2-detect-salt-32bytes!"
REGISTRY = "results/canary/c630d00e-9dda-4265-ad79-e5ae39189b40_registry.json"
REGISTRY_ID = "c630d00e-9dda-4265-ad79-e5ae39189b40"
BENIGN_MIX = dict(n_short=18, n_multi=8, multi_size=5, n_codecomplete=16,
                  n_check=14, n_coverage=6, coverage_size=20)


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _stream_model_id(path: Path) -> str | None:
    """Read the model_id from the stream's first prompt event (the model binding)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("event_type") == "prompt":
                return r["payload"].get("model_id")
    return None


def _pick_a_logs(repo: str, subdir: str, registry_id: str) -> dict[str, dict]:
    """Resolve A's E0/E1 streams by model binding: among the runtime_gpt2 streams,
    keep those whose model_id resolves to A's registry, then take the largest per env
    (the full battery, not a smoke run)."""
    base = Path(repo) / subdir
    out: dict[str, dict] = {}
    for env in ("E0", "E1"):
        cands = []
        for p in sorted(base.glob(f"{env}_*.jsonl")):
            mid = _stream_model_id(p)
            if mid and registry_id in str(mid):
                cands.append((p, p.stat().st_size, str(mid)))
        if cands:
            p, _sz, mid = max(cands, key=lambda t: t[1])
            out[env] = {"path": str(p), "model_id": mid}
    return out


def _reconstruct_gt(repo: str, chunk_size: int) -> dict[str, dict]:
    """Rebuild the runtime-detection plan from A's registry -> map pseudo session_id
    -> {is_attack, family} (same builders/prefixes/salt as the runner)."""
    from forensic.canary_dataset import mia_texts_from_registry
    from forensic.hashing import pseudonymize
    from forensic.mia_pile import MiaTarget, build_mia_pile_plan
    from forensic.mia_score import build_mia_score_plan
    from forensic.pile_detector import build_benign_sessions

    reg = json.loads((Path(repo) / REGISTRY).read_text(encoding="utf-8"))
    canaries = reg["canaries"]
    mem_texts, non_texts = mia_texts_from_registry(reg)

    raw: dict[str, dict] = {}

    def add(sid: str, is_attack: bool, family: str) -> None:
        e = raw.setdefault(sid, {"is_attack": False, "family": None})
        if is_attack:
            e["is_attack"] = True
            if e["family"] in (None, "benign"):
                e["family"] = family
        elif e["family"] is None:
            e["family"] = family

    canary_targets = [
        MiaTarget(target_id=c["canary_id"], domain="canary", full_text=c["prefix"] + c["value"],
                  prefix=c["prefix"], suffix=c["value"], is_member=True,
                  is_secret_bearing=True, secret_kind=c["kind"])
        for c in canaries
    ]
    for i in range(0, len(canary_targets), chunk_size):
        chunk = canary_targets[i:i + chunk_size]
        for c in build_mia_pile_plan(chunk, session_prefix=f"rtext-{i // chunk_size:02d}", max_tokens=32):
            if c.groundtruth.get("is_attack"):
                add(c.body["session_id"], True,
                    c.groundtruth.get("attack_family") or "pretraining_membership_inference")
    doc_targets = (
        [MiaTarget(f"mdoc-{i:04d}", "canary", t, t[:60], t[60:], True, False, None)
         for i, t in enumerate(mem_texts)]
        + [MiaTarget(f"ndoc-{i:04d}", "canary", t, t[:60], t[60:], False, False, None)
           for i, t in enumerate(non_texts)]
    )
    for i in range(0, len(doc_targets), 25):
        chunk = doc_targets[i:i + 25]
        for c in build_mia_score_plan(chunk, session_prefix=f"rtmia-{i // 25:02d}"):
            if c.groundtruth.get("is_attack"):
                add(c.body["session_id"], True,
                    c.groundtruth.get("attack_family") or "pretraining_membership_inference_scorebased")
    for c in build_benign_sessions(corpus_texts=non_texts, session_prefix="rtben", **BENIGN_MIX):
        add(c.body["session_id"], False, "benign")
    for c in canaries:
        add(f"samp-{c['canary_id']}", True, "sampling")
        add(f"adap-{c['canary_id']}", True, "adaptive")

    return {pseudonymize(sid, SALT): v for sid, v in raw.items()}


def _feature_rows(log_path: str, gt_map: dict[str, dict]):
    """Rebuild v1 and v2 feature views from the stream (v2 recomputes the secret-like
    flag with the extended recogniser, exactly as the committed runner)."""
    from forensic import detector_v2 as v2
    from forensic.features import build_features
    from forensic.pipeline import _structural_anti_leak

    records = _read(log_path)
    _structural_anti_leak(records)
    completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                 for r in records if r["event_type"] == "completion"}
    feat_records = [r for r in records if not (
        r["event_type"] == "prompt"
        and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
    gt_records = []
    for r in feat_records:
        if r["event_type"] != "prompt":
            continue
        p = r["payload"]
        sid = r["session_id"]
        info = gt_map.get(str(sid))
        is_atk = bool(info["is_attack"]) if info else False
        gt_records.append({
            "session_id": sid, "sequence_number": int(p["sequence_number"]), "endpoint": p["endpoint"],
            "case_id": f"{sid}-{p['sequence_number']}-{p['endpoint']}", "scenario": None,
            "is_attack": is_atk, "attack_family": (info["family"] if (info and is_atk) else None),
            "objective": None,
        })
    feat_v1 = build_features(feat_records, gt_records)
    resp_by_key = {}
    for rec in records:
        if rec.get("event_type") == "completion":
            p = rec["payload"]
            resp_by_key[(str(rec["session_id"]), int(p.get("sequence_number", 0)))] = p.get("response_raw") or ""
    feat_v2 = []
    for r in feat_v1:
        raw = resp_by_key.get((str(r.get("session_id")), int(r.get("sequence_number", 0))))
        feat_v2.append({**r, "feature_response_contains_secret_like_pattern": v2.contains_secret_like_v2(raw)}
                       if raw is not None else r)
    return feat_v1, feat_v2


def _sessions(feat, gt_map):
    from forensic.pile_detector import aggregate_sessions
    sess = aggregate_sessions(feat)
    matched = 0
    for s in sess:
        info = gt_map.get(str(s.get("session_id")))
        if info is None:
            s["label_is_attack"] = 0
            s["_family"] = "unknown"
        else:
            s["label_is_attack"] = 1 if info["is_attack"] else 0
            s["_family"] = info["family"]
            matched += 1
    return sess, len(sess) - matched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default="/workspace")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--detectors-dir", default="/workspace/results/detectors")
    ap.add_argument("--fpr", type=float, default=0.1)
    args = ap.parse_args()

    from forensic.b7_operating_point import ablation_grouped_cv, auc_and_op
    from forensic.detector_store import load_scorer
    from forensic.online_detector import detection_metrics, posthoc_detect

    det_dir = Path(args.detectors_dir)
    scorers = {}
    for n in ("v1", "v2"):
        p = det_dir / f"{n}.joblib"
        if not p.exists():
            print(f"[!] missing frozen detector {n} in {det_dir}")
            return 2
        scorers[n], _names, _prov = load_scorer(p)

    logs = _pick_a_logs(args.repo_root, "evidence/runtime_gpt2", REGISTRY_ID)
    if "E0" not in logs or "E1" not in logs:
        print(f"[!] could not resolve A's E0/E1 streams (model_id binding to {REGISTRY_ID}) "
              f"under evidence/runtime_gpt2 (found {list(logs)})")
        return 2

    print("[i] reconstructing TARGET A ground truth from the registry plan (CPU)…")
    gt_map = _reconstruct_gt(args.repo_root, args.chunk_size)
    plan_fam = dict(Counter(v["family"] for v in gt_map.values()))
    print(f"[i] reconstructed {len(gt_map)} session labels; planned families={plan_fam}")

    by_env: dict[str, dict] = {}
    all_valid = True
    for env in ("E0", "E1"):
        log = logs[env]["path"]
        feat_v1, feat_v2 = _feature_rows(log, gt_map)
        sess_v1, n_unmatched = _sessions(feat_v1, gt_map)
        sess_v2, _ = _sessions(feat_v2, gt_map)
        sess_by_det = {"v1": sess_v1, "v2": sess_v2}

        # ── validation (no committed summary for A — overwritten by B) ──────────────
        #  (a) model binding: stream model_id resolves to A's registry;
        #  (b) coverage: every stream session got a label;
        #  (c) observed family counts vs planned.
        observed_fam = dict(Counter(s["_family"] for s in sess_v1))
        model_match = REGISTRY_ID in str(logs[env]["model_id"])
        coverage_ok = (n_unmatched == 0)
        # E0 should observe the full plan; E1 may drop rate-limited sessions (flagged).
        fam_match = (observed_fam == plan_fam) if env == "E0" else True
        valid = bool(model_match and coverage_ok and fam_match)
        all_valid = all_valid and valid

        # ── operating-point metrics ────────────────────────────────────────────────
        frozen = {}
        for det in ("v1", "v2"):
            sd = sess_by_det[det]
            y = [int(s["label_is_attack"]) for s in sd]
            fam = [s["_family"] for s in sd]
            op = auc_and_op([scorers[det](s) for s in sd], y, fam, fpr_target=args.fpr)
            m = detection_metrics(posthoc_detect(sd, scorers[det], threshold=0.5))
            op["posthoc_det@0.5"] = round(m["detection_rate"], 3)
            op["posthoc_fa@0.5"] = round(m["false_alarm_rate"], 3)
            frozen[det] = op
        ablation = ablation_grouped_cv(sess_v1, fpr_target=args.fpr, n_splits=5)

        by_env[env] = {
            "log": str(Path(log).relative_to(args.repo_root)),
            "model_id": logs[env]["model_id"], "tracked_in_git": False,
            "n_sessions": len(sess_v1), "n_unmatched": n_unmatched,
            "planned_family_counts": plan_fam, "observed_family_counts": observed_fam,
            "validation": {"ok": valid, "model_match": model_match, "coverage_ok": coverage_ok,
                           "fam_match_E0": fam_match,
                           "note": "A's committed summary was overwritten by the TARGET B run "
                                   "(shared fixed path); validated via model binding + coverage + plan."},
            "frozen": frozen, "ablation": ablation,
        }

    res_dir = Path(args.repo_root) / "results" / "b7_metrics"
    res_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "ok" if all_valid else "validation_failed",
               "target": "gpt2-medium@c630d00e", "regime": "finetune_overfit_wikitext_12ep",
               "fpr_target": args.fpr, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Re-analysis of RECOVERED untracked A streams; no model run, no existing artefact "
                       "modified. A's fixed-path summary was overwritten by TARGET B; here A is rebuilt "
                       "from its surviving streams (model_id binding) + registry. v3/v4 are not in this battery.",
               "by_environment": by_env}
    out = res_dir / "targetA.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== Operating-point re-analysis — TARGET A (recovered streams) ==========")
    for env in ("E0", "E1"):
        a = by_env[env]
        v = a["validation"]
        print(f"\n[{env}] {a['log']}  (model={a['model_id']}, untracked)")
        print(f"  sessions={a['n_sessions']} unmatched={a['n_unmatched']} observed_families={a['observed_family_counts']}")
        print(f"  validation: {'OK' if v['ok'] else 'FAIL'} (model_match={v['model_match']} "
              f"coverage_ok={v['coverage_ok']} fam_match_E0={v['fam_match_E0']})")
        if not v["ok"]:
            print("  [!] reconstruction did NOT validate — new metrics NOT trustworthy.")
        for det in ("v1", "v2"):
            f = a["frozen"][det]
            print(f"    {det}: AUC={f['auc']} TPR@FPR{args.fpr}={f['tpr']} (fpr={f['fpr']}) "
                  f"[posthoc det/FA@0.5={f['posthoc_det@0.5']}/{f['posthoc_fa@0.5']}] per_family={f['per_family']}")
        abl = a["ablation"]
        for k in ("all", "length_only", "behavioral_only"):
            d = abl[k]
            print(f"      {k:16} AUC={d.get('auc')} TPR@FPR{args.fpr}={d.get('tpr')} (n_features={d.get('n_features')})")
    print(f"\n[{'✓' if all_valid else '✗'}] summary: {out}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
