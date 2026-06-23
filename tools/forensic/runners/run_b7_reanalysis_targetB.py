#!/usr/bin/env python3
"""Operating-point RE-ANALYSIS of the COMMITTED TARGET B campaign (no GPU).

Twin of ``run_b7_reanalysis_targetC.py`` for TARGET B (gpt2-medium fine-tuned on
OpenWebText, 1 epoch, realistic — registry ``607022bd``). Reads the committed
runtime-detection streams (``evidence/runtime_gpt2/E0_*.jsonl`` / ``E1_*.jsonl``)
READ-ONLY and recomputes ROC-AUC + TPR@FPR + per-family at a realistic operating
point, plus the length-vs-behavioural ablation, for the two frozen detectors this
battery carried (v1, v2). No model is loaded; the target is never re-run.

The forensic stream carries no labels (two-tier), so the session-level ground
truth is RECONSTRUCTED from the committed registry with the same plan builders and
salt as ``run_runtime_detection_gpt2_real.py``. v1 reads the default secret-like
flag; v2 recomputes it with the extended recogniser (exactly as the committed
runner), so v2's numbers are faithful. The reconstruction is SELF-VALIDATED: it
reproduces the committed per-family posthoc-detected counts for BOTH detectors
before trusting the new metrics. Nothing existing is modified — output goes to
``results/b7_metrics/targetB.json``.

Reproduce (CPU only; needs the committed evidence + the B registry):
    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
      thesis python3 tools/forensic/runners/run_b7_reanalysis_targetB.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Must match run_runtime_detection_gpt2_real.py exactly.
SALT = b"runtime-gpt2-detect-salt-32bytes!"
REGISTRY = "results/canary/607022bd-5f26-4b33-bbb4-e17215a883be_registry.json"
CKPT_SUBSTR = Path(REGISTRY).name[:8]  # "607022bd" — disambiguate B's streams from A's in shared dirs
BENIGN_MIX = dict(n_short=18, n_multi=8, multi_size=5, n_codecomplete=16,
                  n_check=14, n_coverage=6, coverage_size=20)
_FAMILY_ORDER = ["pretraining_membership_inference_scorebased", "pretraining_membership_inference",
                 "sampling", "adaptive", "benign"]


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _git_tracked(repo: str, subdir: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", repo, "ls-files", subdir],
                             capture_output=True, text=True, check=True)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def _owns(path: Path, substr: str) -> bool:
    """True if the stream belongs to this campaign (checkpoint id appears in its records)."""
    try:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if substr in line:
                    return True
                if i >= 200:
                    break
    except OSError:
        pass
    return False


def _pick_logs(repo: str, subdir: str) -> dict[str, str]:
    base = Path(repo) / subdir
    tracked = _git_tracked(repo, subdir)
    out: dict[str, str] = {}
    for env in ("E0", "E1"):
        cand = [t for t in tracked if Path(t).name.startswith(f"{env}_") and t.endswith(".jsonl")]
        # Disambiguate THIS campaign's streams from other campaigns' streams now committed
        # in the same shared dir (e.g. TARGET A streams under evidence/runtime_gpt2).
        owned = [t for t in cand if _owns(Path(repo) / t, CKPT_SUBSTR)]
        cand = owned or cand
        if cand:
            out[env] = str(Path(repo) / cand[0])
        else:
            disk = sorted(base.glob(f"{env}_*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
            disk = [p for p in disk if _owns(p, CKPT_SUBSTR)] or disk
            if disk:
                out[env] = str(disk[0])
    return out


def _reconstruct_gt(repo: str, chunk_size: int) -> dict[str, dict]:
    """Rebuild the runtime-detection plan from the registry -> map pseudo
    session_id -> {is_attack, family} (same builders/prefixes/salt as the runner)."""
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


def _committed(repo: str, env: str) -> dict:
    d = json.loads((Path(repo) / "results" / "runtime_detection" / "runtime_battery.json").read_text("utf-8"))
    r = d["by_environment"][env]
    pf = {det: {x["family"]: x["posthoc_detected"] for x in r[det]["per_family"]} for det in ("v1", "v2")}
    fam = {x["family"]: x["n_sessions"] for x in r["v1"]["per_family"]}
    return {"per_family_posthoc": pf, "family_counts": fam, "n_sessions": r["n_sessions"]}


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

    logs = _pick_logs(args.repo_root, "evidence/runtime_gpt2")
    if "E0" not in logs or "E1" not in logs:
        print(f"[!] could not resolve committed E0/E1 streams under evidence/runtime_gpt2 (found {list(logs)})")
        return 2

    print("[i] reconstructing TARGET B ground truth from the registry plan (CPU)…")
    gt_map = _reconstruct_gt(args.repo_root, args.chunk_size)
    plan_fam = dict(Counter(v["family"] for v in gt_map.values()))
    print(f"[i] reconstructed {len(gt_map)} session labels; planned families={plan_fam}")

    by_env: dict[str, dict] = {}
    all_valid = True
    for env in ("E0", "E1"):
        feat_v1, feat_v2 = _feature_rows(logs[env], gt_map)
        sess_v1, n_unmatched = _sessions(feat_v1, gt_map)
        sess_v2, _ = _sessions(feat_v2, gt_map)
        committed = _committed(args.repo_root, env)
        sess_by_det = {"v1": sess_v1, "v2": sess_v2}

        # ── self-validation: reproduce committed per-family posthoc-detected counts ─
        recomputed_pf = {}
        for det in ("v1", "v2"):
            ph = posthoc_detect(sess_by_det[det], scorers[det], threshold=0.5)
            det_by = {r.session_id: r for r in ph}
            cnt = defaultdict(int)
            for s in sess_by_det[det]:
                if det_by[str(s["session_id"])].detected:
                    cnt[s["_family"]] += 1
            recomputed_pf[det] = {k: cnt.get(k, 0) for k in committed["per_family_posthoc"][det]}
        perfamily_match = all(recomputed_pf[det] == committed["per_family_posthoc"][det] for det in ("v1", "v2"))
        plan_match = (plan_fam == committed["family_counts"])
        nsess_match = (len(sess_v1) == committed["n_sessions"])
        coverage_ok = (n_unmatched == 0)
        valid = bool(perfamily_match and plan_match and nsess_match and coverage_ok)
        all_valid = all_valid and valid

        # ── new operating-point metrics ───────────────────────────────────────────
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
        ablation = ablation_grouped_cv(sess_v1, fpr_target=args.fpr, n_splits=5)  # v1 (default) feature view

        by_env[env] = {
            "log": str(Path(logs[env]).relative_to(args.repo_root)),
            "n_sessions": len(sess_v1), "n_unmatched": n_unmatched,
            "planned_family_counts": plan_fam,
            "observed_family_counts": dict(Counter(s["_family"] for s in sess_v1)),
            "validation": {"ok": valid, "plan_match": plan_match, "nsess_match": nsess_match,
                           "coverage_ok": coverage_ok, "perfamily_match": perfamily_match,
                           "committed_per_family_posthoc": committed["per_family_posthoc"],
                           "recomputed_per_family_posthoc": recomputed_pf},
            "frozen": frozen, "ablation": ablation,
        }

    res_dir = Path(args.repo_root) / "results" / "b7_metrics"
    res_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "ok" if all_valid else "validation_failed",
               "target": "gpt2-medium@607022bd", "regime": "finetune_realistic_owt_1ep",
               "fpr_target": args.fpr, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Re-analysis of committed evidence; no model run, no existing artefact modified. "
                       "v3/v4 are not in this committed battery (v3 lives in the specialization run).",
               "by_environment": by_env}
    out = res_dir / "targetB.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== Operating-point re-analysis — TARGET B (committed evidence) ==========")
    for env in ("E0", "E1"):
        a = by_env[env]
        v = a["validation"]
        print(f"\n[{env}] sessions={a['n_sessions']} unmatched={a['n_unmatched']} "
              f"observed_families={a['observed_family_counts']}")
        print(f"  self-validation: {'OK' if v['ok'] else 'FAIL'} (plan_match={v['plan_match']} "
              f"nsess_match={v['nsess_match']} coverage_ok={v['coverage_ok']} perfamily_match={v['perfamily_match']})")
        if not v["ok"]:
            print(f"    committed per-family posthoc: {v['committed_per_family_posthoc']}")
            print(f"    recomputed per-family posthoc: {v['recomputed_per_family_posthoc']}")
            print("  [!] reconstruction did NOT validate — new metrics NOT trustworthy.")
        for det in ("v1", "v2"):
            f = a["frozen"][det]
            print(f"    {det}: AUC={f['auc']} TPR@FPR{args.fpr}={f['tpr']} (fpr={f['fpr']}) "
                  f"[posthoc det/FA@0.5={f['posthoc_det@0.5']}/{f['posthoc_fa@0.5']}] per_family={f['per_family']}")
        abl = a["ablation"]
        print(f"    ablation (v1 view) length={len(abl['partition']['length'])}f "
              f"behavioral={len(abl['partition']['behavioral'])}f")
        for k in ("all", "length_only", "behavioral_only"):
            d = abl[k]
            print(f"      {k:16} AUC={d.get('auc')} TPR@FPR{args.fpr}={d.get('tpr')} (n_features={d.get('n_features')})")
    print(f"\n[{'✓' if all_valid else '✗'}] summary: {out}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
