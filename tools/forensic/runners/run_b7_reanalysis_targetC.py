#!/usr/bin/env python3
"""Operating-point RE-ANALYSIS of the COMMITTED TARGET C campaign (no GPU).

Reads the already-committed forensic streams of TARGET C
(``evidence/targetC_pile/E0_*.jsonl`` / ``E1_*.jsonl``) READ-ONLY and recomputes
the detection numbers the per-campaign summary lacks: ROC-AUC + TPR@FPR +
per-family at a realistic operating point, plus a length-vs-behavioural ablation.
No model is loaded; the target is never re-run.

The forensic stream carries no labels (two-tier discipline), so the session-level
ground truth is RECONSTRUCTED deterministically by rebuilding the exact campaign
plan with the same builders and pseudonymising the raw session ids with the same
salt as ``run_targetC_pythia_pile_real.py``. The reconstruction is SELF-VALIDATED:
the script first reproduces the committed family counts and detection/false-alarm
at threshold 0.5; only if these match does it trust the new operating-point
metrics. Nothing existing is modified — output goes to ``results/b7_metrics/``.

Reproduce (CPU only; needs the committed evidence + the MIMIR github cache):
    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
      thesis python3 tools/forensic/runners/run_b7_reanalysis_targetC.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_CACHE", "/workspace/data/mimir-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Must match run_targetC_pythia_pile_real.py exactly (used to reproduce session ids).
SALT = b"targetC-pythia-pile-salt-32bytes!"
# Full-run benign session mix (non-smoke), copied from the committed runner.
BENIGN_MIX = dict(n_short=18, n_multi=8, multi_size=5, n_codecomplete=16,
                  n_check=14, n_coverage=6, coverage_size=20)


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _git_tracked(repo: str, subdir: str) -> list[str]:
    try:
        out = subprocess.run(["git", "-C", repo, "ls-files", subdir],
                             capture_output=True, text=True, check=True)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def _pick_logs(repo: str, subdir: str) -> dict[str, str]:
    """Resolve the COMMITTED E0/E1 forensic streams (fallback: largest on disk)."""
    base = Path(repo) / subdir
    tracked = _git_tracked(repo, subdir)
    out: dict[str, str] = {}
    for env in ("E0", "E1"):
        cand = [t for t in tracked if Path(t).name.startswith(f"{env}_") and t.endswith(".jsonl")]
        if cand:
            out[env] = str(Path(repo) / cand[0])
        else:  # fallback: untracked tree — pick the largest matching file
            disk = sorted(base.glob(f"{env}_*.jsonl"), key=lambda p: p.stat().st_size, reverse=True)
            if disk:
                out[env] = str(disk[0])
    return out


def _reconstruct_gt(repo: str, domain: str, n_members: int, n_nonmembers: int,
                    chunk_size: int) -> tuple[dict[str, dict], list[str]]:
    """Rebuild the campaign plan -> (map pseudo session_id -> {is_attack, family},
    arrival-ordered list of pseudo session_ids). The order replicates the committed
    runner's ``order`` (static_base first-seen, then sampling, then adaptive) so the
    stateful v4 can be replayed with the same history it had live."""
    from forensic.hashing import pseudonymize
    from forensic.mia_pile import build_mia_pile_plan, find_mimir_arrow, load_mimir_targets
    from forensic.mia_score import build_mia_score_plan
    from forensic.pile_detector import build_benign_sessions

    arrow = find_mimir_arrow(domain, repo_root=repo)
    if arrow is None:
        raise SystemExit(f"[!] MIMIR cache for '{domain}' not found under {repo}")
    targets = load_mimir_targets(arrow, domain=domain, n_members=n_members, n_nonmembers=n_nonmembers)
    members = [t for t in targets if t.is_member]
    nonmembers = [t for t in targets if not t.is_member]
    secret_targets = [t for t in members if getattr(t, "is_secret_bearing", False)]
    nonmember_corpus = [t.full_text for t in nonmembers]
    all_targets = members + nonmembers

    raw: dict[str, dict] = {}
    order_raw: list[str] = []
    seen: set[str] = set()

    def add(sid: str, is_attack: bool, family: str) -> None:
        if sid not in seen:
            seen.add(sid)
            order_raw.append(sid)
        e = raw.setdefault(sid, {"is_attack": False, "family": None})
        if is_attack:
            e["is_attack"] = True
            if e["family"] in (None, "benign"):
                e["family"] = family
        elif e["family"] is None:
            e["family"] = family

    for i in range(0, len(all_targets), 25):
        chunk = all_targets[i:i + 25]
        for c in build_mia_score_plan(chunk, session_prefix=f"tcmia-{i // 25:02d}"):
            if c.groundtruth.get("is_attack"):
                add(c.body["session_id"], True,
                    c.groundtruth.get("attack_family") or "pretraining_membership_inference_scorebased")
    for i in range(0, len(secret_targets), chunk_size):
        chunk = secret_targets[i:i + chunk_size]
        for c in build_mia_pile_plan(chunk, session_prefix=f"tcext-{i // chunk_size:02d}", max_tokens=64):
            if c.groundtruth.get("is_attack"):
                add(c.body["session_id"], True,
                    c.groundtruth.get("attack_family") or "pretraining_membership_inference")
    for c in build_benign_sessions(corpus_texts=nonmember_corpus, session_prefix="tcben", **BENIGN_MIX):
        add(c.body["session_id"], False, "benign")
    for t in secret_targets:
        add(f"samp-{t.target_id}", True, "sampling")
        add(f"adap-{t.target_id}", True, "adaptive")

    gt_map = {pseudonymize(sid, SALT): v for sid, v in raw.items()}
    order = [pseudonymize(sid, SALT) for sid in order_raw]
    return gt_map, order


def _session_rows(log_path: str, gt_map: dict[str, dict]) -> tuple[list[dict], int]:
    """Rebuild session feature rows from the stream and attach reconstructed labels."""
    from forensic.features import build_features
    from forensic.investigation import augment_feature_rows_with_pile_secrets
    from forensic.pile_detector import aggregate_sessions
    from forensic.pipeline import _structural_anti_leak

    records = _read(log_path)
    _structural_anti_leak(records)
    completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                 for r in records if r["event_type"] == "completion"}
    feat_records = [r for r in records if not (
        r["event_type"] == "prompt"
        and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
    # Synthesise per-request ground truth FROM THE STREAM KEYS, with the label taken
    # from the reconstructed session map (keys are not labels -> not circular). This
    # satisfies build_features' (session_id, seq, endpoint) join without re-running.
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
            "is_attack": is_atk,
            "attack_family": (info["family"] if (info and is_atk) else None),
            "objective": None,
        })
    feats = augment_feature_rows_with_pile_secrets(build_features(feat_records, gt_records), records)
    sess = aggregate_sessions(feats)
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
    path = Path(repo) / "results" / "targetC_pile" / f"targetC_battery_{env}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    r = d["by_environment"][env]
    posthoc = {n: {"det": r["detectors"][n]["posthoc"]["detection_rate"],
                   "fa": r["detectors"][n]["posthoc"]["false_alarm_rate"]}
               for n in ("v1", "v2", "v3")}
    v4 = {"det": r["v4"]["online_detection"], "fa": r["v4"]["online_false_alarm"]}
    fam = {fr["family"]: fr["n"] for fr in r.get("per_family", [])}
    return {"posthoc": posthoc, "v4": v4, "family_counts": fam, "n_sessions": r.get("n_sessions")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default="/workspace")
    ap.add_argument("--domain", default="github")
    ap.add_argument("--n-members", type=int, default=1000)
    ap.add_argument("--n-nonmembers", type=int, default=1000)
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--detectors-dir", default="/workspace/results/detectors")
    ap.add_argument("--fpr", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=2e-3, help="tolerance for the self-validation match")
    ap.add_argument("--gt", default=None, help="load reconstructed ground truth from this JSON (self-contained replay, no MIMIR)")
    ap.add_argument("--dump-gt", default=None, help="reconstruct gt from MIMIR, write it to this JSON, and exit")
    args = ap.parse_args()

    from forensic.b7_operating_point import ablation_grouped_cv, auc_and_op
    from forensic.detector_adaptive import AdaptiveDetector
    from forensic.detector_store import load_scorer
    from forensic.online_detector import detection_metrics, posthoc_detect

    det_dir = Path(args.detectors_dir)
    scorers = {}
    for n in ("v1", "v2", "v3"):
        p = det_dir / f"{n}.joblib"
        if not p.exists():
            print(f"[!] missing frozen detector {n} in {det_dir}")
            return 2
        scorers[n], _names, _prov = load_scorer(p)

    logs = _pick_logs(args.repo_root, "evidence/targetC_pile")
    if "E0" not in logs or "E1" not in logs:
        print(f"[!] could not resolve committed E0/E1 streams under evidence/targetC_pile (found {list(logs)})")
        return 2

    if args.gt and Path(args.gt).exists():
        print(f"[i] loading TARGET C ground truth from {args.gt} (self-contained, no MIMIR)…")
        _g = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        gt_map, order = _g["gt_map"], _g["order"]
    else:
        print("[i] reconstructing TARGET C ground truth from the MIMIR plan (CPU)…")
        gt_map, order = _reconstruct_gt(
            args.repo_root, args.domain, args.n_members, args.n_nonmembers, args.chunk_size)
        if args.dump_gt:
            Path(args.dump_gt).parent.mkdir(parents=True, exist_ok=True)
            Path(args.dump_gt).write_text(
                json.dumps({"gt_map": gt_map, "order": order}), encoding="utf-8")
            print(f"[i] dumped ground truth to {args.dump_gt} ({len(gt_map)} sessions); exiting.")
            return 0
    plan_fam = dict(Counter(v["family"] for v in gt_map.values()))
    print(f"[i] reconstructed {len(gt_map)} session labels; planned families={plan_fam}")

    by_env: dict[str, dict] = {}
    all_valid = True
    for env in ("E0", "E1"):
        sess, n_unmatched = _session_rows(logs[env], gt_map)
        committed = _committed(args.repo_root, env)

        # ── self-validation ──────────────────────────────────────────────────────
        # The committed summary's per_family is the PLANNED count (gt-level); n_sessions
        # is the OBSERVED stream count (E1 drops rate-limited sessions). So validate:
        #  (a) planned families == committed per_family   [plan vs plan]
        #  (b) observed n_sessions == committed n_sessions [stream coverage]
        #  (c) every stream session got a label            [n_unmatched == 0]
        #  (d) det/FA@0.5 reproduced exactly  -> the RIGOROUS label-correctness proof.
        observed_fam = dict(Counter(s["_family"] for s in sess))
        recomputed = {}
        for name, fn in scorers.items():
            m = detection_metrics(posthoc_detect(sess, fn, threshold=0.5))
            recomputed[name] = {"det": round(m["detection_rate"], 6), "fa": round(m["false_alarm_rate"], 6)}
        plan_match = (plan_fam == committed["family_counts"])
        nsess_match = (len(sess) == committed["n_sessions"])
        coverage_ok = (n_unmatched == 0)
        valid_det = all(
            abs(recomputed[n]["det"] - committed["posthoc"][n]["det"]) <= args.tol
            and abs(recomputed[n]["fa"] - committed["posthoc"][n]["fa"]) <= args.tol
            for n in ("v1", "v2", "v3"))
        valid = bool(plan_match and nsess_match and coverage_ok and valid_det)
        all_valid = all_valid and valid

        # ── new operating-point metrics (only meaningful if validation passed) ────
        y = [int(s["label_is_attack"]) for s in sess]
        fam = [s["_family"] for s in sess]
        frozen = {}
        for name, fn in scorers.items():
            frozen[name] = auc_and_op([fn(s) for s in sess], y, fam, fpr_target=args.fpr)
        ablation = ablation_grouped_cv(sess, fpr_target=args.fpr, n_splits=5)

        # ── v4 (adaptive, online, stateful): replay in arrival order ───────────────
        # v4's score depends on history, so it must be fed in the same order it saw
        # live. We reproduce its NATIVE det/FA@0.5 (validation) AND read its score
        # at the operating point (caveat: a post-hoc reading of an online trajectory).
        sess_by = {s["session_id"]: s for s in sess}
        v4 = AdaptiveDetector(scorers["v3"], threshold=0.5)
        v4_score_by, v4_det_by = {}, {}
        for sid in order:
            row = sess_by.get(sid)
            if row is None:
                continue
            out = v4.observe_and_score(row)
            v4_score_by[sid] = out["score"]
            v4_det_by[sid] = out["detected"]
        n_atk = sum(y) or 1
        n_ben = (len(y) - sum(y)) or 1
        v4_det = sum(1 for s in sess if s["label_is_attack"] and v4_det_by.get(s["session_id"])) / n_atk
        v4_fa = sum(1 for s in sess if not s["label_is_attack"] and v4_det_by.get(s["session_id"])) / n_ben
        # v4 is online + self-retraining, so its native det/FA may differ from the
        # committed run by up to ONE borderline session (a session at score~=0.5 can
        # flip under the refit's numerical margin). The frozen detectors validate
        # EXACTLY; v4 is validated within one-session slack.
        v4_tol = max(args.tol, 1.5 / n_atk, 1.5 / n_ben)
        v4_detfa_match = (abs(round(v4_det, 6) - committed["v4"]["det"]) <= v4_tol
                          and abs(round(v4_fa, 6) - committed["v4"]["fa"]) <= v4_tol)
        v4_op = auc_and_op([v4_score_by[s["session_id"]] for s in sess], y, fam, fpr_target=args.fpr)
        v4_op["native_det"] = round(v4_det, 3)
        v4_op["native_fa"] = round(v4_fa, 3)
        v4_op["n_refits"] = v4.n_refits
        v4_op["validation_note"] = "stateful/online: det/FA validated within one-session slack"
        valid = bool(valid and v4_detfa_match)
        all_valid = all_valid and v4_detfa_match
        v4.reset()

        by_env[env] = {
            "log": str(Path(logs[env]).relative_to(args.repo_root)),
            "n_sessions": len(sess), "n_unmatched": n_unmatched,
            "planned_family_counts": plan_fam, "observed_family_counts": observed_fam,
            "validation": {"ok": valid, "plan_match": plan_match, "nsess_match": nsess_match,
                           "coverage_ok": coverage_ok, "detfa_match": valid_det,
                           "v4_detfa_match": v4_detfa_match,
                           "committed_posthoc": committed["posthoc"], "recomputed_posthoc": recomputed,
                           "committed_v4": committed["v4"],
                           "recomputed_v4": {"det": round(v4_det, 6), "fa": round(v4_fa, 6)}},
            "frozen": frozen, "v4": v4_op, "ablation": ablation,
        }

    res_dir = Path(args.repo_root) / "results" / "b7_metrics"
    res_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "ok" if all_valid else "validation_failed",
               "target": "EleutherAI/pythia-2.8b@step99000", "regime": "pretraining",
               "domain": args.domain, "fpr_target": args.fpr,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Re-analysis of committed evidence; no model run, no existing artefact modified.",
               "by_environment": by_env}
    out = res_dir / "targetC.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── console ───────────────────────────────────────────────────────────────────
    print("\n========== Operating-point re-analysis — TARGET C (committed evidence) ==========")
    for env in ("E0", "E1"):
        a = by_env[env]
        v = a["validation"]
        flag = "OK" if v["ok"] else "FAIL"
        print(f"\n[{env}] sessions={a['n_sessions']} unmatched={a['n_unmatched']} "
              f"observed_families={a['observed_family_counts']}")
        print(f"  self-validation: {flag} (plan_match={v['plan_match']} nsess_match={v['nsess_match']} "
              f"coverage_ok={v['coverage_ok']} detfa_match={v['detfa_match']} v4_detfa_match={v['v4_detfa_match']})")
        for n in ("v1", "v2", "v3"):
            print(f"    committed posthoc det/FA@0.5 {n}: "
                  f"{v['committed_posthoc'][n]['det']:.3f}/{v['committed_posthoc'][n]['fa']:.3f}  "
                  f"recomputed {v['recomputed_posthoc'][n]['det']:.3f}/{v['recomputed_posthoc'][n]['fa']:.3f}")
        if not v["ok"]:
            print("  [!] reconstruction did NOT validate — new metrics below are NOT trustworthy.")
        for n in ("v1", "v2", "v3"):
            f = a["frozen"][n]
            print(f"    {n}: AUC={f['auc']} TPR@FPR{args.fpr}={f['tpr']} (fpr={f['fpr']}) per_family={f['per_family']}")
        q = a["v4"]
        print(f"    v4: AUC={q['auc']} TPR@FPR{args.fpr}={q['tpr']} (fpr={q['fpr']}) "
              f"[native det/FA@0.5={q['native_det']}/{q['native_fa']} refits={q['n_refits']}] per_family={q['per_family']}")
        abl = a["ablation"]
        print(f"    ablation length={abl['partition']['length']}")
        print(f"             behavioral={abl['partition']['behavioral']}")
        for k in ("all", "length_only", "behavioral_only"):
            d = abl[k]
            print(f"      {k:16} AUC={d.get('auc')} TPR@FPR{args.fpr}={d.get('tpr')} (n_features={d.get('n_features')})")
    print(f"\n[{'✓' if all_valid else '✗'}] summary: {out}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
