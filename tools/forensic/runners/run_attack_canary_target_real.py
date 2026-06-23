#!/usr/bin/env python3
"""Attacks on the CONTROLLED fine-tuned target (canary).

Loads the fine-tuned checkpoint + canary registry (from
run_finetune_canary_target_real.py), then, through the forensic server, runs:
  * EXTRACTION (prefix-continuation) over the canaries -> success BY REPETITION N
    (the memorisation-vs-repetition curve: how many secrets actually come out);
  * MIA (zlib) over member-vs-non-member docs -> AUC + members confirmed @FPR;
  * a DETECTOR pass (GroupKFold) over the campaign + benign, with the new
    anti-circular TEXTUAL features -> detection + which features discriminate.
Residues are appended to the shared pool. Perfect ground truth (we built the data).

Run inside docker (GPU)::

    docker compose run --rm thesis \\
        python3 tools/forensic/runners/run_attack_canary_target_real.py \\
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
from collections import Counter, defaultdict
from pathlib import Path


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class _ManualClock:
    """Controllable monotonic clock so request pacing is deterministic under the E1
    rate limiter (benign paced -> not throttled; attack bursts -> throttled)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="results/canary/<run_id>_registry.json")
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--environment", choices=["E0", "E1"], default="E0",
                    help="E1 = full defenses (output redaction + rate limiting + score-channel off)")
    ap.add_argument("--burst-cadence", type=float, default=1.0,
                    help="seconds between attacker requests (E1 pacing)")
    ap.add_argument("--paced-cadence", type=float, default=5.0,
                    help="seconds between benign requests (E1 pacing)")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from forensic.backends_transformers import TransformersBackend
    from forensic.canary_dataset import mia_texts_from_registry
    from forensic.defenses import DefenseConfig
    from forensic.detector_ml import build_xy, cross_validate_grouped
    from forensic.features import build_features, normalize_text
    from forensic.harvest import append_jsonl, load_jsonl
    from forensic.hashing import pseudonymize
    from forensic.mia_pile import MiaTarget, build_mia_pile_plan
    from forensic.mia_score import build_mia_score_plan, mia_zlib, roc_auc
    from forensic.mia_strata import _quantile
    from forensic.pile_detector import aggregate_sessions, build_benign_sessions
    from forensic.pipeline import _structural_anti_leak
    from forensic.server import create_app
    from forensic.traffic import assert_no_groundtruth_in_request

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ckpt = reg["checkpoint"]
    canaries = reg["canaries"]
    salt = b"canary-attack-target-salt-32byte!"
    run_id = str(uuid.uuid4())

    # member/non-member docs for the MIA — read from the self-contained rich registry
    # or rebuilt deterministically for the synthetic/Pythia one
    mem_texts, non_texts = mia_texts_from_registry(reg)
    rep_by_canary = {c["canary_id"]: c["repetition"] for c in canaries}
    val_by_canary = {c["canary_id"]: c["value"] for c in canaries}
    print(f"[i] target {ckpt}; {len(canaries)} canaries; MIA {len(mem_texts)} member / {len(non_texts)} non-member")

    print("[i] loading fine-tuned checkpoint…")
    backend = TransformersBackend(model_id=ckpt, model_revision=None, torch_dtype=args.dtype).load()

    # ── EXTRACTION plan (canaries as targets) + MIA plan (docs) + benign ──────
    canary_targets = [
        MiaTarget(target_id=c["canary_id"], domain="canary", full_text=c["prefix"] + c["value"],
                  prefix=c["prefix"], suffix=c["value"], is_member=True,
                  is_secret_bearing=True, secret_kind=c["kind"])
        for c in canaries
    ]
    plan = []
    for i in range(0, len(canary_targets), args.chunk_size):
        chunk = canary_targets[i:i + args.chunk_size]
        plan += [c for c in build_mia_pile_plan(chunk, session_prefix=f"canext-{i//args.chunk_size:02d}",
                                                max_tokens=args.max_tokens) if c.groundtruth["is_attack"]]
    doc_targets = (
        [MiaTarget(f"mdoc-{i:04d}", "canary", t, t[:60], t[60:], True, False, None)
         for i, t in enumerate(mem_texts)]
        + [MiaTarget(f"ndoc-{i:04d}", "canary", t, t[:60], t[60:], False, False, None)
           for i, t in enumerate(non_texts)]
    )
    for i in range(0, len(doc_targets), 25):
        chunk = doc_targets[i:i + 25]
        plan += [c for c in build_mia_score_plan(chunk, session_prefix=f"canmia-{i//25:02d}")
                 if c.groundtruth["is_attack"]]
    # MIRRORED benign (full-length + HIGH-VOLUME coverage, like the main pile_detector)
    # so the operating point is defensible (real FP/FN), not the too-easy AUC=1.0 regime.
    plan += build_benign_sessions(corpus_texts=non_texts, session_prefix="canben",
                                  n_short=18, n_multi=8, multi_size=5, n_codecomplete=16,
                                  n_check=14, n_coverage=6, coverage_size=20)
    print(f"[i] campaign: {len(plan)} requests")

    ev_dir = Path(args.repo_root) / "evidence" / "canary_attack"
    ev_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(ev_dir / f"{run_id}.jsonl")
    is_e1 = args.environment == "E1"
    clock = _ManualClock() if is_e1 else None
    app = create_app(
        log_path=log_path, salt=salt, run_id=run_id,
        experiment_phase=f"canary_attack_{args.environment}",
        model_id=ckpt, model_revision="finetuned", model_hash=backend.model_hash,
        repo_path=args.repo_root,
        experiment_config={"simulation": "canary_attack", "world": "controlled_canary",
                           "environment": args.environment, "groundtruth_separate": True},
        backend=backend, environment=args.environment, system_prompt="",
        expose_logprobs=not is_e1, output_filtering=is_e1,
        defense_config=DefenseConfig() if is_e1 else None, clock=clock,
    )

    print("[i] running campaign through the forensic server…")
    gt_records = []
    with TestClient(app) as client:
        for n, case in enumerate(plan, start=1):
            assert_no_groundtruth_in_request(case)
            if clock is not None:
                sid = str(case.request_json().get("session_id", ""))
                clock.advance(args.paced_cadence if sid.startswith("canben") else args.burst_cadence)
            resp = client.post(case.endpoint, json=case.request_json())
            # E1 may block attacker bursts (429): a blocked request has no body/completion
            body = resp.json() if resp.status_code == 200 else {}
            g = case.groundtruth_json()
            g["session_id"] = pseudonymize(g["session_id"], salt)
            g.update({"prompt_record_hash": body.get("prompt_record_hash") if isinstance(body, dict) else None,
                      "completion_record_hash": body.get("completion_record_hash") if isinstance(body, dict) else None,
                      "http_status": resp.status_code})
            gt_records.append(g)
            if n % 50 == 0:
                print(f"      {n}/{len(plan)} requests…")

    records = _read(log_path)
    _structural_anti_leak(records)
    # E1 may block requests (a prompt with no completion); the detector observes only
    # what got through, so drop prompt records lacking a matching completion
    completed = {(r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
                 for r in records if r["event_type"] == "completion"}
    records = [r for r in records if not (
        r["event_type"] == "prompt"
        and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)]
    feature_rows = build_features(records, gt_records)

    # ── EXTRACTION outcomes by repetition (the curve) ────────────────────────
    resp_by_key = {}
    for rec in records:
        if rec.get("event_type") == "completion":
            p = rec["payload"]
            resp_by_key[(rec["session_id"], int(p.get("sequence_number", 0)), p.get("endpoint"))] = \
                p.get("response_raw") or ""
    by_rep = defaultdict(lambda: [0, 0])   # rep -> [extracted, total]
    extracted_secrets = []
    for g in gt_records:
        if g.get("attack_family") != "pretraining_membership_inference":
            continue
        cid = g.get("target_id")
        rep = rep_by_canary.get(cid)
        if rep is None:
            continue
        resp = resp_by_key.get((g["session_id"], int(g["sequence_number"]), g["endpoint"]), "")
        ok = normalize_text(val_by_canary[cid]) in normalize_text(resp)
        by_rep[rep][1] += 1
        if ok:
            by_rep[rep][0] += 1
            extracted_secrets.append({"canary_id": cid, "kind": g.get("secret_kind"), "repetition": rep})

    # ── MIA (zlib) member vs non-member docs ─────────────────────────────────
    scores, labels = [], []
    for t in doc_targets:
        scores.append(mia_zlib(backend.score_sequence(t.full_text)))
        labels.append(1 if t.is_member else 0)
    mia_auc = roc_auc(scores, labels)
    nonm = [s for s, l in zip(scores, labels) if l == 0]
    memb = [s for s, l in zip(scores, labels) if l == 1]
    thr = _quantile(nonm, 1.0 - args.fpr) if nonm else float("inf")
    mia_confirmed = sum(1 for s in memb if s > thr)

    # ── DETECTOR (with textual features) ─────────────────────────────────────
    session_rows = aggregate_sessions(feature_rows)
    ds_xy = build_xy(session_rows)
    cv = cross_validate_grouped(ds_xy, model_name="logistic", n_splits=5)
    y_true, y_pred, y_score = cv.oof_y_true, cv.oof_y_pred, cv.oof_y_score
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    det_auc = roc_auc(list(y_score), list(y_true))
    labels_s = [s["label_is_attack"] for s in session_rows]
    feat_names = sorted({k for s in session_rows for k in s if k.startswith("feature_")})
    feat_aucs = {n: round(roc_auc([float(s.get(n) or 0.0) for s in session_rows], labels_s), 3)
                 for n in feat_names}
    feat_aucs = dict(sorted(feat_aucs.items(), key=lambda kv: -abs(kv[1] - 0.5)))

    # ── persist residues to the shared pool + summary ────────────────────────
    for s in session_rows:
        s["prov_domain"] = "rich_canary" if reg.get("dataset_kind") == "rich" else "canary"
        s["prov_model"] = f"{reg.get('base_model', 'unknown')}-finetuned"
        s["prov_run_id"] = run_id
    pool = Path(args.repo_root) / "results" / "residue_pool" / "sessions.jsonl"
    append_jsonl(pool, session_rows)

    rep_curve = {rep: {"extracted": v[0], "total": v[1],
                       "rate": round(v[0] / v[1], 3) if v[1] else 0.0}
                 for rep, v in sorted(by_rep.items())}
    ext_by_kind = Counter(s["kind"] for s in extracted_secrets)
    tot_by_kind = Counter(c["kind"] for c in canaries)
    kind_curve = {k: {"extracted": ext_by_kind.get(k, 0), "total": tot_by_kind[k]}
                  for k in sorted(tot_by_kind)}
    summary = {
        "status": "ok", "run_id": run_id, "checkpoint": ckpt,
        "extraction_by_repetition": rep_curve,
        "extraction_by_kind": kind_curve,
        "n_secrets_extracted": len(extracted_secrets),
        "mia": {"auc_zlib": round(mia_auc, 3), "members": len(memb), "non_members": len(nonm),
                "members_confirmed_at_fpr": mia_confirmed, "fpr": args.fpr},
        "detector": {"roc_auc": round(det_auc, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "n_sessions": len(session_rows), "feature_aucs": feat_aucs},
        "pool_total_sessions": len(load_jsonl(pool)),
    }
    res_dir = Path(args.repo_root) / "results" / "canary_attack"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[✓] EXTRACTION by repetition (secrets actually recovered):")
    for rep, v in rep_curve.items():
        print(f"      N={rep:>2}: {v['extracted']:>2}/{v['total']:<2}  (rate {v['rate']:.2f})")
    print(f"      total secrets extracted = {len(extracted_secrets)}")
    print("      by kind: " + ", ".join(f"{k}={v['extracted']}/{v['total']}" for k, v in kind_curve.items()))
    print(f"[✓] MIA (zlib): AUC={mia_auc:.3f}; members confirmed @FPR{args.fpr:.0%} = {mia_confirmed}/{len(memb)}")
    print(f"[✓] DETECTOR: ROC-AUC={det_auc:.3f}  FP={fp} FN={fn}  ({len(session_rows)} sessioni)")
    print("      top feature discriminanti (incl. testuali):")
    for nm, a in list(feat_aucs.items())[:8]:
        print(f"        {nm:42} AUC={a}")
    print(f"\n[✓] residues: {log_path}")
    print(f"[✓] summary: {res_dir / (run_id + '_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
