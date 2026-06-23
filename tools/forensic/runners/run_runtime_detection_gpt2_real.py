#!/usr/bin/env python3
"""Runtime (online) detection on the NEW gpt2 environment — v1 vs v2, full battery.

The mirror, on the new gpt2 target, of the v1/Pythia live-detection experiment
(``run_attack_session_with_live_detection_real.py``). A FRESH campaign covering the
WHOLE attack battery — extraction-prefix, MIA score-based, sampling (25x burst),
adaptive multi-turn, plus mirrored benign — is launched against the fine-tuned gpt2
target THROUGH the forensic server, in E0 (no defences) and E1 (defences on), and
TWO FROZEN detector instances watch it online (``stream_detect``: running session
aggregate after each request → detection DURING the campaign + time-to-detection)
and post-hoc — never stopping anything:

  * v1 — loaded from results/detectors/v1.joblib: the Passata-1 frozen detector
          (pool EXCLUDING gpt2), with v1 secret recognition;
  * v2 — loaded from results/detectors/v2.joblib: trained INCLUDING the gpt2
          residues, AND using the v2 extended secret recogniser (PAN / CONF) on
          this traffic.

Neither detector is fit here — both are LOADED (train-once / load-many). The test
campaign is FRESHLY generated, so its sessions are in neither training pool
(out-of-sample by construction). Writes to a NEW namespace (evidence/runtime_gpt2,
results/runtime_detection); the committed artefacts are untouched.

    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
      -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e HF_HUB_CACHE=/workspace/data/mimir-cache \\
      thesis python3 tools/forensic/runners/run_runtime_detection_gpt2_real.py \\
      --registry results/canary/c630d00e-9dda-4265-ad79-e5ae39189b40_registry.json
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_CACHE", "/workspace/data/mimir-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import statistics
import uuid
from pathlib import Path

_FAMILY_NAMES = {
    "pretraining_membership_inference_scorebased": "MIA score-based",
    "pretraining_membership_inference": "Extraction (prefix)",
    "sampling": "Extraction (sampling 25x)",
    "adaptive": "Extraction (adaptive)",
    "benign": "benign",
}
_FAMILY_ORDER = [
    "pretraining_membership_inference_scorebased",
    "pretraining_membership_inference",
    "sampling",
    "adaptive",
    "benign",
]


class _ManualClock:
    """A controllable monotonic clock: the experiment drives request pacing."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _read_records(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _body(session_id, seq, prompt, max_tokens, sampling_params, surface):
    return {
        "session_id": session_id, "user_id": f"attacker-{session_id}", "prompt": prompt,
        "sequence_number": seq, "actor_type": "external_user",
        "ip_hash": f"ip-{surface}", "user_agent_hash": f"ua-{surface}", "asn_hash": f"asn-{surface}",
        "request_metadata": {"client_surface": surface},
        "sampling_params": sampling_params, "max_tokens": max_tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, help="results/canary/<run_id>_registry.json")
    ap.add_argument("--detectors-dir", default="/workspace/results/detectors")
    ap.add_argument("--envs", default="E0,E1", help="comma list: E0,E1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=12)
    ap.add_argument("--schedule", default="0.0:1,0.7:8,1.0:16", help="sampling temp:n stages")
    ap.add_argument("--budget", type=int, default=5, help="adaptive queries/canary")
    ap.add_argument("--burst-cadence", type=float, default=1.0)
    ap.add_argument("--paced-cadence", type=float, default=5.0)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--repo-root", default="/workspace")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from forensic import detector_v2 as v2
    from forensic.adaptive_attack import run_adaptive_extraction, run_sampling_extraction
    from forensic.backends_transformers import TransformersBackend
    from forensic.canary_dataset import mia_texts_from_registry
    from forensic.defenses import DefenseConfig
    from forensic.detector_store import load_scorer
    from forensic.features import build_features, normalize_text
    from forensic.hashing import pseudonymize
    from forensic.mia_pile import MiaTarget, build_mia_pile_plan, secret_spans
    from forensic.mia_score import build_mia_score_plan
    from forensic.online_detector import detection_metrics, posthoc_detect, stream_detect
    from forensic.pile_detector import aggregate_sessions, build_benign_sessions
    from forensic.pipeline import _structural_anti_leak
    from forensic.server import create_app
    from forensic.traffic import assert_no_groundtruth_in_request

    reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    ckpt = reg["checkpoint"]
    canaries = reg["canaries"]
    mem_texts, non_texts = mia_texts_from_registry(reg)
    schedule = [(float(t), int(n)) for t, n in (s.split(":") for s in args.schedule.split(","))]
    if args.smoke:
        canaries = canaries[:8]
        mem_texts, non_texts = mem_texts[:20], non_texts[:20]
        schedule = [(0.0, 1), (1.0, 3)]
    salt = b"runtime-gpt2-detect-salt-32bytes!"
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    print(f"[i] target {ckpt}; {len(canaries)} canaries; envs={envs}; schedule={schedule}"
          f"{' [SMOKE]' if args.smoke else ''}")

    # ── load the frozen instances (never re-fit) ──────────────────────────────
    det_dir = Path(args.detectors_dir)
    if not (det_dir / "v1.joblib").exists() or not (det_dir / "v2.joblib").exists():
        print(f"[!] frozen instances missing in {det_dir}. Run build_detector_instances_real.py first.")
        return 2
    v1_score, _n1, prov1 = load_scorer(det_dir / "v1.joblib")
    v2_score, _n2, prov2 = load_scorer(det_dir / "v2.joblib")
    print(f"[i] loaded v1 ({prov1.get('n_train_sessions')} sess, no gpt2) and "
          f"v2 ({prov2.get('n_train_sessions')} sess, incl gpt2) — frozen")

    # ── static plan (extraction-prefix + MIA-score + benign), built once ──────
    canary_targets = [
        MiaTarget(target_id=c["canary_id"], domain="canary", full_text=c["prefix"] + c["value"],
                  prefix=c["prefix"], suffix=c["value"], is_member=True,
                  is_secret_bearing=True, secret_kind=c["kind"])
        for c in canaries
    ]
    static_plan = []
    for i in range(0, len(canary_targets), args.chunk_size):
        chunk = canary_targets[i:i + args.chunk_size]
        static_plan += [c for c in build_mia_pile_plan(chunk, session_prefix=f"rtext-{i//args.chunk_size:02d}",
                                                       max_tokens=args.max_tokens) if c.groundtruth["is_attack"]]
    doc_targets = (
        [MiaTarget(f"mdoc-{i:04d}", "canary", t, t[:60], t[60:], True, False, None)
         for i, t in enumerate(mem_texts)]
        + [MiaTarget(f"ndoc-{i:04d}", "canary", t, t[:60], t[60:], False, False, None)
           for i, t in enumerate(non_texts)]
    )
    for i in range(0, len(doc_targets), 25):
        chunk = doc_targets[i:i + 25]
        static_plan += [c for c in build_mia_score_plan(chunk, session_prefix=f"rtmia-{i//25:02d}")
                        if c.groundtruth["is_attack"]]
    bn = dict(n_short=8, n_multi=3, multi_size=4, n_codecomplete=6, n_check=6, n_coverage=2, coverage_size=12) \
        if args.smoke else dict(n_short=18, n_multi=8, multi_size=5, n_codecomplete=16,
                                n_check=14, n_coverage=6, coverage_size=20)
    static_plan += build_benign_sessions(corpus_texts=non_texts, session_prefix="rtben", **bn)

    print("[i] loading fine-tuned checkpoint…")
    backend = TransformersBackend(model_id=ckpt, model_revision=None, torch_dtype=args.dtype).load()
    torch = backend._torch

    ev_dir = Path(args.repo_root) / "evidence" / "runtime_gpt2"
    res_dir = Path(args.repo_root) / "results" / "runtime_detection"
    for d in (ev_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _run_env(env: str) -> dict:
        is_e1 = env == "E1"
        clock = _ManualClock() if is_e1 else None
        run_id = str(uuid.uuid4())
        log_path = str(ev_dir / f"{env}_{run_id}.jsonl")
        app = create_app(
            log_path=log_path, salt=salt, run_id=run_id, experiment_phase=f"runtime_detection_{env}",
            model_id=ckpt, model_revision="main", model_hash=backend.model_hash, repo_path=args.repo_root,
            experiment_config={"simulation": "runtime_detection", "world": "gpt2_rich_canary",
                               "environment": env, "groundtruth_separate": True},
            # logprobs are NOT a detector feature and
            # temperature sampling can emit a -inf logprob the forensic logger
            # rejects; the established sampling runner also keeps them off. E0/E1
            # still differ on redaction + rate-limit, which DO drive the residues.
            backend=backend, environment=env, system_prompt="", expose_logprobs=False,
            output_filtering=is_e1, defense_config=DefenseConfig() if is_e1 else None, clock=clock,
        )
        gt: list[dict] = []
        torch.manual_seed(20260613)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260613)
        print(f"[i] [{env}] running full battery through the forensic server…")
        with TestClient(app) as client:
            # 1) static plan (extraction-prefix + MIA-score + benign)
            for case in static_plan:
                assert_no_groundtruth_in_request(case)
                sid = str(case.request_json().get("session_id", ""))
                if clock is not None:
                    clock.advance(args.paced_cadence if sid.startswith("rtben") else args.burst_cadence)
                resp = client.post(case.endpoint, json=case.request_json())
                g = case.groundtruth_json()
                g["session_id"] = pseudonymize(g["session_id"], salt)
                g["http_status"] = resp.status_code
                gt.append(g)
            # 2) sampling (25x burst per canary)
            for c in canaries:
                raw, pseu, seq = f"samp-{c['canary_id']}", None, {"n": 0}
                pseu = pseudonymize(raw, salt)

                def sample_fn(pfx, temperature, n, _raw=raw, _pseu=pseu, _seq=seq):
                    outs = []
                    sp = {"temperature": temperature} if temperature == 0 else {"temperature": temperature, "top_p": 0.95}
                    for _ in range(n):
                        _seq["n"] += 1
                        if clock is not None:
                            clock.advance(args.burst_cadence)
                        r = client.post("/v1/complete", json=_body(_raw, _seq["n"], pfx, args.max_tokens, sp, "sampling_probe"))
                        outs.append((r.json().get("response", "") or "") if r.status_code == 200 else "")
                        gt.append({"session_id": _pseu, "sequence_number": _seq["n"], "endpoint": "/v1/complete",
                                   "is_attack": True, "attack_family": "sampling",
                                   "scenario": "sampling_extraction", "objective": "extract_secret"})
                    return outs

                run_sampling_extraction(c["prefix"], sample_fn, schedule=schedule, stop_on_candidate=False)
            # 3) adaptive (multi-turn per canary)
            for c in canaries:
                raw, seq = f"adap-{c['canary_id']}", {"n": 0}
                pseu = pseudonymize(raw, salt)

                def query_fn(context, mtok, _raw=raw, _pseu=pseu, _seq=seq):
                    _seq["n"] += 1
                    if clock is not None:
                        clock.advance(args.burst_cadence)
                    r = client.post("/v1/complete", json=_body(_raw, _seq["n"], context, mtok, {"temperature": 0.0}, "adaptive_probe"))
                    gt.append({"session_id": _pseu, "sequence_number": _seq["n"], "endpoint": "/v1/complete",
                               "is_attack": True, "attack_family": "adaptive",
                               "scenario": "adaptive_extraction", "objective": "extract_secret"})
                    return (r.json().get("response", "") or "") if r.status_code == 200 else "", None

                run_adaptive_extraction(c["prefix"], query_fn, budget=args.budget, max_tokens=args.max_tokens)

        records = _read_records(log_path)
        _structural_anti_leak(records)
        # E1 blocked requests log a prompt but no completion → feed build_features
        # only completed requests (what actually got through), as the detector pass does.
        completed = {
            (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"])
            for r in records if r["event_type"] == "completion"
        }
        feat_records = [
            r for r in records
            if not (r["event_type"] == "prompt"
                    and (r["session_id"], int(r["payload"]["sequence_number"]), r["payload"]["endpoint"]) not in completed)
        ]
        feat_v1 = build_features(feat_records, gt)

        # v2 view: recompute the secret-like boolean with v2 recognition
        resp_by_key = {}
        for rec in records:
            if rec.get("event_type") == "completion":
                p = rec["payload"]
                resp_by_key[(str(rec["session_id"]), int(p.get("sequence_number", 0)))] = p.get("response_raw") or ""
        feat_v2 = []
        for r in feat_v1:
            raw = resp_by_key.get((str(r.get("session_id")), int(r.get("sequence_number", 0))))
            if raw is not None:
                r = {**r, "feature_response_contains_secret_like_pattern": v2.contains_secret_like_v2(raw)}
            feat_v2.append(r)

        fam_by_session: dict[str, str] = {}
        for g in gt:
            sid = g["session_id"]
            fam_by_session[sid] = g.get("attack_family") or "unknown" if g.get("is_attack") \
                else fam_by_session.get(sid, "benign")

        # oracle: which canary values actually leaked anywhere this campaign
        all_resps = list(resp_by_key.values())
        joined = normalize_text(" ".join(all_resps))
        secrets_by_kind: dict[str, int] = {}
        for c in canaries:
            if normalize_text(c["value"]) in joined:
                secrets_by_kind[c["kind"]] = secrets_by_kind.get(c["kind"], 0) + 1

        def _eval(feat, score):
            rows = sorted(feat, key=lambda r: (str(r["session_id"]), int(r.get("sequence_number", 0))))
            online = stream_detect(rows, score, threshold=args.threshold)
            posthoc = posthoc_detect(aggregate_sessions(feat), score, threshold=args.threshold)
            return online, posthoc

        def _per_family(online, posthoc):
            on_by = {r.session_id: r for r in online}
            ph_by = {r.session_id: r for r in posthoc}
            fams: dict[str, dict] = {}
            for sid, fam in fam_by_session.items():
                d = fams.setdefault(fam, {"n": 0, "on": 0, "ph": 0, "ttd": []})
                d["n"] += 1
                on, ph = on_by.get(sid), ph_by.get(sid)
                if on and on.detected:
                    d["on"] += 1
                    if on.detected_at_request is not None:
                        d["ttd"].append(on.detected_at_request)
                if ph and ph.detected:
                    d["ph"] += 1
            rows = []
            for fam in _FAMILY_ORDER:
                if fam not in fams:
                    continue
                d = fams[fam]
                rows.append({"family": fam, "name": _FAMILY_NAMES.get(fam, fam), "n_sessions": d["n"],
                             "online_detected": d["on"], "posthoc_detected": d["ph"],
                             "online_ttd_median": statistics.median(d["ttd"]) if d["ttd"] else None})
            return rows

        out = {}
        for name, feat, score in (("v1", feat_v1, v1_score), ("v2", feat_v2, v2_score)):
            online, posthoc = _eval(feat, score)
            out[name] = {"per_family": _per_family(online, posthoc),
                         "online_metrics": detection_metrics(online),
                         "posthoc_metrics": detection_metrics(posthoc)}
        n_sessions = len({g["session_id"] for g in gt})
        return {"env": env, "run_id": run_id, "n_requests": len(gt), "n_sessions": n_sessions,
                "secrets_leaked_by_kind": secrets_by_kind, "v1": out["v1"], "v2": out["v2"]}

    results = {env: _run_env(env) for env in envs}

    summary = {
        "status": "ok", "target": ckpt, "smoke": args.smoke, "threshold": args.threshold,
        "v1_provenance": prov1, "v2_provenance": prov2, "by_environment": results,
    }
    out_path = res_dir / f"runtime_battery{'_smoke' if args.smoke else ''}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── console ───────────────────────────────────────────────────────────────
    for env in envs:
        r = results[env]
        print(f"\n========== {env} — {r['n_sessions']} sessions, {r['n_requests']} requests ==========")
        print(f"  secrets leaked by kind: {r['secrets_leaked_by_kind'] or '{}'}")
        print(f"  {'detector':4} {'family':26} {'ONLINE':>9} {'TTD':>5} {'POST-HOC':>9}")
        for name in ("v1", "v2"):
            for row in r[name]["per_family"]:
                ttd = f"{row['online_ttd_median']:.0f}" if row["online_ttd_median"] is not None else "-"
                print(f"  {name:4} {row['name'][:26]:26} {row['online_detected']:>4}/{row['n_sessions']:<4} "
                      f"{ttd:>5} {row['posthoc_detected']:>4}/{row['n_sessions']:<4}")
        for name in ("v1", "v2"):
            m = r[name]["online_metrics"]
            print(f"  {name} ONLINE: detection {m['detection_rate']:.3f}  false-alarm {m['false_alarm_rate']:.3f}")
    print(f"\n[✓] report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
