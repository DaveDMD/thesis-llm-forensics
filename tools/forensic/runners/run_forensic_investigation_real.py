#!/usr/bin/env python3
"""End-to-end forensic INVESTIGATION on the Pythia/Pile case.

Stages a realistic multi-session campaign by ONE adversary (two extraction
sessions from the same actor + a divergence session from a second actor) plus
benign traffic from distinct legitimate users, runs it through the forensic
server against REAL Pythia, then EXERCISES the three forensic modules on the
collected residues:

  1. timeline reconstruction  — correlate each session's query series into phases
  2. attribution heuristics    — prudential linking of sessions to a shared actor
  3. IR playbook               — triage / snapshot / preservation / report

and emits a structured IR report (JSON) + a human-readable markdown writeup.
No download/training. Run inside docker (GPU), e.g.::

    docker compose run --rm thesis \\
        python3 tools/forensic/runners/run_forensic_investigation_real.py \\
        --model EleutherAI/pythia-1.4b --revision step99000 --domain github --n-secret 40
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
from pathlib import Path


def _read_records(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _find_model_dir(model_id: str, revision: str, repo_root: str) -> Path | None:
    base = Path(repo_root) / "data" / "mimir-cache" / ("models--" + model_id.replace("/", "--"))
    ref = base / "refs" / revision
    if ref.exists():
        d = base / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if d.exists():
            return d
    snaps = sorted((base / "snapshots").glob("*")) if (base / "snapshots").exists() else []
    return snaps[0] if snaps else None


def _model_artifacts(model_dir: Path | None, *, include_weights: bool) -> dict[str, Path]:
    if model_dir is None:
        return {}
    arts: dict[str, Path] = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        p = model_dir / name
        if p.exists():
            arts[name] = p
    if include_weights:
        for p in sorted(list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))):
            arts[p.name] = p
    return arts


def _benign_plan():
    """A handful of benign sessions from DISTINCT legitimate users."""
    from forensic.pile_detector import build_benign_sessions

    return build_benign_sessions(
        session_prefix="legit", n_short=6, n_multi=1, multi_size=3,
        n_codecomplete=0, n_check=0, n_coverage=0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--revision", default="step99000")
    ap.add_argument("--domain", default="github", choices=["github", "arxiv", "wikipedia_(en)"])
    ap.add_argument("--n-secret", type=int, default=40, help="secret-bearing members to probe")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--no-hash-weights", action="store_true", help="skip hashing model weight shards")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from forensic.attribution import correlate_sessions
    from forensic.backends_transformers import TransformersBackend
    from forensic.features import build_features
    from forensic.hashing import pseudonymize
    from forensic.investigation import augment_feature_rows_with_pile_secrets
    from forensic.mia_pile import (
        build_divergence_plan,
        build_mia_pile_plan,
        find_mimir_arrow,
        load_mimir_targets,
    )
    from forensic.pipeline import _structural_anti_leak
    from forensic.playbook import run_playbook
    from forensic.server import create_app
    from forensic.timeline import multi_phase_sessions, reconstruct_timelines
    from forensic.traffic import assert_no_groundtruth_in_request
    from forensic.verifier import EvidenceVerifier

    salt = b"forensic-investigation-salt-32by!"
    run_id = str(uuid.uuid4())

    arrow = find_mimir_arrow(args.domain, repo_root=args.repo_root)
    if arrow is None:
        print(f"[!] MIMIR cache for '{args.domain}' not found")
        return 2
    # secret-bearing members are the ones that leak under extraction
    targets = load_mimir_targets(
        arrow, domain=args.domain, n_members=args.n_secret, n_nonmembers=8, secret_only=True
    )
    members = [t for t in targets if t.is_member]
    if len(members) < 4:
        print(f"[!] too few secret-bearing members ({len(members)})")
        return 2
    half = len(members) // 2
    chunk_a, chunk_b = members[:half], members[half:]

    # ── campaign plan ────────────────────────────────────────────────────────
    # adversary #1 runs TWO extraction sessions (same actor hash, via the
    # builder's constant attacker user_id); adversary #2 runs a divergence
    # session (distinct actor); benign sessions come from distinct legit users.
    plan = []
    plan += [c for c in build_mia_pile_plan(chunk_a, session_prefix="campA", max_tokens=64)
             if c.groundtruth["is_attack"]]
    plan += [c for c in build_mia_pile_plan(chunk_b, session_prefix="campB", max_tokens=64)
             if c.groundtruth["is_attack"]]
    plan += [c for c in build_divergence_plan(session_prefix="campD", repeat=40, max_tokens=120)
             if c.groundtruth["is_attack"]]
    plan += _benign_plan()
    print(f"[i] campaign: {len(plan)} requests "
          f"(extraction A={len(chunk_a)}, B={len(chunk_b)}, divergence + benign)")

    # ── load model + locate artifacts for chain-of-custody snapshot ──────────
    print(f"[i] loading {args.model}@{args.revision} (offline)…")
    backend = TransformersBackend(
        model_id=args.model, model_revision=args.revision, torch_dtype=args.dtype,
    ).load()
    model_dir = _find_model_dir(args.model, args.revision, args.repo_root)
    manifest_arts = _model_artifacts(model_dir, include_weights=False)   # small, for the manifest
    snapshot_arts = _model_artifacts(model_dir, include_weights=not args.no_hash_weights)

    ev_dir = Path(args.repo_root) / "evidence" / "investigation"
    res_dir = Path(args.repo_root) / "results" / "investigation"
    for d in (ev_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)
    log_path = str(ev_dir / f"{run_id}.jsonl")

    app = create_app(
        log_path=log_path, salt=salt, run_id=run_id, experiment_phase="forensic_investigation",
        model_id=args.model, model_revision=args.revision, model_hash=backend.model_hash,
        repo_path=args.repo_root,
        experiment_config={"simulation": "forensic_investigation", "world": "pythia_pile",
                           "groundtruth_separate": True},
        backend=backend, environment="E0", system_prompt="", expose_logprobs=True,
        model_artifacts=manifest_arts, dataset_paths={args.domain: Path(arrow)},
    )

    print("[i] running campaign through the forensic server…")
    gt_records: list[dict] = []
    with TestClient(app) as client:
        for n, case in enumerate(plan, start=1):
            assert_no_groundtruth_in_request(case)
            resp = client.post(case.endpoint, json=case.request_json())
            resp.raise_for_status()
            body = resp.json()
            gt = case.groundtruth_json()
            gt["session_id"] = pseudonymize(gt["session_id"], salt)
            gt.update({
                "prompt_record_hash": body.get("prompt_record_hash"),
                "completion_record_hash": body.get("completion_record_hash"),
                "http_status": resp.status_code,
            })
            gt_records.append(gt)
            if n % 20 == 0:
                print(f"      {n}/{len(plan)} requests…")

    # ── read residues + build observable features (Pile-secret augmented) ────
    records = _read_records(log_path)
    _structural_anti_leak(records)
    feature_rows = build_features(records, gt_records)
    feature_rows = augment_feature_rows_with_pile_secrets(feature_rows, records)

    # ── (1) timeline reconstruction ──────────────────────────────────────────
    timelines = reconstruct_timelines(records, feature_rows=feature_rows)
    mp = multi_phase_sessions(timelines)

    # ── (2) attribution heuristics ───────────────────────────────────────────
    links = correlate_sessions(records, min_confidence=0.3)

    # ── (3) IR playbook ──────────────────────────────────────────────────────
    report = run_playbook(
        log_path=log_path, forensic_records=records, feature_rows=feature_rows,
        model_artifacts=snapshot_arts, index_paths={f"mimir_{args.domain}": Path(arrow)},
        attribution_min_confidence=0.5,
    )

    # ── verify chain independently for the console ───────────────────────────
    verification = EvidenceVerifier(log_path).verify()

    # ── assemble + persist ───────────────────────────────────────────────────
    timeline_summary = {
        sid: {
            "n_events": tl.as_dict()["n_events"],
            "phases_observed": tl.phases_observed,
            "duration_seconds": tl.duration_seconds,
            "actor_consistent": tl.as_dict()["actor_consistent"],
            "n_extraction_events": sum(1 for e in tl.events if e.phase == "extraction_attempt"),
        }
        for sid, tl in sorted(timelines.items())
    }
    out = {
        "run_id": run_id,
        "model": args.model, "revision": args.revision, "domain": args.domain,
        "n_requests": len(records and feature_rows),
        "timeline": timeline_summary,
        "multi_phase_sessions": mp,
        "attribution_links": [l.as_dict() for l in links],
        "ir_report": report,
        "chain_verified": verification.ok,
    }
    (res_dir / f"{run_id}_investigation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ── console summary ──────────────────────────────────────────────────────
    print("\n[✓] (1) TIMELINE RECONSTRUCTION (per session):")
    for sid, s in timeline_summary.items():
        ph = ",".join(s["phases_observed"]) or "benign"
        print(f"      {sid[:18]:18}  n={s['n_events']:>3}  phases=[{ph}]  "
              f"extraction_events={s['n_extraction_events']}")
    print(f"      multi-phase sessions: {mp or '(none)'}")

    print("\n[✓] (2) ATTRIBUTION (prudential correlation, not identity):")
    if not links:
        print("      (no cross-session links above threshold)")
    for l in links:
        print(f"      {l.session_a[:18]} ~ {l.session_b[:18]}  "
              f"conf={l.confidence:.2f}  signals={l.signals}")

    print("\n[✓] (3) IR PLAYBOOK:")
    tri = report["triage"]
    pres = report["preservation"]
    snap = report["snapshot"]
    print(f"      severity={tri['severity']}  suspicious={len(tri['suspicious_sessions'])}  "
          f"links={tri['n_attribution_links']}  phase_summary={tri['phase_summary']}")
    print(f"      snapshot: {len(snap['components'])} components, aggregate_digest={snap['aggregate_digest'][:16]}…")
    print(f"      preservation: chain_verified={pres['chain_verified']}  "
          f"records={pres['total_records']}  manifest_first={pres['manifest_first']}  "
          f"ots_sidecar={pres['ots_sidecar_present']}")
    print(f"      escalation: {report['escalation_recommendation']}")

    print(f"\n[✓] forensic stream: {log_path}")
    print(f"[✓] investigation report: {res_dir / (run_id + '_investigation.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
