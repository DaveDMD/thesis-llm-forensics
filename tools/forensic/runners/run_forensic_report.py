#!/usr/bin/env python3
"""No-GPU forensic re-derivation on a committed evidence log.

Reproduces, from an existing forensic stream, the parts of the forensic chain that are
fully observable (no ground truth, no model):

  (1) timeline structure    — per-session request series, duration, actor consistency
  (2) attribution heuristics — prudential linking of sessions to a shared actor
                               (shared IP/ASN/user-agent hash, temporal proximity, fingerprints)
  (3) chain of custody       — independent hash-chain integrity verification

Phase tagging (e.g. ``extraction_attempt``) and IR-playbook triage severity depend on the
labelled feature view and are reported by the full investigation runner
(``run_forensic_investigation_real.py``); see the committed IR report under
``results/investigation/`` for those. Run inside docker::

    docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic \\
      thesis python3 tools/forensic/runners/run_forensic_report.py --log evidence/<campaign>/<stream>.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_records(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="committed forensic evidence stream (.jsonl)")
    ap.add_argument("--out", default=None, help="report JSON path (default results/forensic_report/<stem>.json)")
    ap.add_argument("--min-confidence", type=float, default=0.3, help="attribution link threshold")
    args = ap.parse_args()

    from forensic.attribution import correlate_sessions
    from forensic.timeline import reconstruct_timelines
    from forensic.verifier import EvidenceVerifier

    log_path = args.log
    records = _read_records(log_path)
    if not records:
        print(f"[!] no records in {log_path}")
        return 2

    # The forensic chain reads only observable records (it is label-free here).
    timelines = reconstruct_timelines(records)
    links = correlate_sessions(records, min_confidence=args.min_confidence)
    verification = EvidenceVerifier(log_path).verify()

    timeline_summary = {
        sid: {
            "n_events": tl.as_dict()["n_events"],
            "duration_seconds": tl.duration_seconds,
            "actor_consistent": tl.as_dict()["actor_consistent"],
        }
        for sid, tl in sorted(timelines.items())
    }
    out = {
        "log": str(log_path), "n_records": len(records), "n_sessions": len(timelines),
        "timeline": timeline_summary,
        "attribution_links": [l.as_dict() for l in links],
        "chain_verified": verification.ok,
    }
    out_path = Path(args.out) if args.out else Path("results") / "forensic_report" / f"{Path(log_path).stem}_forensic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    n_sess = len(timelines)
    dense = len(links) > n_sess  # single-actor traffic correlates ~every session pair
    print(f"[i] log={log_path}  records={len(records)}  sessions={n_sess}")
    print(f"\n[OK] (1) TIMELINE STRUCTURE — {n_sess} sessions reconstructed")
    print(f"\n[OK] (2) ATTRIBUTION — {len(links)} cross-session link hypotheses (conf >= {args.min_confidence})")
    if dense:
        print(f"      non-discriminating here: this stream is single-client traffic "
              f"({len(links)} links over {n_sess} sessions). Distinct-actor attribution is "
              f"demonstrated on the multi-actor incident under forensic-investigation/.")
    elif links:
        for l in links[:8]:
            print(f"      {l.session_a[:16]} ~ {l.session_b[:16]}  conf={l.confidence:.2f}  signals={l.signals}")
    else:
        print("      (no links above threshold — distinct actors or single-session traffic)")
    print(f"\n[OK] (3) CUSTODY — independent hash-chain verification: ok={verification.ok}  records={len(records)}")
    print(f"\n[OK] report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
