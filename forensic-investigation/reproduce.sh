#!/usr/bin/env bash
#
# Forensic chain demonstration — timeline reconstruction, attribution heuristics, IR playbook
# and chain of custody — on a realistic MULTI-ACTOR incident.
#
# This is a cross-cutting, model-agnostic capability: the chain reads request metadata and log
# structure, not model outputs, so it is demonstrated ONCE (not per campaign). The scenario
# stages one adversary across two extraction sessions, a second adversary, and benign traffic
# from distinct legitimate users, so attribution is discriminating.
#
# Modes:
#   --replay  (default, NO GPU): re-derive timeline structure + attribution + custody live from
#             the committed forensic stream, then show the committed full IR report (phases,
#             triage severity, escalation).
#   --regen   (GPU): re-stage the incident against the model and regenerate the full report.
#
set -euo pipefail

MODE="${1:---replay}"
STREAM="evidence/investigation/1ac0d02e-05ba-46d3-89ef-8f32d8f5c1dd.jsonl"
REPORT="results/investigation/1ac0d02e-05ba-46d3-89ef-8f32d8f5c1dd_investigation.json"
SVC=thesis; [ "$MODE" = "--regen" ] && SVC=thesis-gpu
DC=(docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic "$SVC" python3)

case "$MODE" in
  --replay)
    echo "[FORENSIC] REPLAY (no GPU) — live re-derivation from the committed stream:"
    "${DC[@]}" tools/forensic/runners/run_forensic_report.py --log "$STREAM"
    echo
    echo "[FORENSIC] committed full IR report (phase tagging + triage + escalation):"
    python3 - "$REPORT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ph = {}
for s in d["timeline"].values():
    for p in s["phases_observed"]:
        ph[p] = ph.get(p, 0) + 1
ir = d["ir_report"]; tri = ir["triage"]
print(f"      timeline phases observed : {ph}")
print(f"      attribution links        : {len(d['attribution_links'])}")
print(f"      triage severity          : {tri['severity']}  (suspicious sessions: {len(tri['suspicious_sessions'])})")
print(f"      escalation               : {ir['escalation_recommendation']}")
print(f"      preservation chain ok    : {ir['preservation']['chain_verified']}  records: {ir['preservation']['total_records']}")
PY
    echo "[FORENSIC] full report: $REPORT"
    ;;

  --regen)
    echo "[FORENSIC] REGEN (GPU) — re-stage the multi-actor incident and rebuild the report…"
    "${DC[@]}" tools/forensic/runners/run_forensic_investigation_real.py \
      --model EleutherAI/pythia-1.4b --revision step99000 --domain github --n-secret 40
    echo "[FORENSIC] done — see results/investigation/"
    ;;

  *)
    echo "usage: $0 [--replay|--regen]" >&2
    exit 2
    ;;
esac
