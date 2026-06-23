#!/usr/bin/env bash
#
# Reproduce TARGET C — Pythia-2.8b pre-trained on The Pile (no fine-tune), attacked with the
# MIMIR membership set and Pile-resident secrets. The pre-training regime: MIA is moderate and
# domain-dependent, yet extraction is strong (secrets memorized by Pile duplication).
#
# Modes:
#   --replay  (default, NO GPU): re-score the committed targetC streams with the frozen
#             detectors v1/v2/v3 (runtime + post-hoc) and read back the attack battery results.
#   --regen   (GPU): re-run the full battery x {E0,E1} against Pythia-2.8b with the online
#             detectors (v1/v2/v3 frozen + v4 adaptive). No fine-tune — the target is the
#             public pre-trained checkpoint; MIMIR data is read from the HF cache.
#
set -euo pipefail

MODE="${1:---replay}"
SVC=thesis; [ "$MODE" = "--regen" ] && SVC=thesis-gpu
DC=(docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic "$SVC" python3)
RUN="tools/forensic/runners"

case "$MODE" in
  --replay)
    echo "[C] REPLAY (no GPU) — re-scoring committed targetC streams with frozen detectors v1/v2/v3…"
    "${DC[@]}" "$RUN/run_b7_reanalysis_targetC.py" --gt results/targetC_pile/groundtruth.json
    echo
    echo "[C] DETECTOR metrics:  results/b7_metrics/targetC.json"
    echo "[C] ATTACK battery:    results/targetC_pile/targetC_battery_{E0,E1}.json"
    echo "[C] Compare against headline numbers in campaigns/C/manifest.json."
    ;;

  --regen)
    echo "[C] REGEN (GPU) — full battery against pre-trained Pythia-2.8b…"
    for E in E0 E1; do
      "${DC[@]}" "$RUN/run_targetC_pythia_pile_real.py" \
        --model EleutherAI/pythia-2.8b --revision step99000 \
        --domain github --n-members 1000 --n-nonmembers 1000 --envs "$E"
    done
    echo "[C] done — see results/targetC_pile/ and results/{mia_score,mia_pile,sidechannel_mia}/"
    echo "[C] (use --smoke / --mia-sample / --max-secrets on the runner to lower cost)"
    ;;

  --forensics)
    echo "[C] CUSTODY (no GPU) — chain integrity + attribution + timeline structure on the committed stream…"
    "${DC[@]}" "$RUN/run_forensic_report.py" --log evidence/targetC_pile/E0_95bd646d-5f3e-4c40-8355-17c97e21936f.jsonl
    echo "[C] full multi-actor forensic chain (phases/attribution/IR playbook): forensic-investigation/reproduce.sh --replay"
    ;;

  *)
    echo "usage: $0 [--replay|--regen|--forensics]" >&2
    exit 2
    ;;
esac
