#!/usr/bin/env bash
#
# Reproduce TARGET A — gpt2-medium fine-tuned on WikiText for 12 epochs (heavy overfit,
# registry id c630d00e). The upper-bound memorization case: MIA is perfect and extraction
# is rich, which makes the effect of realistic defences (E1) stand out cleanly.
#
# Modes:
#   --replay  (default, NO GPU): re-score A's recovered forensic streams with the frozen
#             detectors (runtime + post-hoc). A's streams share evidence/runtime_gpt2 with
#             TARGET B, so the replay engine selects them by matching model_id to c630d00e.
#   --regen   (GPU): regenerate the target (deterministic fine-tune) and run the full
#             attack battery (MIA white/black-box, prefix, sampling, adaptive x E0/E1) and
#             the runtime detector.
#
set -euo pipefail

MODE="${1:---replay}"
REG="results/canary/c630d00e-9dda-4265-ad79-e5ae39189b40_registry.json"
SVC=thesis; [ "$MODE" = "--regen" ] && SVC=thesis-gpu
DC=(docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic "$SVC" python3)
RUN="tools/forensic/runners"

case "$MODE" in
  --replay)
    echo "[A] REPLAY (no GPU) — re-scoring recovered streams with frozen detectors v1/v2…"
    "${DC[@]}" "$RUN/run_b7_reanalysis_targetA.py"
    echo
    echo "[A] DETECTOR metrics:    results/b7_metrics/targetA.json"
    echo "[A] ATTACK summaries:    results/mia_comparison/mia_comparison_12ep.json"
    echo "                         results/mia_blackbox/mia_blackbox_c630d00e_{E0,E1}.json"
    echo "                         results/sampling_canary/sampling_canary_summary.json"
    echo "                         results/adaptive_canary/adaptive_canary_{E0,E1}_summary.json"
    echo "[A] Compare against headline numbers in campaigns/A/manifest.json."
    ;;

  --regen)
    echo "[A] REGEN (GPU) — full pipeline from the registry…"
    echo "[A] 1/4  fine-tune the target (deterministic, heavy overfit)"
    "${DC[@]}" "$RUN/run_finetune_canary_target_real.py" \
      --base-model gpt2-medium --revision main \
      --dataset rich --background wikitext --epochs 12 \
      --n-generic-members 300 --n-nonmembers 300 --canaries-per-cell 6 --seed 20260612
    echo "[A] 2/4  MIA white-box + black-box"
    "${DC[@]}" "$RUN/run_mia_comparison_canary_real.py" --registry "$REG" --label 12ep
    for E in E0 E1; do
      "${DC[@]}" "$RUN/run_mia_blackbox_canary_real.py" --registry "$REG" --environment "$E"
    done
    echo "[A] 3/4  extraction battery (prefix / sampling / adaptive) x E0,E1"
    for E in E0 E1; do
      "${DC[@]}" "$RUN/run_attack_canary_target_real.py" --registry "$REG" --environment "$E"
      "${DC[@]}" "$RUN/run_sampling_canary_real.py"      --registry "$REG" --environment "$E"
      "${DC[@]}" "$RUN/run_adaptive_canary_real.py"      --registry "$REG" --environment "$E"
    done
    echo "[A] 4/4  runtime detector (online v1/v2) + post-hoc"
    "${DC[@]}" "$RUN/run_runtime_detection_gpt2_real.py" --registry "$REG" --envs E0,E1
    echo "[A] done."
    ;;

  --forensics)
    echo "[A] CUSTODY (no GPU) — chain integrity + attribution + timeline structure on the committed stream…"
    "${DC[@]}" "$RUN/run_forensic_report.py" --log evidence/runtime_gpt2/E0_1984562d-4d2e-45f7-a308-a63fe1a70d27.jsonl
    echo "[A] full multi-actor forensic chain (phases/attribution/IR playbook): forensic-investigation/reproduce.sh --replay"
    ;;

  *)
    echo "usage: $0 [--replay|--regen|--forensics]" >&2
    exit 2
    ;;
esac
