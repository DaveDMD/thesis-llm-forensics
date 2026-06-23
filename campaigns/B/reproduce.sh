#!/usr/bin/env bash
#
# Reproduce TARGET B — gpt2-medium fine-tuned on OpenWebText (1 epoch, realistic regime,
# registry id 607022bd). Runs the attack battery against the target and the detector
# (runtime + post-hoc), producing the attack and detection results.
#
# Two modes:
#   --replay  (default, NO GPU): re-score the committed forensic streams with the frozen
#             detectors (runtime + post-hoc) and read back the attack-efficacy summaries.
#             Demonstrates attack-vs-detection in seconds, without any model.
#   --regen   (GPU): regenerate the target from the registry (deterministic fine-tune),
#             then run the full attack battery (MIA white/black-box, prefix-greedy,
#             sampling, adaptive multi-turn) x {E0, E1} and the runtime detector.
#
# Everything runs inside the project Docker image. Heavy weights/datasets are NOT shipped;
# --regen downloads the public gpt2-medium + OpenWebText and re-creates the checkpoint.
#
set -euo pipefail

MODE="${1:---replay}"
REG="results/canary/607022bd-5f26-4b33-bbb4-e17215a883be_registry.json"
SVC=thesis; [ "$MODE" = "--regen" ] && SVC=thesis-gpu
DC=(docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic "$SVC" python3)
RUN="tools/forensic/runners"

case "$MODE" in
  --replay)
    echo "[B] REPLAY (no GPU) — re-scoring committed streams with frozen detectors v1/v2…"
    "${DC[@]}" "$RUN/run_b7_reanalysis_targetB.py"
    echo
    echo "[B] DETECTOR metrics written to:        results/b7_metrics/targetB.json"
    echo "[B] ATTACK-efficacy summaries (committed):"
    echo "      MIA white-box : results/mia_comparison/mia_comparison_owt1ep.json"
    echo "      MIA black-box : results/mia_blackbox/mia_blackbox_E0.json | _E1.json"
    echo "      sampling      : results/sampling_canary/sampling_canary_607022bd_{E0,E1}_summary.json"
    echo "      adaptive      : results/adaptive_canary/adaptive_canary_607022bd_{E0,E1}_summary.json"
    echo "[B] Compare against expected headline numbers in campaigns/B/manifest.json."
    ;;

  --regen)
    echo "[B] REGEN (GPU) — full pipeline from the registry…"
    echo "[B] 1/4  fine-tune the target (deterministic, ~10-20 min GPU)"
    "${DC[@]}" "$RUN/run_finetune_canary_target_real.py" \
      --base-model gpt2-medium --revision main \
      --dataset rich --background openwebtext --epochs 1 --disjoint-context \
      --n-generic-members 1000 --n-nonmembers 1000 --canaries-per-cell 6 \
      --repetitions 4,16,64,256 --seed 20260620
    echo "[B] 2/4  MIA white-box + black-box"
    "${DC[@]}" "$RUN/run_mia_comparison_canary_real.py" --registry "$REG" --label owt1ep
    for E in E0 E1; do
      "${DC[@]}" "$RUN/run_mia_blackbox_canary_real.py" --registry "$REG" --environment "$E"
    done
    echo "[B] 3/4  extraction battery (prefix / sampling / adaptive) x E0,E1"
    for E in E0 E1; do
      "${DC[@]}" "$RUN/run_attack_canary_target_real.py"  --registry "$REG" --environment "$E"
      "${DC[@]}" "$RUN/run_sampling_canary_real.py"       --registry "$REG" --environment "$E"
      "${DC[@]}" "$RUN/run_adaptive_canary_real.py"       --registry "$REG" --environment "$E"
    done
    echo "[B] 4/4  runtime detector (online v1/v2 over the live attack stream) + post-hoc"
    "${DC[@]}" "$RUN/run_runtime_detection_gpt2_real.py" --registry "$REG" --envs E0,E1
    echo "[B] done — see results/{mia_comparison,mia_blackbox,sampling_canary,adaptive_canary,runtime_detection}/"
    ;;

  --forensics)
    echo "[B] CUSTODY (no GPU) — chain integrity + attribution + timeline structure on the committed stream…"
    "${DC[@]}" "$RUN/run_forensic_report.py" --log evidence/runtime_gpt2/E0_3b11920b-9c13-4c4f-9ad4-5cc3e08146a3.jsonl
    echo "[B] full multi-actor forensic chain (phases/attribution/IR playbook): forensic-investigation/reproduce.sh --replay"
    ;;

  *)
    echo "usage: $0 [--replay|--regen|--forensics]" >&2
    exit 2
    ;;
esac
