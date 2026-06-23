#!/usr/bin/env bash
#
# Reproduce the RAG campaign — Mistral-7B-Instruct (4-bit NF4) behind a Chroma retriever.
# The retrieval world: prompt injection (goal hijacking + prompt leaking), a trigger-gated KB
# backdoor, the popularity / parametric-vs-retrieval effect (Mallen), and the detector applied
# to all families on a single pool with hard verbose negatives.
#
# Modes:
#   --replay  (default, NO GPU): re-score the committed combined-pool streams with the frozen
#             detectors and the v5 behavioral detector, and read back the per-attack summaries.
#   --regen   (GPU): rebuild the Chroma corpus, then run the full work-item battery and the
#             combined detection. Needs the Mistral-7B weights and an NVIDIA GPU.
#
set -euo pipefail

MODE="${1:---replay}"
SVC=thesis; [ "$MODE" = "--regen" ] && SVC=thesis-gpu
DC=(docker compose run --rm -e PYTHONPATH=/workspace/tools:/workspace/tools/forensic "$SVC" python3)
RUN="tools/forensic/runners"

case "$MODE" in
  --replay)
    echo "[RAG] REPLAY (no GPU) — re-scoring the combined pool with frozen v1/v2/v3 + v5…"
    "${DC[@]}" "$RUN/run_b7_v5_rag.py"
    echo
    echo "[RAG] DETECTOR metrics:   results/b7_metrics/v5_rag.json"
    echo "[RAG] ATTACK summaries:   results/combined_rag/combined_detection_summary.json"
    echo "                          results/promptinject_rag/promptinject_summary.json"
    echo "                          results/backdoor_rag/backdoor_summary.json"
    echo "                          results/popularity_rag/popularity_summary.json"
    echo "[RAG] Compare against headline numbers in campaigns/RAG/manifest.json."
    ;;

  --regen)
    echo "[RAG] REGEN (GPU) — full retrieval-world pipeline…"
    echo "[RAG] 1/4  build + ingest the Chroma corpus (CPU)"
    "${DC[@]}" "$RUN/build_rag_corpus.py" --ingest \
      --collection internal-handbook-realistic --embedding-model all-MiniLM-L6-v2 --n-distractors 300
    echo "[RAG] 2/4  attack work-items (PromptInject / backdoor / popularity / retrieval-targeting)"
    "${DC[@]}" "$RUN/run_promptinject_rag_real.py"
    "${DC[@]}" "$RUN/run_backdoor_rag_real.py"
    "${DC[@]}" "$RUN/run_popularity_rag_real.py"
    "${DC[@]}" "$RUN/run_retrieval_targeting_rag_real.py"
    echo "[RAG] 3/4  realistic controls E0 vs E1"
    "${DC[@]}" "$RUN/run_rag_e0_e1_real.py"
    echo "[RAG] 4/4  combined detection (hard negatives) + v5 detector"
    "${DC[@]}" "$RUN/run_combined_rag_detection_real.py"
    "${DC[@]}" "$RUN/run_b7_v5_rag.py"
    echo "[RAG] done — see results/{promptinject_rag,backdoor_rag,popularity_rag,combined_rag,b7_metrics}/"
    ;;

  --forensics)
    echo "[RAG] CUSTODY (no GPU) — chain integrity + attribution + timeline structure on the committed stream…"
    "${DC[@]}" "$RUN/run_forensic_report.py" --log evidence/combined_rag/E0_cmb-rag-E0-b3712267.jsonl
    echo "[RAG] full multi-actor forensic chain (phases/attribution/IR playbook): forensic-investigation/reproduce.sh --replay"
    ;;

  *)
    echo "usage: $0 [--replay|--regen|--forensics]" >&2
    exit 2
    ;;
esac
