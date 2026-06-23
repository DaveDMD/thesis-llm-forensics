# RAG — Mistral-7B-Instruct (NF4) · Chroma retriever

Self-contained reproduction of the **RAG campaign**: a retrieval-augmented service where the
attack surface is distinct from the completion world. An instruction/safety-tuned model
(Mistral-7B) answers over a document store (Chroma + all-MiniLM-L6-v2), and the campaign studies
prompt injection, a trigger-gated knowledge-base backdoor, the popularity / parametric-vs-retrieval
effect, and how the detector behaves on a single pool with hard verbose negatives.

`mistralai/Mistral-7B-Instruct-v0.2 (4-bit NF4)` · `ChromaRetriever (cosine)`

## Environments

`E0` = no defences. `E1` = realistic controls: output redaction (shape-based) + rate limiting +
logprob channel closed.

## How to run

From the repository root (everything runs in the project Docker image):

```bash
# No-GPU replay (default): re-score the committed combined pool with frozen v1/v2/v3 + v5.
campaigns/RAG/reproduce.sh --replay

# Full GPU regeneration: rebuild the Chroma corpus and run the whole work-item battery.
campaigns/RAG/reproduce.sh --regen
```

The **replay** path needs only the committed forensic streams + frozen detectors and runs on CPU.
The **regen** path needs the Mistral-7B weights and a GPU; the corpus/index is rebuilt from the
shipped corpus definition (`forensic.corpus`) — the Chroma store and weights are not in git.

## Forensic substrate (produced during every run)

Append-only, hash-chained evidence log (WORM-like) with pseudonymised user ids, UTC timestamps,
raw prompt/response, model version, **retriever hits**, request metadata and latency;
verifier-checkable integrity, timestamp-anchored, resettable.

## Files

| path | role |
|---|---|
| `manifest.json` | model/retriever, detectors, replay streams, expected results, headline metrics |
| `reproduce.sh` | launcher (`--replay` no-GPU / `--regen` GPU) |
| `README.md` | this file |

The full per-work-item write-up (anti-circularity guardrails, literature) is part of the
thesis manuscript and is not shipped in this repository.
