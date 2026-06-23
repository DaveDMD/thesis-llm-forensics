# TARGET B — gpt2-medium · OpenWebText 1-epoch (realistic)

Self-contained reproduction of campaign **B**: a realistic fine-tuning regime used to study
secret-extraction and membership-inference attacks against a locally hosted LLM, the forensic
artefacts they leave, and an automatic detector that scores them at runtime and post-hoc.

`checkpoint id 607022bd` · registry `results/canary/607022bd-…_registry.json`

## Environments

`E0` = no defences. `E1` = realistic controls: output redaction + rate limiting + logprob
channel closed (`expose_logprobs=False`).

## How to run

Everything runs in the project Docker image (`docker compose`). From the repository root:

```bash
# No-GPU replay (default): re-score the committed forensic streams with the frozen
# detectors (runtime + post-hoc) and read back the attack-efficacy summaries.
campaigns/B/reproduce.sh --replay

# Full GPU regeneration: re-create the target from the registry and run the whole
# attack battery x {E0,E1} plus the runtime detector.
campaigns/B/reproduce.sh --regen
```

The **replay** path needs only the committed evidence streams + frozen detector joblibs and
finishes in seconds on CPU. The **regen** path needs an NVIDIA GPU and downloads the public
gpt2-medium weights and the OpenWebText sample; the fine-tuned checkpoint and working data are
**not** shipped in git (they are regenerated deterministically from the registry).

## Forensic substrate (produced during every run)

The attack battery writes an append-only, **hash-chained** evidence log (WORM-like) with
pseudonymised user ids, UTC timestamps, raw prompt/response, model version, retriever hits,
request metadata and latency. Integrity is checkable with the verifier, and the run is
timestamp-anchored (OpenTimestamps) and resettable.

## Files

| path | role |
|---|---|
| `manifest.json` | machine-readable index: registry, detectors, replay streams, expected results, headline metrics |
| `reproduce.sh` | launcher (`--replay` no-GPU / `--regen` GPU) |
| `README.md` | this file |

Inputs/outputs live in the shared tree (`results/`, `evidence/`, `results/detectors/`); the
exact paths for this campaign are listed in `manifest.json`.
