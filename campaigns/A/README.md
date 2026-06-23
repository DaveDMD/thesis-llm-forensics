# TARGET A — gpt2-medium · WikiText 12-epoch (heavy overfit)

Self-contained reproduction of campaign **A**: the upper-bound memorization case used to
study secret-extraction and membership-inference attacks, the forensic artefacts they leave,
and an automatic detector that scores them at runtime and post-hoc.

`checkpoint id c630d00e` · registry `results/canary/c630d00e-…_registry.json`

## Environments

`E0` = no defences. `E1` = realistic controls: output redaction + rate limiting + logprob
channel closed.

## How to run

From the repository root (everything runs in the project Docker image):

```bash
# No-GPU replay (default): re-score the recovered forensic streams with the frozen detectors.
campaigns/A/reproduce.sh --replay

# Full GPU regeneration: re-create the overfit target and run the whole battery x {E0,E1}.
campaigns/A/reproduce.sh --regen
```

A's runtime streams share `evidence/runtime_gpt2/` with TARGET B; the replay engine selects
A's by matching each record's `model_id` to registry `c630d00e`, so there is no ambiguity.
The fine-tuned checkpoint and working data are not shipped — `--regen` re-creates them
deterministically from the registry.

## Forensic substrate (produced during every run)

The attack battery writes an append-only, hash-chained evidence log (WORM-like) with
pseudonymised user ids, UTC timestamps, raw prompt/response, model version, request metadata
and latency; integrity is verifier-checkable and the run is timestamp-anchored and resettable.

## Files

| path | role |
|---|---|
| `manifest.json` | registry, detectors, replay streams, expected results, headline metrics |
| `reproduce.sh` | launcher (`--replay` no-GPU / `--regen` GPU) |
| `README.md` | this file |
