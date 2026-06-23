# TARGET C — Pythia-2.8b · The Pile / MIMIR (pre-training regime)

Self-contained reproduction of campaign **C**: the pre-training regime, the third point on the
overfit (A) / realistic fine-tune (B) / pre-training (C) spectrum. The model is **pre-trained on
The Pile, with no fine-tune**; membership is evaluated with the MIMIR set and extraction targets
secrets that are resident in the Pile.

`EleutherAI/pythia-2.8b@step99000` · reference `pythia-160m` · MIMIR github split `ngram_13_0.8`

## Environments

`E0` = no defences. `E1` = realistic controls: output redaction + rate limiting + logprob
channel closed.

## How to run

From the repository root (everything runs in the project Docker image):

```bash
# No-GPU replay (default): re-score the committed targetC streams with the frozen detectors.
campaigns/C/reproduce.sh --replay

# Full GPU regeneration: re-run the battery x {E0,E1} against Pythia-2.8b (no fine-tune).
campaigns/C/reproduce.sh --regen
```

There is no fine-tune: `--regen` uses the public pre-trained Pythia-2.8b checkpoint and reads the
MIMIR data from the HuggingFace cache. The frozen detectors `v1/v2/v3` are shipped as joblibs;
the adaptive `v4` is built at runtime from `v3`.

## Forensic substrate (produced during every run)

Append-only, hash-chained evidence log (WORM-like) with pseudonymised user ids, UTC timestamps,
raw prompt/response, model version, request metadata and latency; verifier-checkable integrity,
timestamp-anchored, resettable.

## Files

| path | role |
|---|---|
| `manifest.json` | model/dataset, detectors, replay streams, expected results, headline metrics |
| `reproduce.sh` | launcher (`--replay` no-GPU / `--regen` GPU) |
| `README.md` | this file |
