# Reproducible campaigns

Each subdirectory is a self-contained reproduction of one experimental campaign: it launches the
attack battery against the target and runs the detector (runtime + post-hoc), producing the attack
and detection results. Every campaign offers:

- **`--replay`** (default, **no GPU**): re-score the committed forensic streams with the detectors.
  Shows attack-vs-detection in seconds, with no model — the recommended way to inspect the results.
- **`--regen`** (**GPU**): regenerate the target and run the full battery from scratch. Heavy
  weights/datasets are not shipped; they are downloaded / re-created deterministically.
- **`--forensics`** (**no GPU**): chain-of-custody verification + timeline structure on the
  campaign's committed stream.

```bash
campaigns/<CAMPAIGN>/reproduce.sh --replay        # no GPU (default)
campaigns/<CAMPAIGN>/reproduce.sh --regen         # GPU
campaigns/<CAMPAIGN>/reproduce.sh --forensics     # no GPU, custody check
```

## Campaigns

**C is the principal campaign** (the pre-training regime); A, B and RAG put the spotlight on
specific aspects of the problem.

| campaign | target | regime | what it exercises |
|---|---|---|---|
| [**C**](C/) — principal | Pythia-2.8b / The Pile (MIMIR) | **pre-training** | membership (MIMIR) vs. resident-secret extraction; online vs. post-hoc detection under E0/E1 |
| [A](A/) | gpt2-medium / WikiText 12-epoch | fine-tune, heavy overfit | logprob-based MIA and secret extraction at the overfit upper bound; the E1 access-surface boundary |
| [B](B/) | gpt2-medium / OpenWebText 1-epoch | fine-tune, realistic | membership signal and secret-extraction efficacy under E0/E1 |
| [RAG](RAG/) | Mistral-7B-Instruct (NF4) / Chroma | retrieval | prompt injection, KB backdoor, retrieval-as-leak; detector on hard verbose negatives |

## The detector, in two acts

Two detectors are shipped (not the intermediate versions):

1. **Frozen, length/volume-based (baseline)** — length/volume features, trained on "volume = attack".
2. **Behavioral / adaptive** — keys on `refusal_rate` / `chaining_rate`, not length; evaluated
   out-of-distribution and by held-out domain/model.

## Shared core and the forensic chain

All campaigns share one core (`tools/forensic/`): the hash-chained forensic logger, the E0/E1
defences, the attack runners, the detectors, and the forensic pipeline. Each campaign directory
holds only its `manifest.json`, `reproduce.sh` and `README.md`.

The **forensic chain** (timeline · attribution · IR playbook · custody) is a cross-cutting,
model-agnostic capability — it reads request metadata and log structure, not model outputs — so it
is demonstrated **once**, on a realistic multi-actor incident, under [`../forensic-investigation/`](../forensic-investigation/).

The detailed per-campaign analysis (A/B/C and RAG), the consolidated metrics and the critical
discussion are part of the thesis manuscript and are not shipped in this repository.
