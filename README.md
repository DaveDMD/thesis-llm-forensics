# Forensic instrumentation of a local LLM / RAG service

Experimental apparatus for the forensic analysis of memorization attacks against a locally
hosted LLM/RAG service. Membership-inference and secret-extraction attacks are treated as
generators of observable forensic artefacts: the service produces an append-only, hash-chained
evidence log, an automatic detector scores attack-vs-legitimate sessions at runtime and
post-hoc, and a forensic chain reconstructs the incident (timeline, attribution, IR playbook,
chain of custody).

Everything runs locally inside a single Docker image.

## Quick start (no GPU)

Each campaign re-plays its committed forensic streams through the frozen detectors in seconds,
without any model:

```bash
campaigns/C/reproduce.sh --replay        # principal campaign (Pythia, pre-training)
campaigns/A/reproduce.sh --replay        # gpt2 overfit
campaigns/B/reproduce.sh --replay        # gpt2 realistic
campaigns/RAG/reproduce.sh --replay      # Mistral + retrieval

forensic-investigation/reproduce.sh --replay   # timeline / attribution / IR playbook / custody
```

Add `--regen` to any campaign to regenerate the target and run the full attack battery on a GPU,
or `--forensics` for a no-GPU chain-of-custody check. See [`campaigns/README.md`](campaigns/README.md).

## Structure

```
campaigns/              Four reproducible attack campaigns (target + detector); C is principal
  C/  A/  B/  RAG/       each: manifest.json + reproduce.sh (--replay/--regen/--forensics) + README
forensic-investigation/ Cross-cutting forensic chain demo (model-agnostic, multi-actor incident)
tools/forensic/
  forensic/             Core: hash-chained logger, verifier, FastAPI server, E0/E1 defences,
                        attack engines, detectors, timeline/attribution/playbook
  runners/              Operator runners invoked by the campaigns
docker/                 Dockerfile + pinned requirements
evidence/  results/     Committed forensic streams + detector/attack results (replay inputs)
```

## What it contains

- **Attacks ×E0/E1**: membership inference (white/black-box), secret extraction (prefix,
  sampling, adaptive) for the canary/pre-training campaigns; prompt injection, KB backdoor,
  retrieval-as-leak for RAG. E1 = output redaction + rate limiting + closed logprob channel.
- **Detector, in two acts**: a frozen length/volume-based baseline and a behavioral/adaptive
  detector that keys on `refusal_rate` / `chaining_rate`; generalisation is evaluated by
  held-out domain/model.
- **Forensic chain**: timeline reconstruction, prudential attribution, IR playbook
  (triage/snapshot/preservation/escalation incl. GDPR), and independent chain-of-custody
  verification.

## Prerequisites

- Docker Engine (Linux / WSL2). An NVIDIA GPU + Container Toolkit is required only for the
  `--regen` paths (real model runs); the `--replay` and `--forensics` paths run on CPU.
- For `--regen`, copy `.env.example` to `.env` and set `HUGGINGFACE_TOKEN` to download the
  public model/dataset weights. Weights and working data are not shipped; they are regenerated
  deterministically from the registries.
