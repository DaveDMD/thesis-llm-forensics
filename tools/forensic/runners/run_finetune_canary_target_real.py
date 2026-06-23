#!/usr/bin/env python3
"""Fine-tune a small Pythia on the controlled canary corpus.

Builds the synthetic corpus + canaries (`canary_dataset`), fine-tunes a small Pythia
(default 410m) on the MEMBER documents (causal-LM), and saves a **versioned
checkpoint + manifest + canary registry** so the attacks attack a target
with **perfect ground truth**. Memory-frugal on 8 GB (gradient checkpointing + bf16
autocast). Deterministic (seed). Run inside docker (GPU)::

    docker compose run --rm thesis \\
        python3 tools/forensic/runners/run_finetune_canary_target_real.py \\
        --base-model EleutherAI/pythia-410m --epochs 3
"""
from __future__ import annotations

import os

import sys

os.environ.setdefault("HF_HUB_CACHE", "/workspace/data/mimir-cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/data/hf-datasets")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# HF_HUB_OFFLINE is the global hammer: it makes `datasets`' module factory RAISE a
# ConnectionError instead of falling back to the cached build. The model stays
# offline via TRANSFORMERS_OFFLINE, so set HF_HUB_OFFLINE only when NOT loading a
# Hub dataset (i.e. keep the synthetic/Pythia path byte-identical).
_argv = sys.argv
_rich = ("--dataset" in _argv and _argv.index("--dataset") + 1 < len(_argv)
         and _argv[_argv.index("--dataset") + 1] == "rich")
if not _rich:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import uuid
from pathlib import Path


def _load_wikitext_paragraphs(*, n_needed: int, split: str = "train",
                              min_chars: int = 180) -> list[str]:
    """Deterministically take the first ``n_needed`` real paragraphs from
    WikiText-2-raw (drop blanks / section headers / too-short lines). Offline from
    the HF cache; used as the natural-text background of the rich corpus."""
    from datasets import load_dataset
    rows = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        t = " ".join(r["text"].split())
        if len(t) < min_chars or t.startswith("=") or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= n_needed:
            break
    if len(out) < n_needed:
        raise RuntimeError(f"WikiText yielded only {len(out)} paragraphs, need {n_needed}")
    return out


def _load_openwebtext_paragraphs(*, n_needed: int, dataset_name: str = "stas/openwebtext-10k",
                                 split: str = "train", revision: str = "refs/convert/parquet",
                                 min_chars: int = 180) -> list[str]:
    """Deterministically harvest the first ``n_needed`` natural paragraphs from an
    OpenWebText replica — the open reproduction of GPT-2's own WebText training set,
    so the TARGET B background is IN-DISTRIBUTION for gpt2 (no train/finetune
    domain shift -> the MIA discriminates "seen in finetune" vs "not", not domain).
    Documents are split on newlines into paragraphs; blanks / too-short / duplicate
    lines are dropped. Loaded from the auto-converted PARQUET branch (no remote
    code). Offline from the HF datasets cache (pre-downloaded)."""
    from datasets import load_dataset
    rows = load_dataset(dataset_name, revision=revision, split=split)
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for raw in r["text"].split("\n"):
            t = " ".join(raw.split())
            if len(t) < min_chars or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= n_needed:
                return out
    raise RuntimeError(f"OpenWebText yielded only {len(out)} paragraphs, need {n_needed}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="EleutherAI/pythia-410m")
    ap.add_argument("--revision", default="step99000")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260612)
    # dataset params (defaults reproduce the working run: clear repetition->memorisation
    # curve N=1/4/16/64 -> 1/4/8/9 of 12 on the English corpus)
    ap.add_argument("--n-generic-members", type=int, default=150)
    ap.add_argument("--n-nonmembers", type=int, default=150)
    ap.add_argument("--canaries-per-cell", type=int, default=3)
    # rich = WikiText-backed "Pile-like" corpus + 6 realistic secret framings
    # (default synthetic = unchanged Pythia baseline behaviour)
    ap.add_argument("--dataset", choices=["synthetic", "rich"], default="synthetic")
    ap.add_argument("--context-chars", type=int, default=220)
    ap.add_argument("--wikitext-split", default="train")
    # TARGET B: natural-text background source + repetition / context knobs that
    # turn the overfit target into a realistic-MIA one (large in-distribution
    # background, 1 epoch, members seen once; canaries kept high-rep for extraction)
    ap.add_argument("--background", choices=["wikitext", "openwebtext"], default="wikitext",
                    help="rich background source (openwebtext = GPT-2's own distribution)")
    ap.add_argument("--openwebtext-name", default="stas/openwebtext-10k")
    ap.add_argument("--openwebtext-split", default="train")
    ap.add_argument("--repetitions", default="",
                    help="comma list of canary repetitions (default: dataset DEFAULT_REPETITIONS)")
    ap.add_argument("--disjoint-context", action="store_true",
                    help="rich only: never reuse generic MIA members as canary context")
    ap.add_argument("--repo-root", default="/workspace")
    args = ap.parse_args()

    from forensic.canary_dataset import (
        DEFAULT_REPETITIONS, RICH_KINDS, build_canary_dataset, build_rich_canary_dataset)
    from forensic.finetune import chunk_token_ids, corpus_fingerprint

    reps = (tuple(int(x) for x in args.repetitions.split(",") if x.strip())
            if args.repetitions.strip() else DEFAULT_REPETITIONS)

    # ── build the controlled dataset FIRST ───────────────────────────────────
    # For the rich path this loads the natural background via `datasets`, which MUST
    # happen before importing transformers: transformers with TRANSFORMERS_OFFLINE=1
    # flips huggingface_hub into offline mode, after which datasets' module factory
    # raises a ConnectionError instead of falling back to the cached build.
    if args.dataset == "rich":
        # openwebtext (TARGET B): large in-distribution pool -> a wide context pool
        # so member paragraphs are not incidentally reused. wikitext keeps the exact
        # TARGET A budget (+200) for byte-identical reproducibility.
        ctx_buffer = 6000 if args.background == "openwebtext" else 200
        n_needed = args.n_generic_members + args.n_nonmembers + ctx_buffer
        if args.background == "openwebtext":
            paras = _load_openwebtext_paragraphs(
                n_needed=n_needed, dataset_name=args.openwebtext_name, split=args.openwebtext_split)
            print(f"[i] OpenWebText background ({args.openwebtext_name}): {len(paras)} paragraphs loaded")
        else:
            paras = _load_wikitext_paragraphs(n_needed=n_needed, split=args.wikitext_split)
            print(f"[i] WikiText background: {len(paras)} paragraphs loaded")
        ds = build_rich_canary_dataset(
            background_paragraphs=paras,
            n_generic_members=args.n_generic_members, n_nonmembers=args.n_nonmembers,
            repetitions=reps, kinds=RICH_KINDS,
            n_canaries_per_cell=args.canaries_per_cell, seed=args.seed,
            context_chars=args.context_chars, disjoint_context=args.disjoint_context,
        )
    else:
        ds = build_canary_dataset(
            n_generic_members=args.n_generic_members, n_nonmembers=args.n_nonmembers,
            repetitions=reps, n_canaries_per_cell=args.canaries_per_cell, seed=args.seed,
        )
    member_texts = ds.finetune_texts()
    print(f"[i] dataset({args.dataset}/{args.background if args.dataset=='rich' else '-'}): "
          f"{len(ds.member_documents)} member / {len(ds.nonmember_documents)} non-member docs; "
          f"{len(ds.canaries)} canaries (rep {reps}, disjoint_ctx={args.disjoint_context})")

    # transformers imported AFTER the dataset (see offline note above)
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    run_id = str(uuid.uuid4())

    # ── load base model + tokenizer ──────────────────────────────────────────
    print(f"[i] loading base {args.base_model}@{args.revision}…")
    # the step-revision snapshot holds only weights; the tokenizer (identical across
    # all Pythia checkpoints) lives in the `main` snapshot — fall back to it offline.
    try:
        tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model, revision="main")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.revision).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ── tokenize member corpus -> flat stream -> fixed blocks ────────────────
    eos = tok.eos_token_id
    ids: list[int] = []
    for t in member_texts:
        ids.extend(tok(t)["input_ids"] + [eos])
    blocks = chunk_token_ids(ids, args.block_size)
    print(f"[i] {len(ids)} tokens -> {len(blocks)} blocks of {args.block_size}")
    data = torch.tensor(blocks, dtype=torch.long)

    # ── training loop (AdamW, bf16 autocast, grad accumulation) ──────────────
    opt = AdamW(model.parameters(), lr=args.lr)
    model.train()
    n = data.size(0)
    step = 0
    last_loss = float("nan")
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch))
        opt.zero_grad()
        for bi in range(0, n, args.batch_size):
            batch = data[perm[bi:bi + args.batch_size]].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=batch, labels=batch)
                loss = out.loss / args.grad_accum
            loss.backward()
            if ((bi // args.batch_size) + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            last_loss = float(out.loss.detach())
            step += 1
            if step % 100 == 0:
                print(f"      epoch {epoch} step {step} loss {last_loss:.3f}")
        print(f"[i] epoch {epoch} done; loss {last_loss:.3f}")

    # ── save checkpoint + tokenizer ──────────────────────────────────────────
    out_dir = Path(args.repo_root) / "models" / "finetuned" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # ── canary memorization sanity check (greedy from prefix) ────────────────
    model.eval()
    by_rep: dict[int, list[int]] = {}
    for c in ds.canaries:
        enc = tok(c.host_prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        cont = tok.decode(gen[0][enc["input_ids"].size(1):], skip_special_tokens=True)
        hit = 1 if c.value in cont else 0
        by_rep.setdefault(c.repetition, []).append(hit)
    mem_by_rep = {rep: f"{sum(h)}/{len(h)}" for rep, h in sorted(by_rep.items())}

    # ── manifest + registry (for reproducible attacks) ────────────
    res_dir = Path(args.repo_root) / "results" / "canary"
    res_dir.mkdir(parents=True, exist_ok=True)
    mem_texts, non_texts = ds.mia_pairs()
    manifest = {
        "run_id": run_id, "base_model": args.base_model, "revision": args.revision,
        "checkpoint": str(out_dir), "seed": args.seed, "dataset_kind": args.dataset,
        "hyperparams": {"epochs": args.epochs, "block_size": args.block_size,
                        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr},
        "dataset_params": {"n_generic_members": args.n_generic_members,
                           "n_nonmembers": args.n_nonmembers, "repetitions": list(reps),
                           "canaries_per_cell": args.canaries_per_cell, "seed": args.seed,
                           "context_chars": args.context_chars, "wikitext_split": args.wikitext_split,
                           "background": args.background, "disjoint_context": args.disjoint_context,
                           "openwebtext_name": args.openwebtext_name},
        "corpus_fingerprint": corpus_fingerprint(member_texts),
        "n_member_docs": len(ds.member_documents), "n_nonmember_docs": len(ds.nonmember_documents),
        "n_canaries": len(ds.canaries), "final_loss": last_loss,
        "canary_memorization_by_repetition": mem_by_rep,
        "canaries": ds.extraction_probes(),
    }
    # rich corpus is not regenerable from params alone (WikiText-derived) → make the
    # registry SELF-CONTAINED so the attack runners need not reload/rebuild it
    if args.dataset == "rich":
        manifest["mia_members"] = mem_texts
        manifest["mia_nonmembers"] = non_texts
    (res_dir / f"{run_id}_registry.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _bulky = {"canaries", "mia_members", "mia_nonmembers"}
    (out_dir / "training_manifest.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k not in _bulky}, indent=2), encoding="utf-8")

    print(f"\n[✓] fine-tuned checkpoint: {out_dir}")
    print(f"[✓] final loss: {last_loss:.3f}")
    print(f"[✓] canary memorization by repetition (greedy from prefix): {mem_by_rep}")
    print(f"[✓] registry: {res_dir / (run_id + '_registry.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
