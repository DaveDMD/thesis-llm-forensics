"""Pure helpers for the controlled fine-tune.

The fine-tune itself is a GPU operation (see runners/run_finetune_canary_target_real.py);
this module holds the small, deterministic, testable pieces: chunking a flat token
stream into fixed blocks for causal-LM training, and a corpus fingerprint for the
reproducibility manifest.
"""
from __future__ import annotations

import hashlib


def chunk_token_ids(ids: list[int], block_size: int, *, drop_last: bool = True) -> list[list[int]]:
    """Split a flat token-id stream into fixed-size blocks (causal-LM training).

    With ``drop_last`` (default) a trailing partial block is discarded so every
    block has exactly ``block_size`` tokens.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = [ids[i:i + block_size] for i in range(0, len(ids), block_size)]
    if drop_last and blocks and len(blocks[-1]) < block_size:
        blocks = blocks[:-1]
    return blocks


def corpus_fingerprint(texts: list[str]) -> str:
    """Stable sha256 over the corpus (order-sensitive) for the fine-tune manifest."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


__all__ = ["chunk_token_ids", "corpus_fingerprint"]
