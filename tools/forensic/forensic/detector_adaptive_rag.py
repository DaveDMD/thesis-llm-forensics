"""Adaptive online detector for the RAG world (v4-RAG).

A SEPARATE instance from the completion-world ``detector_adaptive.AdaptiveDetector``
(v4). Same anti-circular skeleton (predict-then-update,
reset-per-run, pseudo-labels OVERWRITE the oracle so the fit never sees the truth,
``pseudo_label_precision`` as an evaluation-only honesty check), but with anchors
re-designed to be observable AND label-free in the RAG/backdoor setting:

* positive anchors (high-precision, no label):
  - a secret-format string appears in a response (``feature_session_secret_like_rate``
    > 0). NOTE: the planted secrets are deliberately NOT secret-shaped (by
    corpus design), so this anchor will NOT fire on a backdoor disclosure — it is kept
    only as a generic signal; the primary anchor is the retrieval anomaly below;
  - RETRIEVAL ANOMALY — the session retrieved, AT RANK 1, a chunk that is globally
    RARE so far (running frequency <= ``rare_max_freq``), after a ``warmup``. This is
    the natural forensic signature of KB poisoning ("a strongly-matched but globally
    unusual document entered the context"). It is a label-free STATISTIC over the
    observed ``chunk_hash`` residual — NEVER the poisoned-document identity.
    A known confound (declared): a corpus padded with many rare distractors makes
    rarity noisy; requiring the rare chunk at RANK 1 (strong match) mitigates it.

* negative anchor — the current scorer is confident-benign (score <= ``neg_score``)
  and no positive trigger fires.

The rank-1 chunk hashes are read from a NON-feature row field
``session_rank1_chunk_hashes`` (attached by the runner from the retrieval residual),
so they pseudo-label sessions but never enter the ML design matrix (``build_xy`` keeps
only ``feature_*`` columns).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Optional

from .detector_adaptive import pseudo_label_precision  # re-exported honesty check
from .detector_store import fit_estimator, score_fn_from

ScoreFn = Callable[[dict], float]


class AdaptiveDetectorRAG:
    """Predict-then-update online detector (v4-RAG) over a stream of session rows."""

    def __init__(
        self,
        base_score: ScoreFn,
        *,
        rare_max_freq: int = 1,
        warmup: int = 8,
        neg_score: float = 0.1,
        min_per_class: int = 3,
        threshold: float = 0.5,
    ) -> None:
        self._base = base_score
        self._rare_max = rare_max_freq
        self._warmup = warmup
        self._neg = neg_score
        self._min = min_per_class
        self._thr = threshold
        self._buffer: list[dict] = []
        self._online: Optional[ScoreFn] = None
        self._rank1_freq: Counter = Counter()  # running freq of rank-1 chunk hashes (past sessions)
        self._seen = 0
        self.n_pos = 0
        self.n_neg = 0
        self.n_refits = 0

    def _score(self, row: dict[str, Any]) -> float:
        return self._online(row) if self._online is not None else self._base(row)

    def _retrieval_anomaly(self, row: dict[str, Any]) -> bool:
        """A globally-rare chunk retrieved at rank 1 (label-free novelty signal)."""
        if self._seen < self._warmup:
            return False
        for h in row.get("session_rank1_chunk_hashes") or []:
            if self._rank1_freq.get(h, 0) <= self._rare_max:
                return True
        return False

    def _positive_trigger(self, row: dict[str, Any]) -> bool:
        if float(row.get("feature_session_secret_like_rate") or 0.0) > 0.0:
            return True
        return self._retrieval_anomaly(row)

    def _pseudo_label(self, row: dict[str, Any], score: float) -> Optional[int]:
        if self._positive_trigger(row):
            return 1
        if score <= self._neg:
            return 0
        return None

    def observe_and_score(self, row: dict[str, Any]) -> dict[str, Any]:
        """Score with the CURRENT model, then learn from the observable anchor and
        update the rarity table (predict-then-update → no intra-session leak)."""
        score = self._score(row)
        pl = self._pseudo_label(row, score)
        if pl is not None:
            buf = {k: v for k, v in row.items() if k != "label_is_attack"}
            buf["label_is_attack"] = pl  # pseudo OVERWRITES oracle; fit never sees truth
            self._buffer.append(buf)
            if pl == 1:
                self.n_pos += 1
            else:
                self.n_neg += 1
            self._maybe_refit()
        # update AFTER scoring: rarity is always judged against PAST sessions
        for h in row.get("session_rank1_chunk_hashes") or []:
            self._rank1_freq[h] += 1
        self._seen += 1
        return {"score": score, "detected": bool(score >= self._thr), "pseudo_label": pl}

    def _maybe_refit(self) -> None:
        if self.n_pos >= self._min and self.n_neg >= self._min:
            try:
                est, names = fit_estimator(self._buffer)
                self._online = score_fn_from(est, names)
                self.n_refits += 1
            except Exception:
                pass  # keep the previous model on a degenerate fit

    def reset(self) -> None:
        """Discard ALL adapted state → back to the frozen base (per-run)."""
        self._buffer.clear()
        self._online = None
        self._rank1_freq.clear()
        self._seen = 0
        self.n_pos = self.n_neg = self.n_refits = 0

    @property
    def adapted(self) -> bool:
        return self._online is not None


__all__ = ["AdaptiveDetectorRAG", "pseudo_label_precision"]
