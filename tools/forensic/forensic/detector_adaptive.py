"""Adaptive online detector (v4) — runtime self-training, reset-per-run. TARGET C.

Prototype: starts from a frozen BASE scorer (v3) and, DURING the campaign, refits an
online logistic on PSEUDO-LABELS derived from high-precision OBSERVABLE anchors —
never the ground-truth label:

  * positive anchor — a secret-format string appeared in a response
    (``feature_session_secret_like_rate`` > 0) OR extreme prompt self-similarity
    (>= ``sim_pos``): near-certain attack signatures, observable at the gateway;
  * negative anchor — the current scorer is confident-benign (score <= ``neg_score``)
    and no positive trigger fires.

The refit uses the FULL current feature set, so the new chaining / novelty features
enter v4 even though the frozen v1/v2/v3 (fixed feature set) cannot use them. The
adaptation is confined to ONE run and discarded by ``reset()`` → no permanent
over-specialisation, the frozen instances are untouched.

Anti-circularity: ``label_is_attack`` is read ONLY for the final evaluation, NEVER
for adaptation; in the training buffer the pseudo-label OVERWRITES it, so the fit
can never see the oracle. The honesty report compares pseudo-labels to ground truth
(evaluation only) to expose anchor precision.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .detector_store import fit_estimator, score_fn_from

ScoreFn = Callable[[dict], float]


class AdaptiveDetector:
    """Predict-then-update online detector over a stream of session rows."""

    def __init__(self, base_score: ScoreFn, *, sim_pos: float = 0.8, neg_score: float = 0.1,
                 min_per_class: int = 3, threshold: float = 0.5) -> None:
        self._base = base_score
        self._sim_pos = sim_pos
        self._neg = neg_score
        self._min = min_per_class
        self._thr = threshold
        self._buffer: list[dict] = []
        self._online: Optional[ScoreFn] = None
        self.n_pos = 0
        self.n_neg = 0
        self.n_refits = 0

    def _score(self, row: dict[str, Any]) -> float:
        return self._online(row) if self._online is not None else self._base(row)

    @staticmethod
    def _positive_trigger(row: dict[str, Any], sim_pos: float) -> bool:
        """High-precision OBSERVABLE attack signatures (no label)."""
        return (float(row.get("feature_session_secret_like_rate") or 0.0) > 0.0
                or float(row.get("feature_session_prompt_self_similarity") or 0.0) >= sim_pos)

    def _pseudo_label(self, row: dict[str, Any], score: float) -> Optional[int]:
        if self._positive_trigger(row, self._sim_pos):
            return 1
        if score <= self._neg:
            return 0
        return None

    def observe_and_score(self, row: dict[str, Any]) -> dict[str, Any]:
        """Score the session with the CURRENT model, then learn from its observable
        anchor (predict-then-update, so detection reflects what v4 knew on arrival)."""
        score = self._score(row)
        pl = self._pseudo_label(row, score)
        if pl is not None:
            buf = {k: v for k, v in row.items() if k != "label_is_attack"}
            buf["label_is_attack"] = pl          # pseudo OVERWRITES oracle; fit never sees truth
            self._buffer.append(buf)
            if pl == 1:
                self.n_pos += 1
            else:
                self.n_neg += 1
            self._maybe_refit()
        return {"score": score, "detected": bool(score >= self._thr), "pseudo_label": pl}

    def _maybe_refit(self) -> None:
        if self.n_pos >= self._min and self.n_neg >= self._min:
            try:
                est, names = fit_estimator(self._buffer)
                self._online = score_fn_from(est, names)
                self.n_refits += 1
            except Exception:
                pass                              # keep the previous model on a degenerate fit

    def reset(self) -> None:
        """Discard ALL adapted state → back to the frozen base (per-run, non-permanent)."""
        self._buffer.clear()
        self._online = None
        self.n_pos = self.n_neg = self.n_refits = 0

    @property
    def adapted(self) -> bool:
        return self._online is not None


def pseudo_label_precision(pseudo: list[Optional[int]], truth: list[int]) -> dict[str, float]:
    """EVALUATION-ONLY: precision of the positive pseudo-anchors against ground truth
    (anti-circular honesty check — never fed back into the detector)."""
    pos = [(p, t) for p, t in zip(pseudo, truth) if p == 1]
    neg = [(p, t) for p, t in zip(pseudo, truth) if p == 0]
    pos_prec = sum(1 for _p, t in pos if t == 1) / len(pos) if pos else float("nan")
    neg_prec = sum(1 for _p, t in neg if t == 0) / len(neg) if neg else float("nan")
    return {"n_pos_anchors": float(len(pos)), "pos_anchor_precision": round(pos_prec, 3),
            "n_neg_anchors": float(len(neg)), "neg_anchor_precision": round(neg_prec, 3)}


__all__ = ["AdaptiveDetector", "pseudo_label_precision"]
