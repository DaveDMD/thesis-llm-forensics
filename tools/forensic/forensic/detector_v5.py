"""Adaptive online detector v5 — behavioural-specialised base (Motore 1) + a
configurable DEFENSE-ACTION pseudo-label anchor (Motore 2). RAG world.

v5 is the constructive counterpart to the RAG finding (frozen completion-world
detectors fail OOD on the RAG attacks, AUC 0.38). It is built to test, *honestly*,
three things on the RAG hard-negative pool:

* **Motore 1 — specialised base.** A base scorer trained IN-DISTRIBUTION on the
  BEHAVIOURAL features (refusal / prompt self-similarity / chaining / novelty, NOT
  length) closes the closed-world gap the frozen detectors leave open. This is the
  knowledge the previous detectors lacked, so it is expected to help E0 AND E1.

* **Motore 2 — defense loop.** Using the DEFENSE ACTION (an output-filter
  redaction — observable at the gateway, label-free) as the positive pseudo-label
  anchor, so the online learner can in principle pick up a family the base never
  saw. This is BOUNDED by the defense's own precision: a content filter that also
  redacts verbose-benign traffic gives noisy supervision. The honesty check
  ``pseudo_label_precision`` exposes exactly how noisy.

* **closed-world limit.** Leave-one-family-out (driven by the runner) shows the
  base fails on a truly novel family, and the noisy defense loop does not rescue it.

Anti-circularity (the two-tier discipline is unchanged):

* the design matrix is ``feature_*`` only (``detector_ml.build_xy``); the defense
  signal is carried on the NON-feature field ``session_defense_action_rate`` and is
  therefore a pseudo-label SOURCE, never a model feature;
* the pseudo-label OVERWRITES the oracle in the training buffer (the fit never sees
  ``label_is_attack``); the oracle is read for evaluation only;
* predict-then-update (the score reflects what v5 knew on arrival); reset-per-run,
  so the adaptation is confined to one campaign and discarded afterwards.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .b7_operating_point import partition_length_behavioral
from .detector_adaptive import pseudo_label_precision  # re-exported honesty check
from .detector_store import fit_estimator, score_fn_from

ScoreFn = Callable[[dict], float]

# Non-feature row field carrying the observable defense action (output-filter
# redaction rate over the session). NOT prefixed ``feature_`` on purpose, so it
# never enters ``build_xy`` — it can only seed a pseudo-label.
DEFENSE_FIELD = "session_defense_action_rate"

_SECRET_FEATURE = "feature_session_secret_like_rate"
_SELFSIM_FEATURE = "feature_session_prompt_self_similarity"


def behavioral_feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Behavioural ``feature_*`` names present in ``rows`` (length/volume excluded)."""
    names = sorted({k for r in rows for k in r if k.startswith("feature_")})
    _, behavioral = partition_length_behavioral(names)
    return behavioral


def _restrict_rows(rows: Sequence[dict[str, Any]], keep: set[str]) -> list[dict[str, Any]]:
    """Drop every ``feature_*`` column not in ``keep`` (non-feature fields kept)."""
    return [{k: v for k, v in r.items() if not k.startswith("feature_") or k in keep}
            for r in rows]


def fit_behavioral_base(
    train_rows: Sequence[dict[str, Any]], *, model_name: str = "logistic", random_state: int = 42
) -> ScoreFn:
    """Fit the Motore-1 base: a logistic over the BEHAVIOURAL features only.

    Restricting to behavioural features keeps the base off the length artefact that
    inverts under hard negatives. Returns a callable ``row -> P(attack)``.
    """
    keep = set(behavioral_feature_names(train_rows))
    est, names = fit_estimator(_restrict_rows(train_rows, keep),
                               model_name=model_name, random_state=random_state)
    return score_fn_from(est, names)


class AdaptiveDetectorV5:
    """Predict-then-update online detector (v5) over a stream of session rows.

    Parameters
    ----------
    base_score : ScoreFn
        Frozen base used until the first online refit (e.g. v3, or a behavioural
        base from :func:`fit_behavioral_base`).
    anchor : {"defense", "defense_gated", "behavioral"}
        Source of the positive pseudo-label:
          * ``defense`` — the defense acted on the session (Motore 2, naive);
          * ``defense_gated`` — defense action AND a behavioural attack signature
            (higher precision, but only fires where the base already suspects);
          * ``behavioral`` — secret-like / extreme self-similarity (v4-style).
    restrict_features : sequence of str, optional
        If given, the online refit is restricted to these feature columns (use the
        behavioural set so the online model stays off the length artefact).
    """

    ANCHORS = ("defense", "defense_gated", "behavioral")

    def __init__(
        self,
        base_score: ScoreFn,
        *,
        anchor: str = "defense",
        sim_pos: float = 0.8,
        neg_score: float = 0.1,
        min_per_class: int = 3,
        threshold: float = 0.5,
        restrict_features: Optional[Sequence[str]] = None,
    ) -> None:
        if anchor not in self.ANCHORS:
            raise ValueError(f"anchor must be one of {self.ANCHORS}, got {anchor!r}")
        self._base = base_score
        self._anchor = anchor
        self._sim_pos = sim_pos
        self._neg = neg_score
        self._min = min_per_class
        self._thr = threshold
        self._restrict = set(restrict_features) if restrict_features is not None else None
        self._buffer: list[dict] = []
        self._online: Optional[ScoreFn] = None
        self.n_pos = 0
        self.n_neg = 0
        self.n_refits = 0

    def _score(self, row: dict[str, Any]) -> float:
        return self._online(row) if self._online is not None else self._base(row)

    def _defense_action(self, row: dict[str, Any]) -> bool:
        """Observable defense action on this session (label-free)."""
        return float(row.get(DEFENSE_FIELD) or 0.0) > 0.0

    def _behavioral_trigger(self, row: dict[str, Any]) -> bool:
        return (float(row.get(_SECRET_FEATURE) or 0.0) > 0.0
                or float(row.get(_SELFSIM_FEATURE) or 0.0) >= self._sim_pos)

    def _positive_trigger(self, row: dict[str, Any]) -> bool:
        if self._anchor == "defense":
            return self._defense_action(row)
        if self._anchor == "defense_gated":
            return self._defense_action(row) and self._behavioral_trigger(row)
        return self._behavioral_trigger(row)

    def _pseudo_label(self, row: dict[str, Any], score: float) -> Optional[int]:
        if self._positive_trigger(row):
            return 1
        if score <= self._neg:
            return 0
        return None

    def observe_and_score(self, row: dict[str, Any]) -> dict[str, Any]:
        """Score with the CURRENT model, then learn from the observable anchor."""
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
                buf = self._buffer
                if self._restrict is not None:
                    buf = _restrict_rows(buf, self._restrict)
                est, names = fit_estimator(buf)
                self._online = score_fn_from(est, names)
                self.n_refits += 1
            except Exception:
                pass                              # keep the previous model on a degenerate fit

    def reset(self) -> None:
        """Discard ALL adapted state → back to the frozen base (per-run)."""
        self._buffer.clear()
        self._online = None
        self.n_pos = self.n_neg = self.n_refits = 0

    @property
    def adapted(self) -> bool:
        return self._online is not None


__all__ = [
    "AdaptiveDetectorV5",
    "fit_behavioral_base",
    "behavioral_feature_names",
    "pseudo_label_precision",
    "DEFENSE_FIELD",
]
