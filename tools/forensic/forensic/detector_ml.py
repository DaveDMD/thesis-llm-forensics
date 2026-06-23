"""Supervised ML detector for the forensic pipeline.

Trains and evaluates a supervised classifier that distinguishes attack from
legitimate requests on the feature rows produced by ``features.build_features``.
It complements — does not replace — the rule-based detector (``detector.py``,
M4): the rule-based detector is the interpretable baseline, this is the learned
comparator, and the evaluation report places them side by side.

Design
------
* Primary target: binary ``label_is_attack``. The multiclass ``label_attack_family``
  is left to a separate diagnostic pass (the X/y builder exposes it).
* Models: logistic regression (primary, interpretable) and random forest
  (comparative). No neural networks.
* Split: GroupKFold over ``session_id`` so no session appears in both train and
  test — attack-session requests are correlated, and per-request splitting would
  leak.
* Scores: the classifier's positive-class probability per request is what the
  ``metrics`` module consumes for TPR@low-FPR; thresholded predictions feed the
  classic metrics and time-to-detection.

The X/y construction (column selection, imputation, encoding) is pure and
testable without scikit-learn; the training/CV functions import sklearn lazily,
so this module imports cleanly in environments without the ML stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .features import feature_columns

# Feature columns whose value can be None (RAG similarity on non-RAG endpoints).
# Imputed to 0.0; the companion flag feature_has_rag_retrieval already encodes
# whether retrieval happened, so 0.0 is unambiguous here.
_SIMILARITY_FEATURES = (
    "feature_top1_similarity_score",
    "feature_max_similarity_score",
    "feature_mean_similarity_score",
)

PRIMARY_LABEL = "label_is_attack"
GROUP_KEY = "session_id"


@dataclass(frozen=True)
class Dataset:
    """Numeric design matrix + labels + groups for grouped cross-validation."""

    X: list[list[float]]
    y: list[int]
    groups: list[str]
    feature_names: list[str]

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _to_float(value: Any) -> float:
    """Encode a feature value to float: bool->0/1, None->0.0, numbers as-is."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def build_xy(
    feature_rows: Sequence[dict[str, Any]],
    *,
    label: str = PRIMARY_LABEL,
    group_key: str = GROUP_KEY,
) -> Dataset:
    """Build the numeric design matrix from feature rows.

    Only ``feature_*`` columns become X (so identifiers and labels can never leak
    into the model). Booleans map to 0/1; missing similarity values impute to
    0.0. ``y`` is the binary label; ``groups`` is the session id used by
    GroupKFold. Feature order is fixed (sorted) for reproducibility.
    """
    if not feature_rows:
        raise ValueError("no feature rows")
    names = sorted(feature_columns(list(feature_rows)))
    if not names:
        raise ValueError("no feature_* columns found")

    X: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    for row in feature_rows:
        X.append([_to_float(row.get(name)) for name in names])
        y.append(1 if row.get(label) else 0)
        groups.append(str(row.get(group_key)))
    return Dataset(X=X, y=y, groups=groups, feature_names=names)


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    test_indices: list[int]
    y_true: list[int]
    y_score: list[float]
    y_pred: list[int]


@dataclass
class CrossValResult:
    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    # Out-of-fold aggregation (each sample predicted exactly once, as test)
    oof_y_true: list[int] = field(default_factory=list)
    oof_y_score: list[float] = field(default_factory=list)
    oof_y_pred: list[int] = field(default_factory=list)
    oof_index: list[int] = field(default_factory=list)


def _make_estimator(model_name: str, *, random_state: int = 42):
    """Instantiate a model. sklearn imported lazily."""
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Scaling matters for logistic regression with mixed-scale features
        # (counts/lengths vs 0/1 flags); a pipeline keeps it leak-free per fold.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model_name: {model_name!r}")


def cross_validate_grouped(
    dataset: Dataset,
    *,
    model_name: str = "logistic",
    n_splits: int = 5,
    threshold: float = 0.5,
    random_state: int = 42,
) -> CrossValResult:
    """GroupKFold CV over sessions, returning out-of-fold scores and predictions.

    Each sample is in the test fold exactly once; the out-of-fold arrays
    (``oof_*``) are the basis for TPR@low-FPR and the classic/operational metrics
    computed by the ``metrics`` module. sklearn and numpy are imported lazily.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    n_groups = len(set(dataset.groups))
    if n_groups < n_splits:
        n_splits = max(2, n_groups)

    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=int)
    groups = np.asarray(dataset.groups)

    result = CrossValResult(model_name=model_name)
    gkf = GroupKFold(n_splits=n_splits)
    oof_score = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        est = _make_estimator(model_name, random_state=random_state)
        est.fit(X[train_idx], y[train_idx])
        proba = est.predict_proba(X[test_idx])[:, 1]
        preds = (proba >= threshold).astype(int)
        oof_score[test_idx] = proba
        result.folds.append(
            FoldResult(
                fold=fold,
                n_train=len(train_idx),
                n_test=len(test_idx),
                test_indices=test_idx.tolist(),
                y_true=y[test_idx].tolist(),
                y_score=proba.tolist(),
                y_pred=preds.tolist(),
            )
        )

    order = [i for i in range(len(y)) if not np.isnan(oof_score[i])]
    result.oof_index = order
    result.oof_y_true = [int(y[i]) for i in order]
    result.oof_y_score = [float(oof_score[i]) for i in order]
    result.oof_y_pred = [1 if oof_score[i] >= threshold else 0 for i in order]
    return result


__all__ = [
    "Dataset",
    "build_xy",
    "FoldResult",
    "CrossValResult",
    "cross_validate_grouped",
    "PRIMARY_LABEL",
    "GROUP_KEY",
]
