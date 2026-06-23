"""Persist a fitted session detector as a FROZEN, reloadable instance.

Train-once / load-many. v1 (Passata-1, frozen) and v2 (specialised) are each fit
ONCE and saved here, then LOADED for every experiment — they are NEVER re-fit, so
each is a single intact instance with a stable identity and explicit provenance
(which residue cells trained it, which secret recogniser it uses). This is what
makes "run the frozen detector on a new campaign" mean loading the same object,
not reconstructing it.

The fit reuses the exact same internals as ``online_detector.fit_session_scorer``
(``build_xy`` + ``_make_estimator``), so a saved instance is identical to the one
that runner would have produced — persistence changes the lifecycle, not the model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

ScoreFn = Callable[[dict[str, Any]], float]


def fit_estimator(
    train_rows: Sequence[dict[str, Any]], *, model_name: str = "logistic", random_state: int = 42
):
    """Fit and return ``(estimator, feature_names)`` (the picklable parts)."""
    import numpy as np

    from .detector_ml import _make_estimator, build_xy

    ds = build_xy(list(train_rows))
    est = _make_estimator(model_name, random_state=random_state)
    est.fit(np.asarray(ds.X, dtype=float), np.asarray(ds.y, dtype=int))
    return est, ds.feature_names


def score_fn_from(estimator, feature_names: Sequence[str]) -> ScoreFn:
    """Rebuild the positive-class probability scorer (same vectorisation as fit)."""
    import numpy as np

    from .detector_ml import _to_float

    names = list(feature_names)

    def score(session_row: dict[str, Any]) -> float:
        x = np.asarray([[_to_float(session_row.get(n)) for n in names]], dtype=float)
        return float(estimator.predict_proba(x)[0, 1])

    return score


def save_scorer(path: str | Path, estimator, feature_names: Sequence[str], *, provenance: dict | None = None) -> Path:
    """Persist the frozen instance + a human-readable provenance sidecar."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimator": estimator, "feature_names": list(feature_names), "provenance": provenance or {}}, p
    )
    p.with_suffix(".json").write_text(
        json.dumps({"feature_names": list(feature_names), "provenance": provenance or {}}, indent=2),
        encoding="utf-8",
    )
    return p


def load_scorer(path: str | Path) -> tuple[ScoreFn, list[str], dict]:
    """Load a frozen instance → ``(score_fn, feature_names, provenance)``."""
    import joblib

    d = joblib.load(Path(path))
    return score_fn_from(d["estimator"], d["feature_names"]), list(d["feature_names"]), d.get("provenance", {})
