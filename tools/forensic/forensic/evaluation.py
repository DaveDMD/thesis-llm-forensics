"""Detector evaluation: ML models on one dataset.

Testable library (returns the report in memory). It evaluates, on the *same*
feature rows and the *same* metric set, the ML detectors:

* ``logistic``      — ML primary (GroupKFold out-of-fold scores);
* ``random_forest`` — ML comparative (GroupKFold out-of-fold scores).

For every detector the same metrics are computed (so the report is directly
comparable): TPR@low-FPR (1% / 5%), classic metrics (precision/recall/F1/FPR/FNR
/accuracy + confusion), and the operational time-to-detection. detection
efficiency is measured per detector from its wall-clock over the evaluated rows.

The rule-based keyword detector is intentionally NOT used: its rules would match
the phrases planted in the strawman attacks (circularity). The supervised ML
detector (``detector_ml``) is therefore the sole detector. No keyword baseline is
reintroduced here; honest features (and any honest baseline) are derived from the
observed residuals.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

from .detector_ml import build_xy, cross_validate_grouped
from .metrics import (
    classic_metrics,
    detection_efficiency,
    time_to_detection,
    tpr_at_fpr_grid,
)

ML_MODELS = ("logistic", "random_forest")


def _session_seq_arrays(feature_rows: Sequence[dict[str, Any]]):
    sids = [str(r.get("session_id")) for r in feature_rows]
    seqs = [int(r.get("sequence_number", 0)) for r in feature_rows]
    return sids, seqs


def _metrics_block(
    *,
    y_true: list[int],
    y_score: list[float],
    y_pred: list[int],
    session_ids: list[str],
    sequence_numbers: list[int],
    target_fprs: Sequence[float],
    n_requests: int,
    processing_seconds: float,
) -> dict[str, Any]:
    n_detections = int(sum(y_pred))
    return {
        "tpr_at_fpr": tpr_at_fpr_grid(y_true, y_score, target_fprs=target_fprs),
        "classic": classic_metrics(y_true, y_pred),
        "time_to_detection": {
            k: v
            for k, v in time_to_detection(
                session_ids, sequence_numbers, y_pred, y_true
            ).items()
            if k != "per_session"  # keep the summary compact; per-session on request
        },
        "efficiency": detection_efficiency(
            n_requests=n_requests,
            n_detections=n_detections,
            processing_seconds=processing_seconds,
        ),
    }


def evaluate_ml_model(
    feature_rows: Sequence[dict[str, Any]],
    *,
    model_name: str,
    n_splits: int = 5,
    threshold: float = 0.5,
    target_fprs: Sequence[float] = (0.01, 0.05),
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate one ML model with GroupKFold and return its metric block."""
    ds = build_xy(list(feature_rows))
    start = time.perf_counter()
    cv = cross_validate_grouped(
        ds, model_name=model_name, n_splits=n_splits, threshold=threshold,
        random_state=random_state,
    )
    processing_seconds = time.perf_counter() - start

    # Align session/seq to the out-of-fold order.
    sids_all, seqs_all = _session_seq_arrays(feature_rows)
    sids = [sids_all[i] for i in cv.oof_index]
    seqs = [seqs_all[i] for i in cv.oof_index]

    block = _metrics_block(
        y_true=cv.oof_y_true,
        y_score=cv.oof_y_score,
        y_pred=cv.oof_y_pred,
        session_ids=sids,
        sequence_numbers=seqs,
        target_fprs=target_fprs,
        n_requests=len(cv.oof_index),
        processing_seconds=processing_seconds,
    )
    block["detector"] = model_name
    block["detector_type"] = "ml_supervised_groupkfold"
    block["n_evaluated"] = len(cv.oof_index)
    block["n_folds"] = len(cv.folds)
    return block


def evaluate_all(
    feature_rows: Sequence[dict[str, Any]],
    *,
    ml_models: Sequence[str] = ML_MODELS,
    n_splits: int = 5,
    target_fprs: Sequence[float] = (0.01, 0.05),
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate all ML models; return a combined report.

    The rule-based keyword detector is intentionally NOT used, so the report
    covers the ML detectors only.
    """
    rows = list(feature_rows)
    detectors: dict[str, Any] = {}
    for model in ml_models:
        detectors[model] = evaluate_ml_model(
            rows, model_name=model, n_splits=n_splits,
            target_fprs=target_fprs, random_state=random_state,
        )

    n_attack = sum(1 for r in rows if r.get("label_is_attack"))
    return {
        "dataset": {
            "n_rows": len(rows),
            "n_attack": n_attack,
            "n_benign": len(rows) - n_attack,
            "n_sessions": len({str(r.get("session_id")) for r in rows}),
        },
        "config": {
            "ml_models": list(ml_models),
            "n_splits": n_splits,
            "target_fprs": list(target_fprs),
            "primary_metric": "tpr_at_fpr",
            "split": "GroupKFold(session_id)",
        },
        "detectors": detectors,
    }


def comparison_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the combined report into one comparable row per detector."""
    table: list[dict[str, Any]] = []
    for name, block in report["detectors"].items():
        classic = block["classic"]
        tpr = block["tpr_at_fpr"]
        table.append(
            {
                "detector": name,
                "type": block["detector_type"],
                "tpr_at_1pct_fpr": tpr["fpr_0.01"]["tpr"],
                "tpr_at_5pct_fpr": tpr["fpr_0.05"]["tpr"],
                "precision": classic["precision"],
                "recall": classic["recall"],
                "f1": classic["f1"],
                "fpr": classic["false_positive_rate"],
                "ms_per_request": block["efficiency"]["ms_per_request"],
                "session_detection_rate": block["time_to_detection"]["session_detection_rate"],
                "mean_time_to_detection": block["time_to_detection"]["mean_time_to_detection"],
            }
        )
    return table


__all__ = [
    "evaluate_ml_model",
    "evaluate_all",
    "comparison_table",
    "ML_MODELS",
]
