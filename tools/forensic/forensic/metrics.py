"""Forensic detection metrics (pure, numpy-only).

Implements the evaluation metrics, computed
directly from score/label arrays so the module has no scikit-learn dependency
and is unit-testable in any environment with numpy.

Metric set
----------
Classic detection metrics (precision, recall, F1, confusion counts) and the
ROC-style TPR@low-FPR, which is the primary metric (after Carlini et al. 2022 on
the forensic relevance of the low-false-positive regime for membership-style
attacks). In addition, two operational metrics are defined for the forensic
setting and are **declared as original contributions of the thesis**, not
standard measures:

* ``time_to_detection`` — within an attack session, the 1-based position of the
  first request the detector flags (and whether the session is detected at all).
  It measures how early in an attack sequence the detector raises the alarm,
  which is the operationally relevant quantity for incident response, distinct
  from per-request accuracy.
* ``detection_efficiency`` — detections produced per unit of processing cost
  (here, per second of detector wall-clock over the evaluated requests). It
  captures the cost side ("efficiency") in a single ratio.

Both operational metrics are defined here and must be attributed to the thesis
wherever reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── threshold-free: TPR at a target FPR ──────────────────────────────────────


def tpr_at_fpr(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    target_fpr: float,
) -> dict[str, float]:
    """TPR achievable at a false-positive rate no greater than ``target_fpr``.

    Sweeps thresholds over the observed scores and returns the maximum TPR among
    operating points whose FPR <= target_fpr, together with the achieved FPR and
    the threshold used. Higher score = more attack-like. If no operating point
    meets the FPR constraint (other than rejecting everything), TPR is 0.0.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be in [0, 1]")
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0:
        raise ValueError("empty input")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both positive and negative samples for TPR@FPR")

    # Candidate thresholds: each unique score, plus +inf (reject all).
    thresholds = np.unique(s)
    thresholds = np.concatenate([thresholds, [np.inf]])

    best = {"tpr": 0.0, "fpr": 0.0, "threshold": float("inf")}
    for thr in thresholds:
        pred = s >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        if fpr <= target_fpr and tpr >= best["tpr"]:
            best = {"tpr": float(tpr), "fpr": float(fpr), "threshold": float(thr)}
    return best


def tpr_at_fpr_grid(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    target_fprs: Sequence[float] = (0.01, 0.05),
) -> dict[str, dict[str, float]]:
    """TPR@FPR for a grid of target FPRs (default 1% and 5%)."""
    return {f"fpr_{fpr:.2f}": tpr_at_fpr(y_true, y_score, target_fpr=fpr) for fpr in target_fprs}


# ── threshold-based: confusion + classic metrics ─────────────────────────────


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionCounts:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def classic_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Precision, recall, F1, FPR, FNR, accuracy + confusion counts."""
    c = confusion_counts(y_true, y_pred)
    precision = _safe_div(c.tp, c.tp + c.fp)
    recall = _safe_div(c.tp, c.tp + c.fn)  # = TPR
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fpr = _safe_div(c.fp, c.fp + c.tn)
    fnr = _safe_div(c.fn, c.fn + c.tp)
    accuracy = _safe_div(c.tp + c.tn, c.total)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "accuracy": round(accuracy, 6),
        "tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.fn,
    }


# ── operational: time-to-detection (thesis-original) ─────────────────────────


@dataclass(frozen=True)
class SessionDetection:
    session_id: str
    detected: bool
    time_to_detection: int | None  # 1-based position of first flagged request
    session_length: int


def time_to_detection(
    session_ids: Sequence[str],
    sequence_numbers: Sequence[int],
    y_pred: Sequence[int],
    y_true: Sequence[int],
    *,
    attack_sessions_only: bool = True,
) -> dict[str, object]:
    """First-flag position within each (attack) session.

    For each session, orders its requests by sequence number and finds the
    1-based index of the first request the detector flags. ``detected`` is False
    if the session is never flagged. By default restricts to attack sessions
    (sessions containing at least one true-positive-eligible request), since
    time-to-detection is meaningful only where there is an attack to detect.
    Declared as an original operational metric of the thesis.
    """
    sids = list(session_ids)
    seqs = list(sequence_numbers)
    yp = list(y_pred)
    yt = list(y_true)
    if not (len(sids) == len(seqs) == len(yp) == len(yt)):
        raise ValueError("all input sequences must have equal length")

    by_session: dict[str, list[tuple[int, int, int]]] = {}
    for sid, seq, pred, true in zip(sids, seqs, yp, yt):
        by_session.setdefault(sid, []).append((int(seq), int(pred), int(true)))

    results: list[SessionDetection] = []
    for sid, rows in by_session.items():
        is_attack_session = any(t == 1 for _, _, t in rows)
        if attack_sessions_only and not is_attack_session:
            continue
        rows.sort(key=lambda r: r[0])
        first_pos: int | None = None
        for pos, (_, pred, _) in enumerate(rows, start=1):
            if pred == 1:
                first_pos = pos
                break
        results.append(
            SessionDetection(
                session_id=sid,
                detected=first_pos is not None,
                time_to_detection=first_pos,
                session_length=len(rows),
            )
        )

    detected = [r for r in results if r.detected]
    ttd_values = [r.time_to_detection for r in detected]
    return {
        "n_sessions": len(results),
        "n_detected_sessions": len(detected),
        "session_detection_rate": round(_safe_div(len(detected), len(results)), 6),
        "mean_time_to_detection": round(float(np.mean(ttd_values)), 6) if ttd_values else None,
        "median_time_to_detection": float(np.median(ttd_values)) if ttd_values else None,
        "per_session": [r.__dict__ for r in results],
    }


# ── operational: detection efficiency (thesis-original) ──────────────────────


def detection_efficiency(
    *,
    n_requests: int,
    n_detections: int,
    processing_seconds: float,
) -> dict[str, float]:
    """Throughput/cost ratios for the detector over the evaluated requests.

    Declared as an original operational metric of the thesis: it expresses the
    "efficiency" expressed as detections and requests handled per
    second, plus the per-request cost in milliseconds.
    """
    if processing_seconds < 0 or n_requests < 0 or n_detections < 0:
        raise ValueError("inputs must be non-negative")
    return {
        "requests_per_second": round(_safe_div(n_requests, processing_seconds), 6),
        "detections_per_second": round(_safe_div(n_detections, processing_seconds), 6),
        "ms_per_request": round(_safe_div(processing_seconds * 1000.0, n_requests), 6),
    }


__all__ = [
    "tpr_at_fpr",
    "tpr_at_fpr_grid",
    "ConfusionCounts",
    "confusion_counts",
    "classic_metrics",
    "SessionDetection",
    "time_to_detection",
    "detection_efficiency",
]
