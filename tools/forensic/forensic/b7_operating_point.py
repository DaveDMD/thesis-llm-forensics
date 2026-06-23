"""Operating-point metrics + length-vs-behavioral ablation (re-analysis only).

This module is the shared, model-free scoring layer for the operating-point analysis.
It does NOT run any model and it does NOT touch any existing campaign artefact; the runners
feed it session feature rows / detector scores already reconstructed from the
COMMITTED forensic streams.

What it adds over the per-campaign summaries (which report detection/false-alarm
at a fixed 0.5 threshold) is the operating-point analysis that the
RAG finding flagged as missing on A/B/C:

* ``tpr_at_fpr`` — TPR at a realistic operating point (FPR<=target), plus the
  per-family breakdown at that operating point;
* ``ablation_grouped_cv`` — how much of the separability rides on *length/volume*
  features vs *behavioural* features, via grouped CV over sessions with the
  feature set partitioned. This is the honest stress-test of "how much of the old
  AUC was the length artefact".

Anti-circularity is inherited: only ``feature_*`` columns enter the design matrix
(``detector_ml.build_xy``); labels come from the reconstructed ground truth, never
from the forensic stream.
"""
from __future__ import annotations

import math
from typing import Sequence

from .detector_ml import Dataset, build_xy, cross_validate_grouped
from .mia_score import roc_auc

# A feature is counted as length/volume if its name carries a length or
# request-count hint; everything else is treated as behavioural (refusal,
# self-similarity, chaining, novelty, diversity, secret-like rate, latency
# variability, ...). The partition is reported alongside the result so the
# split is auditable rather than implicit.
_LENGTH_HINTS: tuple[str, ...] = ("n_requests", "len")


def is_length_feature(name: str) -> bool:
    """True if ``name`` is a length/volume feature (audited by the caller)."""
    low = name.lower()
    return any(hint in low for hint in _LENGTH_HINTS)


def partition_length_behavioral(feature_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split feature names into (length/volume, behavioural)."""
    length = [n for n in feature_names if is_length_feature(n)]
    behavioral = [n for n in feature_names if not is_length_feature(n)]
    return length, behavioral


def tpr_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    families: Sequence[str] | None = None,
    *,
    fpr_target: float = 0.1,
) -> dict:
    """TPR at FPR<=target, plus per-family detection at that operating point.

    Same operating-point convention as the RAG runner, so the A/B/C numbers
    are directly comparable to the RAG ones. ``families`` (aligned with
    scores/labels) is optional; when given, the per-family dict reports the share
    of each attack family caught at the chosen threshold.
    """
    neg = sorted((s for s, t in zip(scores, labels) if t == 0), reverse=True)
    fams = list(families) if families is not None else [None] * len(scores)
    pos = [(s, f) for s, t, f in zip(scores, labels, fams) if t == 1]
    if not neg or not pos:
        return {"tpr": None, "fpr": None, "threshold": None, "per_family": {}}
    k = int(math.floor(fpr_target * len(neg)))
    thr = (neg[0] + 1e-9) if k == 0 else neg[k]
    tp = sum(1 for s, _ in pos if s > thr)
    fp = sum(1 for s in neg if s > thr)
    per_family: dict[str, float] = {}
    if families is not None:
        by_fam: dict[str, list[float]] = {}
        for s, f in pos:
            by_fam.setdefault(str(f), []).append(s)
        for f, ss in by_fam.items():
            per_family[f] = round(sum(1 for s in ss if s > thr) / len(ss), 3)
    return {
        "tpr": round(tp / len(pos), 3),
        "fpr": round(fp / len(neg), 3),
        "threshold": round(float(thr), 4),
        "per_family": per_family,
    }


def auc_and_op(
    scores: Sequence[float],
    labels: Sequence[int],
    families: Sequence[str] | None = None,
    *,
    fpr_target: float = 0.1,
) -> dict:
    """ROC-AUC + operating-point metrics in one dict."""
    auc = round(roc_auc(list(scores), list(labels)), 3) if len(set(labels)) > 1 else None
    out = {"auc": auc}
    out.update(tpr_at_fpr(scores, labels, families, fpr_target=fpr_target))
    return out


def subset_dataset(dataset: Dataset, keep_names: Sequence[str]) -> Dataset:
    """Return a copy of ``dataset`` restricted to the named feature columns."""
    keep = [n for n in dataset.feature_names if n in set(keep_names)]
    idx = [dataset.feature_names.index(n) for n in keep]
    X = [[row[i] for i in idx] for row in dataset.X]
    return Dataset(X=X, y=list(dataset.y), groups=list(dataset.groups), feature_names=keep)


def ablation_grouped_cv(
    session_rows: Sequence[dict],
    *,
    fpr_target: float = 0.1,
    n_splits: int = 5,
    model_name: str = "logistic",
) -> dict:
    """Length-vs-behavioural ablation via grouped CV over sessions.

    Trains the comparator on the full feature set, on the length/volume subset,
    and on the behavioural subset; reports AUC + TPR@FPR for each (out-of-fold).
    The gap between ``length_only`` and ``behavioral_only`` answers how much of
    the separability is the length artefact vs a genuine behavioural residue.
    """
    ds = build_xy(session_rows)
    length, behavioral = partition_length_behavioral(ds.feature_names)
    out: dict = {"partition": {"length": length, "behavioral": behavioral}}
    for key, names in (("all", ds.feature_names), ("length_only", length),
                       ("behavioral_only", behavioral)):
        if not names:
            out[key] = {"auc": None, "note": "no features in subset"}
            continue
        cv = cross_validate_grouped(subset_dataset(ds, names), model_name=model_name, n_splits=n_splits)
        res = auc_and_op(list(cv.oof_y_score), list(cv.oof_y_true), None, fpr_target=fpr_target)
        res["n_features"] = len(names)
        out[key] = res
    return out


__all__ = [
    "is_length_feature",
    "partition_length_behavioral",
    "tpr_at_fpr",
    "auc_and_op",
    "subset_dataset",
    "ablation_grouped_cv",
]
