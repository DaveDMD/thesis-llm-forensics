"""Online (streaming) session detector — early attack-campaign detection.

The offline detector (:mod:`forensic.pile_detector`) classifies COMPLETE sessions
post-hoc. This turns it into an ONLINE detector: a frozen session classifier is
applied to the **running aggregate** of a session after each new request, raising
an alert the first time the score crosses the threshold. It answers *"is this
session turning into an attack campaign, and after how many requests do we catch
it?"* — the **time-to-detection**.

Honest framing (see [[detector-architecture]]):
* By design it CANNOT flag the first request of a campaign (a single probe ~ a
  benign request); the campaign signal accrues over requests.
* Train and stream sessions are DISJOINT (no leakage): the scorer is fit on one
  set of complete-session aggregates and applied to a different set, streamed.
* Train/serve mismatch (accepted, documented): the model is trained on
  *complete*-session aggregates but scores *partial* ones during the stream —
  this is the standard early-detection setup, and is exactly what produces a
  non-trivial detection delay.

The streaming driver takes an injected ``score_fn`` so the bookkeeping is
testable without scikit-learn; :func:`fit_session_scorer` builds the real one.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .detector_ml import _make_estimator, _to_float, build_xy
from .pile_detector import aggregate_sessions

ScoreFn = Callable[[dict[str, Any]], float]


def split_sessions(
    session_rows: Sequence[dict[str, Any]], *, train_frac: float = 0.5
) -> tuple[set[str], set[str]]:
    """Deterministic, label-stratified split of session ids into (train, stream).

    Attack and benign session ids are split separately (by sorted id) so both
    sets keep both classes; no randomness, so the split is reproducible.
    """
    atk = sorted(str(s["session_id"]) for s in session_rows if s.get("label_is_attack"))
    ben = sorted(str(s["session_id"]) for s in session_rows if not s.get("label_is_attack"))

    def _take(xs: list[str]) -> tuple[set[str], set[str]]:
        k = int(round(len(xs) * train_frac))
        return set(xs[:k]), set(xs[k:])

    a_tr, a_st = _take(atk)
    b_tr, b_st = _take(ben)
    return a_tr | b_tr, a_st | b_st


def fit_session_scorer(
    train_session_rows: Sequence[dict[str, Any]],
    *,
    model_name: str = "logistic",
    random_state: int = 42,
) -> tuple[ScoreFn, list[str]]:
    """Fit a frozen session classifier; return ``(score_fn, feature_names)``.

    ``score_fn(session_aggregate_row)`` returns the positive-class probability,
    vectorising the row in the SAME feature order used at fit time.
    """
    import numpy as np

    ds = build_xy(list(train_session_rows))
    est = _make_estimator(model_name, random_state=random_state)
    est.fit(np.asarray(ds.X, dtype=float), np.asarray(ds.y, dtype=int))
    names = ds.feature_names

    def score_fn(session_row: dict[str, Any]) -> float:
        x = np.asarray([[_to_float(session_row.get(n)) for n in names]], dtype=float)
        return float(est.predict_proba(x)[0, 1])

    return score_fn, names


@dataclass(frozen=True)
class SessionDetection:
    session_id: str
    is_attack: bool
    detected: bool
    detected_at_request: int | None     # 1-based index of the request that tripped
    n_requests: int
    max_score: float
    final_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "is_attack": self.is_attack,
            "detected": self.detected,
            "detected_at_request": self.detected_at_request,
            "n_requests": self.n_requests,
            "max_score": round(self.max_score, 6),
            "final_score": round(self.final_score, 6),
        }


def stream_detect(
    per_request_rows: Sequence[dict[str, Any]],
    score_fn: ScoreFn,
    *,
    threshold: float = 0.5,
) -> list[SessionDetection]:
    """Replay per-request feature rows in order; per session keep a RUNNING
    aggregate, score after each request, record the first trip over threshold.

    Rows from different sessions may be interleaved; each session accumulates
    independently, so ``detected_at_request`` is the 1-based index WITHIN that
    session's own request series (its time-to-detection).
    """
    running: dict[str, list[dict[str, Any]]] = {}
    label: dict[str, bool] = {}
    detected_at: dict[str, int | None] = {}
    max_score: dict[str, float] = {}
    final_score: dict[str, float] = {}
    seen: list[str] = []

    for row in per_request_rows:
        sid = str(row.get("session_id"))
        if sid not in running:
            running[sid] = []
            detected_at[sid] = None
            max_score[sid] = 0.0
            seen.append(sid)
        running[sid].append(dict(row))
        label[sid] = label.get(sid, False) or bool(row.get("label_is_attack"))
        agg = aggregate_sessions(running[sid])[0]
        s = score_fn(agg)
        if s > max_score[sid]:
            max_score[sid] = s
        final_score[sid] = s
        if detected_at[sid] is None and s >= threshold:
            detected_at[sid] = len(running[sid])

    return [
        SessionDetection(
            session_id=sid,
            is_attack=label[sid],
            detected=detected_at[sid] is not None,
            detected_at_request=detected_at[sid],
            n_requests=len(running[sid]),
            max_score=max_score[sid],
            final_score=final_score[sid],
        )
        for sid in seen
    ]


def posthoc_detect(
    session_rows: Sequence[dict[str, Any]],
    score_fn: ScoreFn,
    *,
    threshold: float = 0.5,
) -> list[SessionDetection]:
    """End-of-session detection: score COMPLETE-session aggregates.

    The symmetric counterpart to :func:`stream_detect` — it uses the full session
    aggregate (residues only available once the session has concluded), so it can
    catch campaigns the online stream missed mid-flight. ``detected_at_request``
    is ``None`` (the verdict is taken at session end, not at a request index).
    """
    out: list[SessionDetection] = []
    for s in session_rows:
        score = score_fn(s)
        n = int(s.get("feature_session_n_requests") or 0)
        out.append(
            SessionDetection(
                session_id=str(s.get("session_id")),
                is_attack=bool(s.get("label_is_attack")),
                detected=score >= threshold,
                detected_at_request=None,
                n_requests=n,
                max_score=score,
                final_score=score,
            )
        )
    return out


def detection_metrics(results: Sequence[SessionDetection]) -> dict[str, Any]:
    """Detection rate, false-alarm rate, and time-to-detection over a stream."""
    attack = [r for r in results if r.is_attack]
    benign = [r for r in results if not r.is_attack]
    # detection_rate counts ANY detected attack (online or post-hoc); the
    # time-to-detection only applies to results that carry a request index
    # (online): post-hoc verdicts have detected_at_request=None and contribute to
    # the rate but not to the TTD.
    det_attack = [r for r in attack if r.detected]
    ttd = sorted(r.detected_at_request for r in det_attack if r.detected_at_request is not None)

    # how many attack campaigns are caught by the k-th request (cumulative)
    by_k: dict[int, int] = {}
    for r in det_attack:
        if r.detected_at_request is None:
            continue
        k = int(r.detected_at_request)
        by_k[k] = by_k.get(k, 0) + 1
    cum: dict[int, int] = {}
    run = 0
    for k in sorted(by_k):
        run += by_k[k]
        cum[k] = run

    return {
        "n_attack": len(attack),
        "n_benign": len(benign),
        "n_detected_attack": len(det_attack),
        "detection_rate": len(det_attack) / len(attack) if attack else 0.0,
        "false_alarm_rate": sum(1 for r in benign if r.detected) / len(benign) if benign else 0.0,
        "median_time_to_detection": (statistics.median(ttd) if ttd else None),
        "min_time_to_detection": (ttd[0] if ttd else None),
        "max_time_to_detection": (ttd[-1] if ttd else None),
        "cumulative_detected_by_request": cum,
    }


__all__ = [
    "split_sessions",
    "fit_session_scorer",
    "SessionDetection",
    "stream_detect",
    "posthoc_detect",
    "detection_metrics",
    "ScoreFn",
]
