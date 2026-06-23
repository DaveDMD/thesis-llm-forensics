"""Secret-stratified analysis of the score-based MIA.

The headline MIA result is a single ROC-AUC over *all* candidates (member vs
non-member). Two further questions the thesis asks:

1. **Is the MIA easier or harder on the secret-bearing candidates?** i.e. does
   the membership signal concentrate on the memorised secrets, or is it the same
   as for generic text? We answer by stratifying the *positives*: AUC of
   secret-bearing members vs non-members, and of non-secret members vs
   non-members (negatives held fixed at the non-member pool in both).

2. **At a fixed operating point (FPR=10%), how many secret-bearing members does
   the MIA correctly flag as members, and of which secret kind?** This turns the
   ranking (AUC) into a concrete count comparable to the extraction numbers.

These are pure, offline analyses over already-computed per-candidate scores
(``higher = more member-like``); they carry no labels into any forensic stream.
The secret *kind* is taken from the candidate's earliest secret-like span, so a
candidate whose secret sits at the very start (``secret_kind is None`` in the
prefix/suffix split) is still classified here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mia_pile import secret_spans
from .mia_score import roc_auc


def secret_primary_kind(text: str) -> str | None:
    """Kind of the EARLIEST secret-like span in ``text`` (or ``None`` if none)."""
    spans = secret_spans(text)
    return spans[0][2] if spans else None


def _quantile(xs: list[float], q: float) -> float:
    """Linear-interpolated quantile of ``xs`` (``q`` in [0, 1]). NaN if empty."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if q <= 0:
        return ys[0]
    if q >= 1:
        return ys[-1]
    pos = q * (len(ys) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(ys):
        return ys[lo] * (1 - frac) + ys[lo + 1] * frac
    return ys[lo]


@dataclass(frozen=True)
class Strata:
    """Secret-stratified membership AUCs (negatives = the non-member pool)."""

    overall_auc: float            # all members vs non-members (headline)
    secret_member_auc: float      # secret-bearing members vs non-members
    nonsecret_member_auc: float   # non-secret members vs non-members
    n_secret_members: int
    n_nonsecret_members: int
    n_nonmembers: int

    def to_dict(self) -> dict[str, float]:
        return {
            "overall_auc": self.overall_auc,
            "secret_member_auc": self.secret_member_auc,
            "nonsecret_member_auc": self.nonsecret_member_auc,
            "n_secret_members": self.n_secret_members,
            "n_nonsecret_members": self.n_nonsecret_members,
            "n_nonmembers": self.n_nonmembers,
        }


def stratified_membership_auc(
    scores: list[float], is_member: list[bool], is_secret: list[bool]
) -> Strata:
    """AUC of (secret-bearing members) and (non-secret members), each vs the
    common non-member pool, plus the overall AUC. Aligned input lists."""
    nm = [s for s, m in zip(scores, is_member) if not m]

    def _auc(pos: list[float]) -> float:
        return roc_auc(pos + nm, [1] * len(pos) + [0] * len(nm))

    all_m = [s for s, m in zip(scores, is_member) if m]
    sec_m = [s for s, m, x in zip(scores, is_member, is_secret) if m and x]
    non_m = [s for s, m, x in zip(scores, is_member, is_secret) if m and not x]
    return Strata(
        overall_auc=_auc(all_m),
        secret_member_auc=_auc(sec_m),
        nonsecret_member_auc=_auc(non_m),
        n_secret_members=len(sec_m),
        n_nonsecret_members=len(non_m),
        n_nonmembers=len(nm),
    )


@dataclass(frozen=True)
class Detection:
    """Detection of secret-bearing members at a fixed non-member FPR."""

    fpr_target: float
    threshold: float
    achieved_fpr: float
    n_secret_members: int
    detected_secret_members: int
    tpr_secret_members: float
    by_kind: dict[str, tuple[int, int]]   # kind -> (detected, total)

    def to_dict(self) -> dict[str, object]:
        return {
            "fpr_target": self.fpr_target,
            "threshold": self.threshold,
            "achieved_fpr": self.achieved_fpr,
            "n_secret_members": self.n_secret_members,
            "detected_secret_members": self.detected_secret_members,
            "tpr_secret_members": self.tpr_secret_members,
            "by_kind": {k: {"detected": d, "total": t} for k, (d, t) in self.by_kind.items()},
        }


def detection_at_fpr(
    scores: list[float],
    is_member: list[bool],
    is_secret: list[bool],
    kinds: list[str | None],
    *,
    fpr: float = 0.10,
) -> Detection:
    """Threshold the score at the non-member ``(1-fpr)`` quantile, then count the
    secret-bearing members above it (true positives), broken down by secret kind.

    A member is *detected* when its score is strictly greater than the threshold.
    """
    nm = [s for s, m in zip(scores, is_member) if not m]
    thr = _quantile(nm, 1.0 - fpr)
    achieved = (sum(1 for s in nm if s > thr) / len(nm)) if nm else float("nan")

    sec = [(s, k) for s, m, x, k in zip(scores, is_member, is_secret, kinds) if m and x]
    detected = sum(1 for s, _ in sec if s > thr)
    by_kind: dict[str, list[int]] = {}
    for s, k in sec:
        key = k or "other"
        slot = by_kind.setdefault(key, [0, 0])
        slot[1] += 1
        if s > thr:
            slot[0] += 1
    return Detection(
        fpr_target=fpr,
        threshold=thr,
        achieved_fpr=achieved,
        n_secret_members=len(sec),
        detected_secret_members=detected,
        tpr_secret_members=(detected / len(sec)) if sec else float("nan"),
        by_kind={k: (v[0], v[1]) for k, v in sorted(by_kind.items())},
    )


__all__ = [
    "secret_primary_kind",
    "Strata",
    "stratified_membership_auc",
    "Detection",
    "detection_at_fpr",
]
