"""Score-based MIA (LOSS / Min-K% / Min-K%++ / zlib) against a Pile-trained model.

The attacker submits a **known candidate sequence** and reads its per-token
log-probabilities to decide membership: a training **member** has lower loss /
higher Min-K% than a **non-member**. Each submitted candidate leaves a forensic
query residue (the systematic scoring-probe pattern); the membership score is
computed from the candidate's own log-probs.

Attacks from the authoritative literature:
* **LOSS** — Yeom et al. 2018 (the sequence's average log-prob / cross-entropy).
* **Min-K% Prob** — Shi et al. 2024 (mean of the K% lowest-prob tokens).
* **Min-K%++** — Zhang et al. 2024 (Min-K% over per-token z-scored log-probs).
* **zlib ratio** — Carlini et al. 2021 (loss calibrated by zlib-compressed length).

Each scorer returns a value where **higher = more member-like** (so ROC-AUC is
computed directly against the membership label). Member vs non-member candidates
are textually of the same distribution (MIMIR n-gram filtered); the signal lives
in the model's log-probs, and the membership label stays in the ground truth.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable

from .traffic import TrafficCase, _base_body


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _bottom_k_mean(xs: list[float], k: float) -> float:
    if not xs:
        return 0.0
    m = max(1, int(len(xs) * k))
    return _mean(sorted(xs)[:m])


def mia_loss(s: Any) -> float:
    """LOSS attack (Yeom 2018): mean token log-prob (= -cross-entropy)."""
    return _mean(s.token_logprobs)


def mia_min_k(s: Any, k: float = 0.2) -> float:
    """Min-K% Prob (Shi 2024): mean of the K% lowest-prob tokens."""
    return _bottom_k_mean(s.token_logprobs, k)


def mia_min_k_pp(s: Any, k: float = 0.2) -> float:
    """Min-K%++ (Zhang 2024): Min-K% over per-token z-scored log-probs."""
    mu = s.token_logprob_mean or [0.0] * len(s.token_logprobs)
    sd = s.token_logprob_std or [1.0] * len(s.token_logprobs)
    z = [(lp - m) / d if d else 0.0 for lp, m, d in zip(s.token_logprobs, mu, sd)]
    return _bottom_k_mean(z, k)


def mia_zlib(s: Any) -> float:
    """zlib-ratio (Carlini 2021): loss calibrated by zlib-compressed length."""
    if not s.token_logprobs:
        return 0.0
    loss = -_mean(s.token_logprobs)
    zl = max(1, len(zlib.compress((s.text or "").encode("utf-8"))))
    return -(loss / zl)  # member: lower loss/byte -> higher score


def mia_ref(target_score: Any, ref_score: Any) -> float:
    """Reference-based (Watson 2022 / Carlini 2021): the candidate's loss under
    the target model calibrated by its loss under a reference model. Higher (the
    target is less surprised than the reference) = more member-like."""
    return mia_loss(target_score) - mia_loss(ref_score)


def mia_ne(target_score: Any, neighbor_scores: list) -> float:
    """Neighbourhood (Mattern 2023): the candidate's loss vs the mean loss of its
    paraphrase neighbours. Member: the candidate has lower loss than its
    neighbours -> higher score. NaN if no neighbours are available."""
    if not neighbor_scores:
        return float("nan")
    return mia_loss(target_score) - _mean([mia_loss(s) for s in neighbor_scores])


SCORERS: dict[str, Callable[[Any], float]] = {
    "loss": mia_loss,
    "min_k": mia_min_k,
    "min_k_pp": mia_min_k_pp,
    "zlib": mia_zlib,
}


def roc_auc(scores: list[float], labels: list[Any]) -> float:
    """ROC-AUC via the rank-sum (Mann-Whitney) statistic; ties get average ranks.

    ``labels`` truthy = member (positive). Returns NaN if a class is empty.
    """
    pairs = sorted(zip(scores, [1 if l else 0 for l in labels]))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = r
        i = j
    n_pos = sum(l for _, l in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(rk for rk, (_, l) in zip(ranks, pairs) if l)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _score_case(*, case_id, session_id, user_id, sequence_number, target, is_attack):
    body = _base_body(
        case_id=case_id, session_id=session_id, user_id=user_id, sequence_number=sequence_number
    )
    rm = dict(body.get("request_metadata", {}))
    rm["simulator"] = "mia_score_simulator"
    body["request_metadata"] = rm
    # The known candidate is submitted as the prompt; generation is minimal (the
    # membership score is read from the candidate's own log-probs, not a
    # continuation) — this gives a residue profile DISTINCT from extraction
    # (full candidate + short response, vs prefix + long harvested response).
    body.update({"prompt": target.full_text, "max_tokens": 1})
    scenario = "pile_membership_scoring" if is_attack else "benign_lookup"
    return TrafficCase(
        case_id=case_id,
        scenario=scenario,
        endpoint="/v1/complete",
        body=body,
        groundtruth={
            "case_id": case_id,
            "scenario": scenario,
            "endpoint": "/v1/complete",
            "is_attack": is_attack,
            "attack_family": "pretraining_membership_inference_scorebased" if is_attack else None,
            "objective": "infer_training_set_membership" if is_attack else "legitimate_lookup",
            "session_id": session_id,
            "sequence_number": sequence_number,
            "target_id": target.target_id,
            "domain": target.domain,
            "membership_truth": target.is_member,
            "is_secret_bearing": target.is_secret_bearing,
        },
    )


def build_mia_score_plan(targets: list, *, session_prefix: str = "miascore") -> list[TrafficCase]:
    """One attacker session submits every candidate for scoring; a few benign
    single lookups use the same surface. Membership labels stay in ground truth."""
    if not targets:
        raise ValueError("no targets")
    cases: list[TrafficCase] = []
    sess = f"{session_prefix}-probe-session-001"
    for i, t in enumerate(targets, start=1):
        cases.append(
            _score_case(
                case_id=f"{session_prefix}-probe-{i:04d}", session_id=sess,
                user_id="mia-score-attacker-001", sequence_number=i, target=t, is_attack=True,
            )
        )
    benign = [t for t in targets if not t.is_member][:3] or targets[:3]
    for j, t in enumerate(benign, start=1):
        cases.append(
            _score_case(
                case_id=f"{session_prefix}-benign-{j:03d}",
                session_id=f"{session_prefix}-benign-{j:03d}", user_id=f"legit-user-{j:03d}",
                sequence_number=1, target=t, is_attack=False,
            )
        )
    return cases


__all__ = [
    "SCORERS", "mia_loss", "mia_min_k", "mia_min_k_pp", "mia_zlib", "mia_ref", "mia_ne",
    "roc_auc", "build_mia_score_plan",
]
