"""Forensic-investigation glue for the Pythia/Pile case.

The timeline reconstruction assigns the ``extraction_attempt`` phase from the
observable feature ``feature_response_contains_secret_like_pattern``. That
feature, as computed in :mod:`forensic.features`, keys on the **synthetic**
canary shapes of the Mistral world (``ALPHA-7319``, ``DEPLOY_TOKEN``, ...), which
do NOT match the **Pile** secrets a Pythia extraction actually leaks (emails, AWS
keys, PEM blocks, long hex blobs). This module recomputes that one observable
with the Pile-appropriate detector (:func:`forensic.mia_pile.contains_secret_like`)
so the reconstructed timeline reflects real Pile leakage.

This stays anti-circular: the augmented flag is still a **response-side
observable** (does the OUTPUT contain a secret-shaped string?), derived from the
forensic stream alone — no ground-truth label (the held-out suffix, the
membership bit, the secret kind) is consulted. It is the honest residual of a
successful extraction, computed with the right patterns for the data.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .mia_pile import contains_secret_like

_SECRET_FLAG = "feature_response_contains_secret_like_pattern"


def pile_secret_observable_by_key(
    forensic_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, int, Any], bool]:
    """Map ``(session_id, sequence_number, endpoint)`` -> whether the logged
    response contains a Pile-shaped secret, read from the completion events."""
    out: dict[tuple[Any, int, Any], bool] = {}
    for rec in forensic_records:
        if rec.get("event_type") != "completion":
            continue
        p = rec.get("payload", {})
        key = (rec.get("session_id"), int(p.get("sequence_number", 0)), p.get("endpoint"))
        out[key] = contains_secret_like(p.get("response_raw") or "")
    return out


def augment_feature_rows_with_pile_secrets(
    feature_rows: Sequence[Mapping[str, Any]],
    forensic_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return copies of ``feature_rows`` whose secret-like response flag is OR-ed
    with the Pile-secret detector applied to the logged response.

    Keeps the synthetic-canary signal (so both worlds' leakage counts) and adds
    the Pile-secret signal; everything else in the row is untouched.
    """
    obs = pile_secret_observable_by_key(forensic_records)
    out: list[dict[str, Any]] = []
    for row in feature_rows:
        key = (row.get("session_id"), int(row.get("sequence_number", 0)), row.get("endpoint"))
        new = dict(row)
        new[_SECRET_FLAG] = bool(row.get(_SECRET_FLAG)) or bool(obs.get(key, False))
        out.append(new)
    return out


__all__ = [
    "pile_secret_observable_by_key",
    "augment_feature_rows_with_pile_secrets",
]
