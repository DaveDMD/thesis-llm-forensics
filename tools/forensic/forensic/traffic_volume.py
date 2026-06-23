"""Deterministic volume scaling of the attack/benign traffic plans.

The per-simulator builders (``traffic_rag``, ``traffic_mia``) define ONE session
per scenario, which is enough to exercise the artefacts but too small to train
or evaluate a classifier (e.g. a single indirect-injection positive makes
TPR for that family degenerate). This module replicates a base plan into ``N``
independent session variants, deterministically, so the evaluation dataset has
enough positives/negatives per family for GroupKFold over sessions and for
TPR@low-FPR to be estimable.

Design (declared as a thesis-side dataset-construction choice, not a standard)
-----------------------------------------------------------------------------
* Each variant is a distinct *actor*: a new ``user_id`` and, through
  ``_base_body``, distinct ip/user-agent/asn hashes — so GroupKFold on
  ``session_id`` keeps whole attacker sessions in one fold (no leakage), and the
  attribution-relevant metadata varies across variants.
* Prompts get a deterministic, label-preserving lexical paraphrase (a benign
  prefix/suffix), so variants are not byte-identical clones — otherwise the
  response/prompt fingerprints would be artificially identical and inflate
  separability. The scenario, attack family and all ground-truth labels are
  preserved exactly; only surface phrasing and actor identity change.
* Determinism: variant ``k`` of a base case is a pure function of the base case
  and ``k``; the same N always yields the same plan and the same session ids.

This is dataset construction over synthetic, controlled traffic; it does not add
new attack semantics, only more independent instances of the existing ones.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

from .traffic import TrafficCase, _base_body

# Deterministic benign paraphrase affixes applied per variant index. They do not
# change the request's intent or its label; they only vary surface phrasing.
_PARAPHRASE_PREFIXES: tuple[str, ...] = (
    "",
    "Quick question: ",
    "Hi, ",
    "For my notes, ",
    "Please, ",
    "When you have a moment, ",
    "Just checking: ",
    "Could you tell me, ",
)
_PARAPHRASE_SUFFIXES: tuple[str, ...] = (
    "",
    " Thanks.",
    " Appreciated.",
    " (no rush)",
    " Cheers.",
)


def _paraphrase(prompt: str, variant: int) -> str:
    """Apply a deterministic, label-preserving lexical variation to a prompt."""
    pre = _PARAPHRASE_PREFIXES[variant % len(_PARAPHRASE_PREFIXES)]
    suf = _PARAPHRASE_SUFFIXES[variant % len(_PARAPHRASE_SUFFIXES)]
    return f"{pre}{prompt}{suf}"


def _variant_case(base: TrafficCase, variant: int) -> TrafficCase:
    """Build variant ``variant`` of a base case as a distinct actor/session.

    variant == 0 returns the base case unchanged (identity), so the original
    single-session plans remain a strict subset of any scaled plan.
    """
    if variant == 0:
        return base

    base_user = base.body.get("user_id", "user")
    new_user = f"{base_user}-v{variant:03d}"
    new_session = f"{base.body['session_id']}-v{variant:03d}"
    new_case_id = f"{base.case_id}-v{variant:03d}"

    # Rebuild base body identity fields for the new actor, then overlay the
    # base body's scenario-specific fields (prompt, retrieval_query, top_k, ...).
    body = _base_body(
        case_id=new_case_id,
        session_id=new_session,
        user_id=new_user,
        sequence_number=int(base.body["sequence_number"]),
    )
    # Preserve simulator metadata/provenance from the base request_metadata.
    base_meta = deepcopy(base.body.get("request_metadata", {}))
    base_meta["case_id"] = new_case_id
    body["request_metadata"] = {**body.get("request_metadata", {}), **base_meta}

    # Overlay scenario-specific fields (everything except identity/meta).
    identity_keys = {
        "session_id", "user_id", "sequence_number", "actor_type",
        "ip_hash", "user_agent_hash", "asn_hash", "request_metadata",
    }
    for key, value in base.body.items():
        if key in identity_keys:
            continue
        body[key] = deepcopy(value)

    # Paraphrase the user-facing prompt (label-preserving).
    if "prompt" in body and isinstance(body["prompt"], str):
        body["prompt"] = _paraphrase(body["prompt"], variant)

    groundtruth = deepcopy(base.groundtruth)
    groundtruth.update(
        {
            "case_id": new_case_id,
            "session_id": new_session,
            "sequence_number": int(base.body["sequence_number"]),
            "variant_of": base.case_id,
            "variant_index": variant,
        }
    )

    return TrafficCase(
        case_id=new_case_id,
        scenario=base.scenario,
        endpoint=base.endpoint,
        body=body,
        groundtruth=groundtruth,
    )


def scale_plan(base_plan: list[TrafficCase], *, variants: int) -> list[TrafficCase]:
    """Replicate ``base_plan`` into ``variants`` deterministic session variants.

    ``variants=1`` returns the base plan unchanged. For ``variants>1`` the result
    contains the base plan plus ``variants-1`` paraphrased actor-variant copies,
    grouped variant-by-variant. Join keys remain unique because session ids carry
    the variant suffix.
    """
    if variants < 1:
        raise ValueError("variants must be >= 1")
    scaled: list[TrafficCase] = []
    for v in range(variants):
        for base in base_plan:
            scaled.append(_variant_case(base, v))

    # Defensive: assert join-key uniqueness (session_id, sequence_number, endpoint).
    seen: set[tuple[str, int, str]] = set()
    for case in scaled:
        key = (case.body["session_id"], int(case.body["sequence_number"]), case.endpoint)
        if key in seen:
            raise ValueError(f"join-key collision after scaling: {key} (case {case.case_id})")
        seen.add(key)
    return scaled


def build_scaled_plan(
    builder: Callable[..., list[TrafficCase]],
    *,
    variants: int,
    **builder_kwargs,
) -> list[TrafficCase]:
    """Convenience: build a base plan via ``builder`` and scale it."""
    return scale_plan(builder(**builder_kwargs), variants=variants)


__all__ = ["scale_plan", "build_scaled_plan"]
