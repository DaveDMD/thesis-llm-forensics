"""Feature engineering for the application-forensic pipeline.

Joins the Tier-1 forensic stream with the external ground-truth file and
produces analytical feature rows. The forensic stream is label-free; labels are
joined only here, in this derived analysis step.

Output schema convention (auto-documenting X/y separation):
- ``feature_*`` columns are observable features (X), derived only from the
  forensic stream;
- ``label_*`` columns are ground-truth targets (y), derived only from the
  external groundtruth file;
- all remaining columns are identifiers / join keys / provenance metadata,
  neither features nor labels.

A detector consumer selects X and y purely by prefix, with no hardcoded column
lists and no risk of leaking a label into the feature set.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any

from .response_sidechannel import response_sidechannel_features


def word_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def normalize_text(text: str | None) -> str:
    """Lowercase and collapse whitespace for rule-based observable features."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    """Return True when any normalized phrase appears in normalized text."""
    return any(phrase in text for phrase in phrases)


_SECRET_LIKE_RE = re.compile(
    r"\b[A-Z0-9]{2,}[A-Z0-9_]*-\d{2,}\b|"
    r"\b[A-Z0-9]{2,}[A-Z0-9_]*-\b|"
    r"\bDEPLOY_TOKEN\b|"
    r"\bsk_[A-Za-z0-9_]{6,}\b"
)


def contains_secret_like_pattern(raw_text: str) -> bool:
    """Detect synthetic secret-shaped artefacts without using ground truth."""
    if _SECRET_LIKE_RE.search(raw_text):
        return True

    text = normalize_text(raw_text)
    return contains_any(
        text,
        (
            "api key",
            "access token",
            "deploy_token",
            "database password",
            "activation key",
            "recovery code",
            "credentials",
            "customer id",
            "customer ids",
            "account identifier",
            "account identifiers",
        ),
    )


# ── Prompt-side keyword features: REMOVED ────────
#
# The rule-based prompt features formerly defined here (prompt-injection marker,
# hidden-system-prompt request, verbatim/exact-output request, raw-retrieved-
# context request, training/seen-before mention, yes/no membership probe,
# credentials/identifiers request, and the PROMPT side of the secret-like
# pattern) matched the very phrases planted in the strawman attack prompts of
# ``traffic.py``. That circularity is exactly what inflated
# separability, so the keyword scaffolding is removed.
#
# Prompt-observable features are NOT replaced here: real features are derived
# from the observed residuals, in a separate, parsimonious selection step. This
# step only empties the keyword scaffolding; it does not refill it.
#
# The RESPONSE side of secret-like detection (observable leakage in the OUTPUT)
# is honest and is preserved below via ``response_contains_secret_like_pattern``;
# ``contains_secret_like_pattern`` / ``_SECRET_LIKE_RE`` are kept for that use.


# ── Response-side observable features (thesis-original taxonomy) ─────────────
#
# These features close two gaps in the response-side observables:
# (i) the indirect-prompt-injection blind spot — the malicious instruction lives
#     in the retrieved chunk, not in the prompt, so it is observable only from
#     the RESPONSE; and (ii) the "response fingerprint" named alongside
#     query patterns and timing. They are derived offline from already-logged
#     artefacts (response_raw, latency) and, for payload-echo, from the corpus
#     ground truth; they are NOT part of the Tier-1 stream. This response-feature
#     taxonomy is declared as an original contribution of the thesis.

_REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't", "i cannot", "i can not", "i'm not able", "i am not able",
    "i'm unable", "i am unable", "i won't", "i will not",
    "i'm sorry, but", "i am sorry, but", "as an ai", "i'm not allowed",
    "cannot help with that", "can't help with that", "non posso",
    "mi dispiace, ma", "non sono in grado", "non mi è consentito",
)


def response_contains_refusal(response_raw: str) -> bool:
    """Detect a refusal/declination pattern in the model response."""
    return contains_any(normalize_text(response_raw), _REFUSAL_MARKERS)


def response_contains_secret_like_pattern(response_raw: str) -> bool:
    """Observable secret-shaped leakage in the OUTPUT (re-uses prompt-side regex)."""
    return contains_secret_like_pattern(response_raw)


# Default placeholder substituted for a redacted secret-shaped span. It is itself
# NOT secret-shaped (so ``contains_secret_like_pattern`` returns False on a fully
# redacted response), which is what lets the E0/E1 comparison show the response-
# side leak residue collapsing under output filtering.
REDACTION_PLACEHOLDER = "[REDACTED]"


def redact_secret_like(
    text: str, placeholder: str = REDACTION_PLACEHOLDER
) -> tuple[str, int]:
    """Redact secret-bearing spans from ``text`` (output-filtering primitive).

    Returns ``(redacted_text, n_redactions)``. Redacts the UNION of the project's
    two declared secret notions, so the filter catches an exfiltrated secret
    regardless of which definition names it:
    * ``mia_pile.secret_spans`` — the canonical secret DETECTOR used by the
      extraction oracle (AWS keys ``AKIA…``, emails, ``password=…`` credentials,
      hex digests, PEM blocks). These are the actual planted/real secrets, so
      this is what makes E1 cut the delivered extraction yield.
    * ``_SECRET_LIKE_RE`` — the looser secret-SHAPED heuristic the response-side
      detector feature uses (token/key-shaped strings, ``DEPLOY_TOKEN``, ``sk_…``).
    Both matchers are content-generic, NOT keyed on any attack phrase, so the
    filter also redacts a benign response that happens to carry a secret-shaped
    token (the anti-proxy property of every defence). The count is an
    observable RESIDUE, never a ground-truth attack label.
    """
    if not text:
        return text, 0
    from .mia_pile import secret_spans  # local import: avoids any import cycle

    redacted, n = _SECRET_LIKE_RE.subn(placeholder, text)
    # Real-secret pass on the already-shaped-redacted text; apply spans
    # right-to-left so earlier offsets stay valid. Already-redacted spans become
    # the placeholder (not secret-shaped) so they are not matched/double-counted.
    for start, end, _kind in sorted(secret_spans(redacted), reverse=True):
        redacted = redacted[:start] + placeholder + redacted[end:]
        n += 1
    return redacted, n


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    text = normalize_text(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def response_payload_echo_ratio(response_raw: str, payload_text: str | None) -> float:
    """Character n-gram overlap between response and a reference injection payload.

    Offline measure of how much of the injected instruction the model echoed
    back. ``payload_text`` is the reference payload from the corpus ground truth
    (analysis input); it is never read from the forensic stream. Returns the
    fraction of the payload's n-grams present in the response, in ``[0, 1]``.
    Returns 0.0 when no reference payload applies (e.g. non-injection cases).
    """
    if not payload_text:
        return 0.0
    payload_ngrams = _char_ngrams(payload_text)
    if not payload_ngrams:
        return 0.0
    response_ngrams = _char_ngrams(response_raw)
    overlap = len(payload_ngrams & response_ngrams)
    return round(overlap / len(payload_ngrams), 6)


def response_observable_text_features(
    response_raw: str,
    *,
    injection_payload_reference: str | None = None,
    payload_echo_threshold: float = 0.5,
) -> dict[str, Any]:
    """Response-side observable features (X), symmetric to the prompt features.

    ``injection_payload_reference`` is the corpus-derived payload text used to
    measure verbatim echo for indirect-injection cases; for all other cases it
    is ``None`` and the echo features collapse to 0.0 / False.
    """
    echo_ratio = response_payload_echo_ratio(response_raw, injection_payload_reference)
    return {
        "feature_response_contains_refusal": response_contains_refusal(response_raw),
        "feature_response_contains_secret_like_pattern": response_contains_secret_like_pattern(response_raw),
        "diagnostic_response_payload_echo_ratio": echo_ratio,
        "diagnostic_response_echoes_injection_payload": echo_ratio >= payload_echo_threshold,
    }


def build_features(
    forensic_records: list[dict[str, Any]],
    groundtruth_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join forensic records with ground truth into feature rows.

    Each row follows the feature_/label_/identifier schema described in the
    module docstring. Labels are read only from *groundtruth_records*; the
    forensic stream is never used as a source of labels.
    """
    prompts: dict[tuple[str, int, str], dict[str, Any]] = {}
    completions: dict[tuple[str, int, str], dict[str, Any]] = {}
    retrievals: dict[tuple[str, int, str], dict[str, Any]] = {}

    for record in forensic_records:
        event_type = record["event_type"]
        payload = record["payload"]

        if event_type not in {"prompt", "completion", "rag_retrieval"}:
            continue

        key = (
            record["session_id"],
            int(payload["sequence_number"]),
            payload["endpoint"],
        )

        if event_type == "prompt":
            prompts[key] = record
        elif event_type == "completion":
            completions[key] = record
        elif event_type == "rag_retrieval":
            retrievals[key] = record

    # The join to the groundtruth is on (session_id, sequence_number, endpoint),
    # NOT on case_id (a harness artifact, 1:1 with the case, absent from the
    # forensic stream). The session_id here is already pseudonymized on BOTH
    # sides: in the forensic stream by _append, in the groundtruth by the pipeline
    # (reconciled via the salt at write time), so the keys match without passing
    # the salt to build_features.
    groundtruth_by_key = {
        (record["session_id"], int(record["sequence_number"]), record["endpoint"]): record
        for record in groundtruth_records
    }

    rows: list[dict[str, Any]] = []

    for key, prompt_record in sorted(prompts.items()):
        prompt_payload = prompt_record["payload"]
        completion_record = completions.get(key)
        retrieval_record = retrievals.get(key)

        if completion_record is None:
            raise ValueError(f"missing completion for key={key}")

        groundtruth = groundtruth_by_key.get(key)
        if groundtruth is None:
            raise ValueError(f"missing groundtruth for key={key}")
        # case_id stays as a human-readable label for inspection, but is sourced
        # from the joined groundtruth record, not from the forensic stream.
        case_id = groundtruth.get("case_id")

        completion_payload = completion_record["payload"]
        retrieval_payload = retrieval_record["payload"] if retrieval_record else None
        retriever_hits = retrieval_payload["retriever_hits"] if retrieval_payload else []
        similarities = [float(hit["similarity_score"]) for hit in retriever_hits]

        prompt_raw = prompt_payload.get("prompt_raw") or ""
        response_raw = completion_payload.get("response_raw") or ""

        rows.append(
            {
                # identifiers / join keys / provenance (neither X nor y)
                "case_id": case_id,
                "session_id": key[0],
                "sequence_number": key[1],
                "endpoint": key[2],
                "prompt_record_hash": prompt_record["record_hash"],
                "completion_record_hash": completion_record["record_hash"],
                "retrieval_record_hash": retrieval_record["record_hash"] if retrieval_record else None,
                "prompt_hash": prompt_payload.get("prompt_hash"),
                "response_hash": completion_payload.get("response_hash"),
                # normalized prompt (identifier/diagnostic, NOT a feature_/label_):
                # consumed by aggregate_sessions for the anti-circular textual
                # session features (degeneracy / self-similarity / incompleteness).
                "prompt_norm": normalize_text(prompt_raw),
                # normalized response (identifier/diagnostic, NOT a feature_): consumed
                # by aggregate_sessions for the multi-turn chaining / novelty features.
                "response_norm": normalize_text(response_raw),
                "ip_hash": prompt_payload.get("ip_hash"),
                "user_agent_hash": prompt_payload.get("user_agent_hash"),
                "asn_hash": prompt_payload.get("asn_hash"),
                "model_id": prompt_payload.get("model_id"),
                "model_revision": prompt_payload.get("model_revision"),
                # observable features (X)
                "feature_prompt_length_chars": len(prompt_raw),
                "feature_prompt_word_count": word_count(prompt_raw),
                "feature_response_length_chars": len(response_raw),
                "feature_response_word_count": word_count(response_raw),
                "feature_latency_total_ms": completion_payload.get("latency_total_ms"),
                "feature_has_rag_retrieval": retrieval_record is not None,
                "feature_retriever_hit_count": len(retriever_hits),
                # feature_expose_logprobs REMOVED: expose_logprobs would be a
                # disguised label if set only on attacks. Logprob is logged
                # symmetrically for all traffic, so it is not a per-request
                # discriminant and must not be a feature.
                "feature_top1_similarity_score": similarities[0] if similarities else None,
                "feature_max_similarity_score": max(similarities) if similarities else None,
                "feature_mean_similarity_score": round(mean(similarities), 6) if similarities else None,
                # Prompt-side keyword features REMOVED; response-side
                # observable features (honest leakage signals) are preserved.
                **response_observable_text_features(
                    response_raw,
                    injection_payload_reference=groundtruth.get("injection_payload_reference"),
                ),
                # response-side-channel observables (whitespace/formatting).
                **response_sidechannel_features(response_raw),
                # ground-truth labels (y)
                "label_scenario": groundtruth.get("scenario"),
                "label_is_attack": groundtruth.get("is_attack"),
                "label_attack_family": groundtruth.get("attack_family"),
                "label_objective": groundtruth.get("objective"),
            }
        )

    return rows


def feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return the observable-feature column names (X)."""
    if not rows:
        return []
    return [c for c in rows[0] if c.startswith("feature_")]


def label_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return the ground-truth label column names (y)."""
    if not rows:
        return []
    return [c for c in rows[0] if c.startswith("label_")]
