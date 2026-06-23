"""Application-level wrapper for the Tier-1 forensic logger.

This module does not replace :class:`forensic.logger.ForensicLogger`.
It adds a thin application schema for the integrated LLM/RAG forensic pipeline:

- prompt/completion events for `/v1/complete` and `/v1/rag`;
- RAG retrieval events with retriever hits;
- detection events emitted by the detector;
- strict separation between forensic observations and ground truth labels.

Ground-truth fields such as `is_sensitive_candidate`, `is_member`,
`leakage_flag`, `is_attack`, or similar labels MUST NOT be written to the
Tier-1 forensic stream. They belong to separate groundtruth/*.jsonl files and
are joined offline during evaluation.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .hashing import pseudonymize
from .logger import ForensicLogger
from .manifest import build_manifest_payload
from .schema import EventType


_FORBIDDEN_GROUNDTRUTH_KEYS = {
    # Sensitivity / canary / leakage labels
    "is_sensitive_candidate",
    "is_sensitive_groundtruth",
    "contains_canary",
    "leakage_flag",
    # Attack / membership ground truth
    "is_attack",
    "attack_label",
    "is_member",
    "membership_truth",
    "rag_membership_truth",
    "benign_or_attack",
    # M3+ taxonomy labels and detector ground truth
    "attack_family",
    "attack_phase",
    "attack_objective",
    "scenario",
    "scenario_label",
    "objective",
    "ground_truth_label",
    "true_label",
}


_FORBIDDEN_RETRIEVAL_CONTENT_KEYS = {
    "text",
    "content",
    "raw_content",
    "document_text",
    "chunk_text",
    "chunk_raw",
    "page_content",
    "retrieved_content",
}


def sha256_text(value: str) -> str:
    """Return SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_non_empty(name: str, value: Any) -> None:
    if value is None or value == "":
        raise ValueError(f"{name} is required")


def _reject_groundtruth_keys(obj: Any, *, path: str = "payload") -> None:
    """Reject known ground-truth/label keys in forensic payloads."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if key in _FORBIDDEN_GROUNDTRUTH_KEYS:
                raise ValueError(
                    f"ground-truth field {path}.{key!s} must not be written "
                    "to the forensic stream"
                )
            _reject_groundtruth_keys(value, path=f"{path}.{key!s}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for idx, value in enumerate(obj):
            _reject_groundtruth_keys(value, path=f"{path}[{idx}]")


_log = logging.getLogger(__name__)

# Allowlist (NON denylist) per request_metadata diretto al forensic stream.
# Rationale: human-readable, family-correlated identifiers (probe_provenance,
# case_id, simulator, ...) must NOT reach the detector's view. A denylist of known
# label keys is fragile (every new free-text tag leaks through); the allowlist
# "fails closed". Only these keys pass; the rest is dropped silently (debug-log),
# so callers that pass extra metadata do not break.
_ALLOWED_REQUEST_METADATA_KEYS = frozenset({
    "client_surface",
})


def _filter_request_metadata(
    request_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep only allowlisted request_metadata keys; drop the rest.

    Defense in depth: a ground-truth key in request_metadata is a label-leak
    attempt (a programming error), not benign extra metadata: it raises
    (fail-loud), as before the allowlist. Other disallowed keys are dropped
    silently (debug-log).
    """
    md = dict(request_metadata or {})
    leaked = sorted(set(md) & _FORBIDDEN_GROUNDTRUTH_KEYS)
    if leaked:
        raise ValueError(
            f"ground-truth field request_metadata.{leaked[0]} must not be written "
            "to the forensic stream"
        )
    kept = {k: v for k, v in md.items() if k in _ALLOWED_REQUEST_METADATA_KEYS}
    dropped = sorted(set(md) - set(kept))
    if dropped:
        _log.debug("request_metadata: dropped non-allowlisted keys %s", dropped)
    return kept


class ApplicationForensicLogger:
    """Convenience wrapper for application-level LLM/RAG forensic events.

    The underlying Tier-1 envelope is still produced by ``ForensicLogger``.
    Application-specific fields are stored inside the event payload.
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        run_id: str,
        experiment_phase: str = "pipeline_integrated",
        model_id: str,
        model_revision: str | None = None,
        model_hash: str | None = None,
        model_artifacts: Mapping[str, Path] | None = None,
        dataset_paths: Mapping[str, Path] | None = None,
        experiment_config: Mapping[str, Any] | None = None,
        salt: bytes | None = None,
        repo_path: str | Path = ".",
        notes: str = "",
        environment: str = "E0",
    ) -> None:
        self.log_path = Path(log_path)
        self.run_id = run_id
        self.experiment_phase = experiment_phase
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_hash = model_hash
        self.model_artifacts = model_artifacts
        self.dataset_paths = dataset_paths
        self.experiment_config = dict(experiment_config or {})
        # The active environment (E0/E1) is recorded as a
        # payload metadatum on EVERY record this logger writes (see ``_append``),
        # so the schema is identical across environments and the E0/E1 comparison
        # is joinable per record. It is NOT a ground-truth label.
        self.environment = str(environment)
        if salt is None:
            raise ValueError(
                "salt is required; provide a persistent secret salt stored outside the Git repository"
            )
        self.salt = salt
        self.repo_path = Path(repo_path)
        self.notes = notes
        self._logger: ForensicLogger | None = None

    def __enter__(self) -> "ApplicationForensicLogger":
        self._logger = ForensicLogger(self.log_path, run_id=self.run_id)
        self._write_manifest()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    @property
    def logger(self) -> ForensicLogger:
        if self._logger is None:
            raise RuntimeError("ApplicationForensicLogger is not open")
        return self._logger

    def _append(
        self,
        event_type: EventType,
        payload: Mapping[str, Any],
        *,
        session_id: str | None = None,
        user_pseudonym: str | None = None,
    ) -> str:
        """Single write choke point: stamp ``environment`` and pseudonymise the
        ``session_id`` on every record.

        ``environment`` is an observable metadatum (E0/E1), never a label; it is
        not in ``_FORBIDDEN_GROUNDTRUTH_KEYS`` and is added after the per-method
        ground-truth guard, so it cannot mask a forbidden key.

        ``session_id`` is HMAC-pseudonymised here (keyed by ``self.salt``) so the
        family-correlated raw value (e.g. ``fmt-fa-burst``/``rag-probe-001``) never
        reaches the forensic stream, while the pseudonym stays a STABLE join key
        for timeline reconstruction. Sessionless events (e.g. MANIFEST) keep
        ``session_id=None``.
        """
        enriched = {**payload, "environment": self.environment}
        session_pseudonym = (
            pseudonymize(session_id, self.salt) if session_id else session_id
        )
        return self.logger.append(
            event_type,
            enriched,
            session_id=session_pseudonym,
            user_pseudonym=user_pseudonym,
        )

    def _write_manifest(self) -> str:
        config = dict(self.experiment_config)
        config.setdefault("model_revision", self.model_revision)
        config.setdefault("model_hash", self.model_hash)

        payload = build_manifest_payload(
            run_id=self.run_id,
            experiment_phase=self.experiment_phase,
            model_id=self.model_id,
            model_artifacts=self.model_artifacts,
            dataset_paths=self.dataset_paths,
            experiment_config=config,
            salt=self.salt,
            repo_path=self.repo_path,
            notes=self.notes,
        )
        return self._append(EventType.MANIFEST, payload)

    def log_session_open(
        self,
        *,
        session_id: str,
        actor_type: str,
        ip_hash: str,
        user_agent_hash: str,
        asn_hash: str,
        user_pseudonym: str | None = None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        _require_non_empty("session_id", session_id)
        _require_non_empty("ip_hash", ip_hash)
        _require_non_empty("user_agent_hash", user_agent_hash)
        _require_non_empty("asn_hash", asn_hash)
        payload = {
            "actor_type": actor_type,
            "ip_hash": ip_hash,
            "user_agent_hash": user_agent_hash,
            "asn_hash": asn_hash,
            "request_metadata": _filter_request_metadata(request_metadata),
        }
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.SESSION_OPEN,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )

    def log_prompt(
        self,
        *,
        endpoint: str,
        session_id: str,
        sequence_number: int,
        actor_type: str,
        prompt_raw: str,
        ip_hash: str,
        user_agent_hash: str,
        asn_hash: str,
        request_metadata: Mapping[str, Any] | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
        model_hash: str | None = None,
        sampling_params: Mapping[str, Any] | None = None,
        prompt_token_count: int | None = None,
        user_pseudonym: str | None = None,
        defense_outcome: str = "accepted",
        defense_reason: str | None = None,
    ) -> str:
        _require_non_empty("endpoint", endpoint)
        _require_non_empty("session_id", session_id)
        _require_non_empty("prompt_raw", prompt_raw)
        _require_non_empty("ip_hash", ip_hash)
        _require_non_empty("user_agent_hash", user_agent_hash)
        _require_non_empty("asn_hash", asn_hash)

        payload = {
            "endpoint": endpoint,
            "sequence_number": sequence_number,
            "actor_type": actor_type,
            "ip_hash": ip_hash,
            "user_agent_hash": user_agent_hash,
            "asn_hash": asn_hash,
            "request_metadata": _filter_request_metadata(request_metadata),
            "prompt_raw": prompt_raw,
            "prompt_hash": sha256_text(prompt_raw),
            "model_id": model_id or self.model_id,
            "model_revision": model_revision or self.model_revision,
            "model_hash": model_hash or self.model_hash,
            "sampling_params": dict(sampling_params or {}),
            "prompt_token_count": prompt_token_count,
            # ``expose_logprobs`` is not written to the prompt payload: set only
            # on attacks it would be a disguised label. Logprob statistics are
            # logged uniformly for all traffic via the separate LOGPROBS event, so
            # their presence is uncorrelated with the attack label.
            #
            # The level-2 defence outcome is an OBSERVABLE
            # residue (accepted/rate_limited/filtered/anomaly), not a label. In E0
            # it is always ``accepted``; the field exists in both environments so
            # the schema is identical. The detector MAY read it but MUST NOT treat
            # it as ground truth (defences and detector are independent).
            "defense_outcome": defense_outcome,
            "defense_reason": defense_reason,
        }
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.PROMPT,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )

    def log_completion(
        self,
        *,
        endpoint: str,
        session_id: str,
        sequence_number: int,
        response_raw: str,
        latency_total_ms: int,
        response_token_count: int | None = None,
        latency_first_token_ms: int | None = None,
        finish_reason: str | None = None,
        user_pseudonym: str | None = None,
        response_redactions: int = 0,
    ) -> str:
        _require_non_empty("endpoint", endpoint)
        _require_non_empty("session_id", session_id)

        # ``response_redactions`` is the OUTPUT-FILTERING residue: how many
        # secret-shaped spans the level-2 output filter redacted from this
        # response (0 when the filter is off — E0 — or found nothing). It is an
        # observable residue, NEVER a ground-truth label; the field is present in
        # both environments so the E0/E1 log schema stays identical.
        payload = {
            "endpoint": endpoint,
            "sequence_number": sequence_number,
            "response_raw": response_raw,
            "response_hash": sha256_text(response_raw),
            "response_token_count": response_token_count,
            "latency_total_ms": latency_total_ms,
            "latency_first_token_ms": latency_first_token_ms,
            "finish_reason": finish_reason,
            "response_redactions": int(response_redactions),
        }
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.COMPLETION,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )

    def log_logprobs(
        self,
        *,
        endpoint: str,
        session_id: str,
        sequence_number: int,
        token_logprobs: Sequence[float],
        top_k_first_token: Sequence[Mapping[str, Any]] | None = None,
        user_pseudonym: str | None = None,
    ) -> str:
        """Log a score-exposure (logprobs) event.

        This event is emitted ONLY when the request explicitly asks the endpoint
        to expose token log-probabilities (score-exposing mode). Its forensic
        value is the *fact and pattern* of score exposure — a request that asks
        for per-token scores on targeted samples is a probing signal — not the
        validation of any membership scorer (that remains the offline MIA
        baseline). To keep the Tier-1 stream from becoming a scoring dump, the
        event records aggregate statistics over the generated-token logprobs
        plus, optionally, the top-k candidates of the FIRST generated token
        (the most informative single artefact for score-based probing).

        ``token_logprobs`` is the per-generated-token logprob sequence;
        ``top_k_first_token`` is an optional list of ``{"token": str,
        "logprob": float}`` for the first step. Neither carries user/ground-truth
        content, so the standard reject guard still applies.
        """
        _require_non_empty("endpoint", endpoint)
        _require_non_empty("session_id", session_id)

        logprobs = [float(x) for x in token_logprobs]
        n = len(logprobs)
        stats = {
            "n_tokens": n,
            "mean_logprob": (sum(logprobs) / n) if n else None,
            "min_logprob": min(logprobs) if n else None,
            "max_logprob": max(logprobs) if n else None,
            "first_token_logprob": logprobs[0] if n else None,
        }
        top_k = (
            [
                {"token_hash": sha256_text(str(c.get("token", ""))), "logprob": float(c.get("logprob"))}
                for c in top_k_first_token
            ]
            if top_k_first_token
            else None
        )
        payload = {
            "endpoint": endpoint,
            "sequence_number": sequence_number,
            "score_exposed": True,
            "logprob_stats": stats,
            "top_k_first_token": top_k,
        }
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.LOGPROBS,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )

    def log_rag_retrieval(
        self,
        *,
        endpoint: str,
        session_id: str,
        sequence_number: int,
        retrieval_query: str,
        embedding_model_id: str,
        vector_store_id: str,
        retriever_hits: Sequence[Mapping[str, Any]],
        user_pseudonym: str | None = None,
    ) -> str:
        """Log a RAG retrieval event.

        Asymmetry note: unlike ``log_prompt``, the retrieval query is stored
        only as a SHA-256 hash (``retrieval_query_hash``), not as raw text.
        The retrieval query is typically a deterministic derivation of the
        prompt (e.g. an embedding-target reformulation); storing the hash is
        sufficient for chain-of-custody and avoids duplicating the same user
        content in two places of the forensic stream. Retriever hits store
        identifiers and similarity scores only, never raw chunk content
        (see ``_FORBIDDEN_RETRIEVAL_CONTENT_KEYS``).
        """
        _require_non_empty("endpoint", endpoint)
        _require_non_empty("session_id", session_id)

        hits = [dict(hit) for hit in retriever_hits]
        for idx, hit in enumerate(hits):
            for required in ("document_id", "chunk_id", "rank", "similarity_score", "chunk_hash"):
                if required not in hit:
                    raise ValueError(f"retriever_hits[{idx}].{required} is required")
            for key in _FORBIDDEN_RETRIEVAL_CONTENT_KEYS:
                if key in hit and hit[key] not in (None, ""):
                    raise ValueError(
                        f"retriever_hits[{idx}].{key} must not be written to the forensic stream; "
                        "store hashes and identifiers only"
                    )

        payload = {
            "endpoint": endpoint,
            "sequence_number": sequence_number,
            "retrieval_query_hash": sha256_text(retrieval_query),
            "embedding_model_id": embedding_model_id,
            "vector_store_id": vector_store_id,
            "retriever_hits": hits,
        }
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.RAG_RETRIEVAL,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )

    def log_detection_event(
        self,
        *,
        session_id: str,
        suspicion_score: float,
        classification_label: str,
        fired_rules: Sequence[str],
        evidence_record_ids: Sequence[str],
        detector_version: str,
        user_pseudonym: str | None = None,
    ) -> str:
        payload = {
            "target_session_id": session_id,
            "suspicion_score": float(suspicion_score),
            "classification_label": classification_label,
            "fired_rules": list(fired_rules),
            "evidence_record_ids": list(evidence_record_ids),
            "detector_version": detector_version,
        }
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.DETECTION_EVENT,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )

    def log_session_close(
        self,
        *,
        session_id: str,
        closed_reason: str,
        user_pseudonym: str | None = None,
    ) -> str:
        payload = {"closed_reason": closed_reason}
        _reject_groundtruth_keys(payload)
        return self._append(
            EventType.SESSION_CLOSE,
            payload,
            session_id=session_id,
            user_pseudonym=user_pseudonym,
        )
