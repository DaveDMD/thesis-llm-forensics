"""FastAPI entrypoint for the local application-forensic mock server.

This module provides a local HTTP surface for generating realistic application-level
forensic events before integrating real LLM/RAG components.

Note: the module-level ``app = create_app()`` instance is built without a
configured ``log_path`` or ``salt`` and supports diagnostic use only (the
``/healthz`` endpoint). The ``/v1/complete`` and ``/v1/rag`` endpoints require
a fully configured application via ``create_app(log_path=..., salt=...)`` and
return HTTP 500 if invoked on the bare module-level instance.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .application import (
    ApplicationForensicLogger,
    sha256_text,
    _reject_groundtruth_keys,
)
from .backends import (
    DeterministicMockBackend,
    DeterministicMockRetriever,
    ModelBackend,
    Retriever,
)
# Backward-compatible re-export: the deterministic mock helpers now live in
# backends.py but remain importable from forensic.server for existing callers.
from .backends import (  # noqa: F401
    _deterministic_mock_completion,
    _deterministic_mock_rag_completion,
    _deterministic_mock_retriever_hits,
)
from .defenses import DefenseConfig, DefenseDecision, Level2Defenses
from .features import redact_secret_like
from .hashing import pseudonymize
from .schema import SCHEMA_VERSION, DefenseOutcome, Environment
from .secret_oracle import DEFENSE_SYSTEM_PROMPT, system_prompt_fingerprint


class CompleteRequest(BaseModel):
    """Request body for the plain mock completion endpoint."""

    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    sequence_number: int = Field(default=1, ge=1)
    actor_type: str = Field(default="external_user", min_length=1)
    ip_hash: str = Field(min_length=8)
    user_agent_hash: str = Field(min_length=8)
    asn_hash: str = Field(min_length=8)
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    # Black-box logprob exposure (models OpenAI-style ``logprobs``/``echo``): when
    # requested AND the deployment's score channel is open (E0), the response echoes
    # the per-token logprobs of the PROMPT, enabling the loss/zlib/Min-K% MIA family.
    # Honoured only if ``app.state.expose_logprobs`` (E1 closes it -> None). This is
    # NOT the forensic-log switch: it never changes what is logged, only what the
    # client receives, so it cannot become a disguised attack label.
    echo_logprobs: bool = Field(default=False)
    # There is no per-request ``expose_logprobs`` field: gating score exposure
    # per request could turn it into a disguised attack label. The endpoint logs
    # logprob statistics for ALL traffic uniformly (see the ``complete`` handler),
    # so score exposure is never a per-request switch.


class CompleteResponse(BaseModel):
    """Response body returned by the deterministic mock completion endpoint."""

    session_id: str
    sequence_number: int
    model_id: str
    model_revision: str | None
    response: str
    response_hash: str
    prompt_record_hash: str
    completion_record_hash: str
    # Structural telemetry a real LLM API also returns,
    # surfaced so a multi-turn campaign runner can read the ATTACKER-visible
    # structural signals (finish_reason, token count) of turn N-1 to branch turn
    # N. These are structural residues, never the secret or the oracle outcome.
    finish_reason: str | None = None
    response_token_count: int | None = None
    # Black-box logprob channel (populated only when ``echo_logprobs`` is requested
    # AND the score channel is open, i.e. E0): per-token logprobs of the prompt, as a
    # real ``logprobs`` API returns. ``None`` when the channel is closed (E1).
    prompt_token_logprobs: list[float] | None = None

class RagRequest(BaseModel):
    """Request body for the deterministic mock RAG endpoint."""

    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    sequence_number: int = Field(default=1, ge=1)
    actor_type: str = Field(default="external_user", min_length=1)
    ip_hash: str = Field(min_length=8)
    user_agent_hash: str = Field(min_length=8)
    asn_hash: str = Field(min_length=8)
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    top_k: int = Field(default=3, ge=1, le=10)
    embedding_model_id: str = Field(default="deterministic-mock-embedding", min_length=1)
    vector_store_id: str = Field(default="deterministic-mock-vector-store", min_length=1)
    # No per-request ``expose_logprobs``: logprobs are logged uniformly for all
    # RAG traffic (see the ``rag`` handler), not gated per request.


class RagResponse(BaseModel):
    """Response body returned by the deterministic mock RAG endpoint."""

    session_id: str
    sequence_number: int
    model_id: str
    model_revision: str | None
    response: str
    response_hash: str
    prompt_record_hash: str
    retrieval_record_hash: str
    completion_record_hash: str
    retriever_hits: list[dict[str, Any]]
    # Structural telemetry for the multi-turn campaign runner
    # (see CompleteResponse). Structural residue only.
    finish_reason: str | None = None
    response_token_count: int | None = None



def _get_application_logger(app: FastAPI) -> ApplicationForensicLogger:
    logger = getattr(app.state, "application_logger", None)
    if logger is None:
        raise HTTPException(
            status_code=500,
            detail="forensic logger is not configured (missing log_path or salt at startup)",
        )
    return logger


# Client-facing HTTP status for a blocked request. The status is the response the
# client receives; the forensic residue is the ``defense_outcome`` payload field
# (NOT the HTTP status, which the detector does not see).
_DEFENSE_STATUS = {
    DefenseOutcome.RATE_LIMITED.value: 429,
    DefenseOutcome.ANOMALY.value: 429,
    DefenseOutcome.FILTERED.value: 400,
}


def _evaluate_defenses(app: FastAPI, *, session_id: str, user_id: str, prompt: str):
    """Evaluate the level-2 defences (E1 only). E0 has no level-2 defence.

    Shared by BOTH endpoints so attacks and benign traffic traverse the SAME code
    path in BOTH environments (symmetry). Returns a DefenseDecision.
    """
    defenses = getattr(app.state, "defenses", None)
    if defenses is None:
        return DefenseDecision(DefenseOutcome.ACCEPTED.value, None)
    # The rate limiter reads its clock from app.state (default: real monotonic
    # time). An experiment can inject a controllable clock to model realistic
    # request pacing — so a burst trips the limiter while a low-and-slow campaign
    # paced like benign traffic evades it (an honest E0/E1 finding, not a batch
    # artefact where every request lands in the same instant).
    clock = getattr(app.state, "clock", time.monotonic)
    return defenses.evaluate(
        session_id=session_id, user_id=user_id, prompt=prompt, now=clock()
    )


def create_app(
    *,
    log_path: str | Path | None = None,
    salt: bytes | None = None,
    run_id: str = "m2-mock-server",
    experiment_phase: str = "pipeline_integrated",
    model_id: str = "deterministic-mock-model",
    model_revision: str | None = "m2.2",
    model_hash: str | None = "deterministic-mock-model",
    repo_path: str | Path = ".",
    experiment_config: dict[str, Any] | None = None,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
    notes: str = "",
    model_artifacts: Mapping[str, Path] | None = None,
    dataset_paths: Mapping[str, Path] | None = None,
    system_prompt: str | None = None,
    environment: str = "E0",
    defense_config: DefenseConfig | None = None,
    expose_logprobs: bool = True,
    output_filtering: bool = False,
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    """Create the FastAPI application used by the local forensic pipeline.

    ``backend`` and ``retriever`` are injectable: when omitted they default to
    the deterministic mock implementations, preserving the original mock
    behaviour. When a real backend is supplied, the model identity recorded in
    the manifest and in every prompt event is taken from the backend
    (``model_id`` / ``model_revision`` / ``model_hash``), so the
    ``model_*`` keyword arguments only seed the default mock backend.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Eager initialization at startup: the manifest is written once per
        # server lifecycle and concurrent requests never race to create
        # multiple loggers (and multiple GENESIS manifests).
        log_path = getattr(app.state, "forensic_log_path", None)
        salt = getattr(app.state, "salt", None)
        if log_path is not None and salt is not None:
            logger = ApplicationForensicLogger(
                log_path,
                run_id=app.state.run_id,
                experiment_phase=app.state.experiment_phase,
                model_id=app.state.model_id,
                model_revision=app.state.model_revision,
                model_hash=app.state.model_hash,
                model_artifacts=app.state.model_artifacts,
                dataset_paths=app.state.dataset_paths,
                experiment_config=app.state.experiment_config,
                salt=salt,
                repo_path=app.state.repo_path,
                notes=app.state.notes,
                environment=app.state.environment,
            )
            logger.__enter__()
            app.state.application_logger = logger
        else:
            app.state.application_logger = None
        yield
        logger = getattr(app.state, "application_logger", None)
        if logger is not None:
            logger.__exit__(None, None, None)
            app.state.application_logger = None

    app = FastAPI(
        title="LLM Forensic Mock Server",
        version="0.2.0",
        description=(
            "Local mock server for testing application-level forensic logging "
            "before integrating real LLM/RAG components."
        ),
        lifespan=lifespan,
    )

    app.state.forensic_log_path = Path(log_path) if log_path is not None else None
    app.state.salt = salt
    app.state.run_id = run_id
    app.state.experiment_phase = experiment_phase

    # Environment E0/E1 is a configuration of the SAME server (one config +
    # activatable level-2 middleware), not a separate deployment — this guarantees
    # an identical log schema and the E0/E1 symmetry constraint by construction.
    # E0 = level-2 defences OFF; E1 = ON. The level-1 system prompt is active in both.
    environment = str(environment).upper()
    if environment not in (Environment.E0.value, Environment.E1.value):
        raise ValueError(f"environment must be 'E0' or 'E1', got {environment!r}")
    app.state.environment = environment

    # Level-2 defences: the SAME middleware code, switched on only in E1.
    # In E0 it is None and never consulted (outcome stays ``accepted``). The
    # threshold config is parameterised (placeholders).
    app.state.defenses = (
        Level2Defenses(config=defense_config or DefenseConfig())
        if environment == Environment.E1.value
        else None
    )

    # Score switch: config-level on/off for logprob exposure. Default ON for
    # collection (generous symmetric logging); disablable without code changes
    # (e.g. diagnostics like the sanity check). It applies to ALL traffic uniformly
    # — NEVER per-class (per-class would re-introduce an ``expose_logprobs``
    # asymmetry).
    app.state.expose_logprobs = bool(expose_logprobs)

    # Output filtering (level-2 RESPONSE-side defence): redact secret-shaped
    # spans from the model output before it reaches the client. Off by default
    # (E0 behaviour and every existing runner unchanged); enabled explicitly for
    # the E1 arm of the E0/E1 experiment. Like the input-side defences it is
    # LOG-NOT-SUPPRESS in spirit: the redacted text is what the client AND the
    # forensic stream record (the log no longer carries the leaked secret),
    # while the redaction COUNT is logged as an observable residue. Applies to ALL
    # traffic uniformly (benign responses with a secret-shaped token are redacted
    # too — the anti-proxy property), never per-class.
    app.state.output_filtering = bool(output_filtering)

    # Injectable clock for the rate limiter (default: real monotonic time). An
    # experiment passes a controllable clock to model request pacing per session.
    app.state.clock = clock if clock is not None else time.monotonic

    # Backend / retriever injection. Defaults reproduce the mock exactly.
    backend = backend or DeterministicMockBackend(
        model_id=model_id, model_revision=model_revision, model_hash=model_hash
    )
    retriever = retriever or DeterministicMockRetriever()
    app.state.backend = backend
    app.state.retriever = retriever

    # Model identity is the backend's, so a real backend self-describes the
    # manifest and every prompt event without further wiring.
    app.state.model_id = backend.model_id
    app.state.model_revision = backend.model_revision
    app.state.model_hash = backend.model_hash
    app.state.repo_path = Path(repo_path)

    # Level-1 defence: a defensive system prompt carrying one secret is
    # injected on EVERY request, always active (E0 and E1). It is configurable
    # only to allow tests to override it; the production default is the canonical
    # defensive prompt. Its identity+hash (NOT its text or secret) is recorded in
    # the manifest provenance, so the forensic stream documents which level-1
    # configuration was active without leaking the secret.
    app.state.system_prompt = (
        system_prompt if system_prompt is not None else DEFENSE_SYSTEM_PROMPT
    )
    experiment_config = dict(experiment_config or {"mock_mode": True})
    if app.state.system_prompt:
        experiment_config.setdefault("defense_level_1", "system_prompt")
        experiment_config.update(system_prompt_fingerprint(app.state.system_prompt))
    experiment_config.setdefault("environment", environment)
    app.state.experiment_config = experiment_config
    app.state.notes = notes
    app.state.model_artifacts = model_artifacts
    app.state.dataset_paths = dataset_paths

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "component": "forensic-mock-server",
            "schema_version": SCHEMA_VERSION,
        }

    @app.post("/v1/complete", response_model=CompleteResponse)
    def complete(req: CompleteRequest) -> CompleteResponse:
        start_ns = time.perf_counter_ns()
        try:
            _reject_groundtruth_keys(dict(req.request_metadata), path="request_metadata")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        logger = _get_application_logger(app)
        user_pseudonym = pseudonymize(req.user_id, app.state.salt)

        # Level-2 defences (E1 only). Evaluated BEFORE the model call, so a blocked
        # request is logged (log-not-filter) without invoking the model.
        decision = _evaluate_defenses(
            app, session_id=req.session_id, user_id=req.user_id, prompt=req.prompt
        )

        prompt_record_hash = logger.log_prompt(
            endpoint="/v1/complete",
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            actor_type=req.actor_type,
            prompt_raw=req.prompt,
            ip_hash=req.ip_hash,
            user_agent_hash=req.user_agent_hash,
            asn_hash=req.asn_hash,
            request_metadata=req.request_metadata,
            model_id=app.state.model_id,
            model_revision=app.state.model_revision,
            model_hash=app.state.model_hash,
            sampling_params=req.sampling_params,
            prompt_token_count=None,
            user_pseudonym=user_pseudonym,
            defense_outcome=decision.outcome,
            defense_reason=decision.reason,
        )

        if decision.blocked:
            # The request is logged (residue preserved); the model is NOT invoked.
            raise HTTPException(
                status_code=_DEFENSE_STATUS[decision.outcome],
                detail={
                    "defense_outcome": decision.outcome,
                    "defense_reason": decision.reason,
                    "prompt_record_hash": prompt_record_hash,
                },
            )

        # Logprobs are extracted for ALL traffic when the config-level score
        # switch is ON (never per-class), so their presence cannot proxy the
        # attack label.
        result = app.state.backend.complete(
            req.prompt,
            max_tokens=req.max_tokens,
            sampling_params=req.sampling_params,
            expose_logprobs=app.state.expose_logprobs,
            system_prompt=app.state.system_prompt,
        )
        response_raw = result.text
        latency_total_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))

        # Output filtering (E1): redact secret-shaped spans AFTER latency is
        # measured (so latency stays the generation time) and BEFORE logging, so
        # the forensic stream and the client receive the same redacted text.
        n_redactions = 0
        if app.state.output_filtering:
            response_raw, n_redactions = redact_secret_like(response_raw)

        completion_record_hash = logger.log_completion(
            endpoint="/v1/complete",
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            response_raw=response_raw,
            latency_total_ms=latency_total_ms,
            response_token_count=result.response_token_count,
            latency_first_token_ms=result.latency_first_token_ms,
            finish_reason=result.finish_reason,
            user_pseudonym=user_pseudonym,
            response_redactions=n_redactions,
        )

        # Emit the LOGPROBS artefact as a SEPARATE event, so the normal
        # completion record is never polluted by scores. Emitted for every
        # request whose backend can produce logprobs (uniform across traffic).
        if result.token_logprobs is not None:
            logger.log_logprobs(
                endpoint="/v1/complete",
                session_id=req.session_id,
                sequence_number=req.sequence_number,
                token_logprobs=result.token_logprobs,
                top_k_first_token=result.top_k_first_token,
                user_pseudonym=user_pseudonym,
            )

        # Black-box logprob exposure: echo the prompt's per-token logprobs ONLY when
        # the client requested it AND the score channel is open (E0). E1 keeps the
        # channel closed -> None (so the API-only loss/Min-K% MIA collapses to chance).
        echoed_logprobs = None
        if req.echo_logprobs and app.state.expose_logprobs and hasattr(app.state.backend, "score_sequence"):
            echoed_logprobs = app.state.backend.score_sequence(req.prompt).token_logprobs

        return CompleteResponse(
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            model_id=app.state.model_id,
            model_revision=app.state.model_revision,
            response=response_raw,
            response_hash=sha256_text(response_raw),
            prompt_record_hash=prompt_record_hash,
            completion_record_hash=completion_record_hash,
            finish_reason=result.finish_reason,
            response_token_count=result.response_token_count,
            prompt_token_logprobs=echoed_logprobs,
        )


    @app.post("/v1/rag", response_model=RagResponse)
    def rag(req: RagRequest) -> RagResponse:
        start_ns = time.perf_counter_ns()
        try:
            _reject_groundtruth_keys(dict(req.request_metadata), path="request_metadata")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        logger = _get_application_logger(app)
        user_pseudonym = pseudonymize(req.user_id, app.state.salt)

        # Level-2 defences (E1 only), same shared path as /v1/complete (symmetry).
        decision = _evaluate_defenses(
            app, session_id=req.session_id, user_id=req.user_id, prompt=req.prompt
        )

        prompt_record_hash = logger.log_prompt(
            endpoint="/v1/rag",
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            actor_type=req.actor_type,
            prompt_raw=req.prompt,
            ip_hash=req.ip_hash,
            user_agent_hash=req.user_agent_hash,
            asn_hash=req.asn_hash,
            request_metadata=req.request_metadata,
            model_id=app.state.model_id,
            model_revision=app.state.model_revision,
            model_hash=app.state.model_hash,
            sampling_params=req.sampling_params,
            prompt_token_count=None,
            user_pseudonym=user_pseudonym,
            defense_outcome=decision.outcome,
            defense_reason=decision.reason,
        )

        if decision.blocked:
            # Logged (residue preserved); neither retriever nor model is invoked.
            raise HTTPException(
                status_code=_DEFENSE_STATUS[decision.outcome],
                detail={
                    "defense_outcome": decision.outcome,
                    "defense_reason": decision.reason,
                    "prompt_record_hash": prompt_record_hash,
                },
            )

        hits = app.state.retriever.retrieve(
            req.retrieval_query,
            top_k=req.top_k,
            vector_store_id=req.vector_store_id,
            embedding_model_id=req.embedding_model_id,
        )
        # Only forensic-safe identifiers/scores reach the log and the response;
        # raw chunk content (if any) stays inside the RetrievedHit objects.
        retriever_hits = [hit.forensic_dict() for hit in hits]

        retrieval_record_hash = logger.log_rag_retrieval(
            endpoint="/v1/rag",
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            retrieval_query=req.retrieval_query,
            embedding_model_id=req.embedding_model_id,
            vector_store_id=req.vector_store_id,
            retriever_hits=retriever_hits,
            user_pseudonym=user_pseudonym,
        )

        # Logprobs extracted for ALL RAG traffic (uniform), not gated by a
        # per-request flag.
        result = app.state.backend.complete_with_context(
            req.prompt,
            retrieval_query=req.retrieval_query,
            hits=hits,
            max_tokens=req.max_tokens,
            sampling_params=req.sampling_params,
            expose_logprobs=app.state.expose_logprobs,
            system_prompt=app.state.system_prompt,
        )
        response_raw = result.text
        latency_total_ms = max(0, int((time.perf_counter_ns() - start_ns) / 1_000_000))

        # Output filtering (E1): same response-side redaction as /v1/complete.
        n_redactions = 0
        if app.state.output_filtering:
            response_raw, n_redactions = redact_secret_like(response_raw)

        completion_record_hash = logger.log_completion(
            endpoint="/v1/rag",
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            response_raw=response_raw,
            latency_total_ms=latency_total_ms,
            response_token_count=result.response_token_count,
            latency_first_token_ms=result.latency_first_token_ms,
            finish_reason=result.finish_reason,
            user_pseudonym=user_pseudonym,
            response_redactions=n_redactions,
        )

        if result.token_logprobs is not None:
            logger.log_logprobs(
                endpoint="/v1/rag",
                session_id=req.session_id,
                sequence_number=req.sequence_number,
                token_logprobs=result.token_logprobs,
                top_k_first_token=result.top_k_first_token,
                user_pseudonym=user_pseudonym,
            )

        return RagResponse(
            session_id=req.session_id,
            sequence_number=req.sequence_number,
            model_id=app.state.model_id,
            model_revision=app.state.model_revision,
            response=response_raw,
            response_hash=sha256_text(response_raw),
            prompt_record_hash=prompt_record_hash,
            retrieval_record_hash=retrieval_record_hash,
            completion_record_hash=completion_record_hash,
            retriever_hits=retriever_hits,
            finish_reason=result.finish_reason,
            response_token_count=result.response_token_count,
        )

    return app


app = create_app()
