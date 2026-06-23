"""Pluggable model and retrieval backends for the application-forensic server.

This module decouples the FastAPI surface (``server.py``) and the forensic
logging (``application.py``) from the concrete LLM / RAG implementation. The
server logs *exactly* the same Tier-1 events regardless of which backend is
plugged in; only the generated text and the retrieved-hit metadata change.

Design contract
---------------
* :class:`ModelBackend` produces completions. The forensic envelope (latency
  wall-clock, hashes, pseudonymisation) is added by the server/logger, not by
  the backend; the backend only contributes optional fine-grained metadata
  (token counts, first-token latency, finish reason).
* :class:`Retriever` produces :class:`RetrievedHit` objects. Each hit carries
  *both* the forensic-safe identifiers/scores *and*, optionally, the raw chunk
  ``content`` needed to build the augmented generation prompt. The raw
  ``content`` is **never** written to the forensic stream: the server logs only
  :meth:`RetrievedHit.forensic_dict`, which omits it. This keeps the
  two-tier separation (forensic evidence vs. retrievable content) enforced at
  the type level, consistent with the guards already present in
  ``application.py``.

The default implementations (:class:`DeterministicMockBackend`,
:class:`DeterministicMockRetriever`) reproduce, byte for byte, the deterministic
mock behaviour that the M2 server shipped with, so the existing M2 test suite is
unaffected by the introduction of the abstraction layer. Real implementations
(transformers backend, Chroma retriever) live in dedicated modules and import
their heavy dependencies lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .application import sha256_text


# ── result / hit value objects ────────────────────────────────────────────


@dataclass(frozen=True)
class CompletionResult:
    """Output of a model completion.

    ``text`` is the only mandatory field. The remaining fields are optional
    fine-grained telemetry that a real backend can supply and that the server
    forwards verbatim to :meth:`ApplicationForensicLogger.log_completion`.
    Total latency is measured by the server as wall-clock time around the call,
    so it is intentionally *not* part of this object.
    """

    text: str
    response_token_count: int | None = None
    latency_first_token_ms: int | None = None
    finish_reason: str | None = "stop"
    # Score-exposure (populated only when the request asks for logprobs).
    token_logprobs: list[float] | None = None
    top_k_first_token: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class SequenceScore:
    """Teacher-forced per-token scoring of a candidate sequence (for score-based
    MIA: LOSS/Min-K%/Min-K%++/zlib). ``token_logprobs[i]`` is the log-prob of the
    actual (i+1)-th token given its prefix; ``token_logprob_mean/std`` are the
    mean/std of the log-prob distribution at that position (needed by Min-K%++).
    """

    text: str
    n_tokens: int
    token_logprobs: list[float]
    token_logprob_mean: list[float] | None = None
    token_logprob_std: list[float] | None = None


@dataclass(frozen=True)
class RetrievedHit:
    """A single retrieval hit.

    The forensic-safe fields (``document_id``, ``chunk_id``, ``rank``,
    ``similarity_score``, ``chunk_hash``) are exactly those required by
    :meth:`ApplicationForensicLogger.log_rag_retrieval`. ``content`` holds the
    raw chunk text used by the backend to build the augmented prompt and is
    **never** logged: :meth:`forensic_dict` drops it.
    """

    document_id: str
    chunk_id: str
    rank: int
    similarity_score: float
    chunk_hash: str
    content: str | None = None  # raw text for generation only — never logged

    def forensic_dict(self) -> dict[str, Any]:
        """Return only the forensic-safe fields (no raw content)."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "similarity_score": self.similarity_score,
            "chunk_hash": self.chunk_hash,
        }


# ── protocols ──────────────────────────────────────────────────────────────


@runtime_checkable
class ModelBackend(Protocol):
    """A local LLM backend.

    Implementations expose model identity (``model_id`` / ``model_revision`` /
    ``model_hash``) so the server can record it in the manifest and in every
    prompt event, and two generation methods: plain completion and
    context-augmented (RAG) completion.
    """

    model_id: str
    model_revision: str | None
    model_hash: str | None

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult: ...

    def complete_with_context(
        self,
        prompt: str,
        *,
        retrieval_query: str,
        hits: Sequence[RetrievedHit],
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult: ...


@runtime_checkable
class Retriever(Protocol):
    """A retrieval backend over a controlled corpus."""

    def retrieve(
        self,
        retrieval_query: str,
        *,
        top_k: int,
        vector_store_id: str,
        embedding_model_id: str,
    ) -> list[RetrievedHit]: ...


# ── deterministic mock implementations (behaviour-preserving) ───────────────
#
# These reproduce the M2 mock functions exactly. They are kept here (rather than
# in server.py) so that both the server and the traffic simulators share a
# single source of mock behaviour. server.py re-exports the three functions for
# backward compatibility.


def _deterministic_mock_completion(prompt: str, *, max_tokens: int) -> str:
    """Generate a deterministic mock response dependent on the input prompt."""
    digest = sha256_text(f"{prompt}|max_tokens={max_tokens}")[:16]
    return (
        f"MOCK_RESPONSE[{digest}]: deterministic local completion generated "
        f"for prompt_length={len(prompt)} and max_tokens={max_tokens}."
    )


def _deterministic_mock_retriever_hits(
    retrieval_query: str,
    *,
    top_k: int,
    vector_store_id: str,
) -> list[dict[str, Any]]:
    """Generate deterministic mock retriever hits without raw chunk contents."""
    hits: list[dict[str, Any]] = []
    for rank in range(1, top_k + 1):
        seed = f"{retrieval_query}|{vector_store_id}|rank={rank}"
        digest = sha256_text(seed)
        jitter = int(digest[:4], 16) / 0xFFFF
        similarity = max(0.0, 0.98 - ((rank - 1) * 0.07) - (jitter * 0.01))
        hits.append(
            {
                "document_id": f"mock-doc-{digest[:8]}",
                "chunk_id": f"mock-chunk-{digest[8:16]}",
                "rank": rank,
                "similarity_score": round(similarity, 6),
                "chunk_hash": sha256_text(f"{seed}|chunk"),
            }
        )
    return hits


def _deterministic_mock_logprobs(
    text: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Produce deterministic, plausible logprobs for the mock backend.

    Derives a stable pseudo-logprob per generated token from a hash of the
    text, in a realistic negative range, plus a top-3 candidate list for the
    first token. Used only when ``expose_logprobs=True``, so the default
    completion path is byte-identical to before.
    """
    tokens = text.split()
    logprobs: list[float] = []
    for index, tok in enumerate(tokens):
        h = int(sha256_text(f"{tok}|{index}")[:6], 16) / 0xFFFFFF
        # map into roughly [-6.0, -0.1]
        logprobs.append(round(-0.1 - h * 5.9, 6))
    if not logprobs:
        logprobs = [round(-0.5, 6)]
    first = logprobs[0]
    top_k_first_token = [
        {"token": tokens[0] if tokens else "<bos>", "logprob": first},
        {"token": "<alt1>", "logprob": round(first - 0.7, 6)},
        {"token": "<alt2>", "logprob": round(first - 1.4, 6)},
    ]
    return logprobs, top_k_first_token


def _deterministic_mock_rag_completion(
    prompt: str,
    retrieval_query: str,
    retriever_hits: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> str:
    """Generate a deterministic mock RAG answer dependent on prompt and hits."""
    hit_fingerprint = ",".join(hit["chunk_hash"][:12] for hit in retriever_hits)
    digest = sha256_text(
        f"{prompt}|{retrieval_query}|{hit_fingerprint}|max_tokens={max_tokens}"
    )[:16]
    return (
        f"MOCK_RAG_RESPONSE[{digest}]: deterministic local RAG completion "
        f"generated with {len(retriever_hits)} retrieval hits."
    )


class DeterministicMockBackend:
    """Default backend reproducing the M2 deterministic mock model."""

    def __init__(
        self,
        *,
        model_id: str = "deterministic-mock-model",
        model_revision: str | None = "m2.2",
        model_hash: str | None = "deterministic-mock-model",
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_hash = model_hash

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        # The mock is deterministic and never "leaks": it accepts ``system_prompt``
        # to satisfy the level-1 injection contract but intentionally ignores it
        # for text generation, so the byte-stable mock output (and the apparatus
        # tests) are unaffected. Real leakage behaviour is exercised only by the
        # transformers backend on the real model (post-launch).
        text = _deterministic_mock_completion(prompt, max_tokens=max_tokens)
        logprobs, top_k = _deterministic_mock_logprobs(text) if expose_logprobs else (None, None)
        return CompletionResult(
            text=text,
            finish_reason="stop",
            token_logprobs=logprobs,
            top_k_first_token=top_k,
        )

    def complete_with_context(
        self,
        prompt: str,
        *,
        retrieval_query: str,
        hits: Sequence[RetrievedHit],
        max_tokens: int,
        sampling_params: Mapping[str, Any],
        expose_logprobs: bool = False,
        system_prompt: str | None = None,
    ) -> CompletionResult:
        hit_dicts = [hit.forensic_dict() for hit in hits]
        text = _deterministic_mock_rag_completion(
            prompt, retrieval_query, hit_dicts, max_tokens=max_tokens
        )
        logprobs, top_k = _deterministic_mock_logprobs(text) if expose_logprobs else (None, None)
        return CompletionResult(
            text=text,
            finish_reason="stop",
            token_logprobs=logprobs,
            top_k_first_token=top_k,
        )

    def score_sequence(self, text: str) -> SequenceScore:
        """Deterministic, plausible per-token scoring of *text* (no real model).

        The mock never memorises, so the score carries no membership signal
        (member and non-member look alike) — real signal needs the transformers
        backend. Used by the apparatus tests so they need no GPU.
        """
        toks = text.split() or [text[:8]]
        lps: list[float] = []
        mus: list[float] = []
        sds: list[float] = []
        for i, t in enumerate(toks):
            h = int(sha256_text(f"{self.model_hash}|{text[:64]}|{i}|{t}")[:8], 16) / 0xFFFFFFFF
            lps.append(-0.5 - 4.0 * h)        # in [-4.5, -0.5]
            mus.append(-3.0 - h)              # distribution mean (more negative)
            sds.append(1.0 + 0.5 * h)
        return SequenceScore(
            text=text, n_tokens=len(toks), token_logprobs=lps,
            token_logprob_mean=mus, token_logprob_std=sds,
        )


class DeterministicMockRetriever:
    """Default retriever reproducing the M2 deterministic mock hits."""

    def retrieve(
        self,
        retrieval_query: str,
        *,
        top_k: int,
        vector_store_id: str,
        embedding_model_id: str,
    ) -> list[RetrievedHit]:
        raw = _deterministic_mock_retriever_hits(
            retrieval_query, top_k=top_k, vector_store_id=vector_store_id
        )
        return [
            RetrievedHit(
                document_id=hit["document_id"],
                chunk_id=hit["chunk_id"],
                rank=hit["rank"],
                similarity_score=hit["similarity_score"],
                chunk_hash=hit["chunk_hash"],
                content=None,  # mock has no raw content to generate from
            )
            for hit in raw
        ]


__all__ = [
    "CompletionResult",
    "SequenceScore",
    "RetrievedHit",
    "ModelBackend",
    "Retriever",
    "DeterministicMockBackend",
    "DeterministicMockRetriever",
]
