"""Real retrieval backend over a Chroma vector store.

Implements :class:`forensic.backends.Retriever` by querying an existing Chroma
collection. ``chromadb`` is imported lazily inside :meth:`ChromaRetriever.retrieve`,
so importing this module does not require the vector-store stack.

Two-tier discipline
--------------------
Each :class:`RetrievedHit` carries the raw chunk ``content`` (needed by the LLM
backend to build the augmented prompt) *and* a ``chunk_hash`` over that content.
The server logs only :meth:`RetrievedHit.forensic_dict`, which excludes the raw
content; the content never reaches the forensic stream. The query→hit mapping is
factored into the pure function :func:`map_query_result_to_hits` so it can be
unit-tested without a live Chroma instance.

Scope note
----------
This module is the *query* side. Building the controlled corpus (ingestion of
documents with known secrets and one poisoned document for indirect prompt
injection) into a Chroma collection is a separate deliverable and is not part of
this module.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .application import sha256_text
from .backends import RetrievedHit


def cosine_distance_to_similarity(distance: float) -> float:
    """Convert a cosine *distance* to a similarity in ``[0, 1]``.

    Assumes the Chroma collection uses cosine space (``hnsw:space = "cosine"``),
    where distance = 1 - cosine_similarity. The result is clamped to ``[0, 1]``.
    Pass a different callable to :class:`ChromaRetriever` for other metrics.
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


def map_query_result_to_hits(
    *,
    ids: Sequence[str],
    documents: Sequence[str | None],
    distances: Sequence[float],
    metadatas: Sequence[Mapping[str, Any] | None],
    top_k: int,
    distance_to_similarity: Callable[[float], float] = cosine_distance_to_similarity,
) -> list[RetrievedHit]:
    """Map one Chroma query result (single-query slices) to ``RetrievedHit`` list.

    ``document_id`` is taken from metadata (``document_id`` then ``source``),
    falling back to the chunk id. ``chunk_hash`` is computed over the raw chunk
    content; the content is retained on the hit for generation but is never
    logged.
    """
    hits: list[RetrievedHit] = []
    for rank, (chunk_id, document, distance, metadata) in enumerate(
        zip(ids, documents, distances, metadatas), start=1
    ):
        if rank > top_k:
            break
        meta = dict(metadata or {})
        document_id = str(meta.get("document_id") or meta.get("source") or chunk_id)
        content = document if document is not None else ""
        hits.append(
            RetrievedHit(
                document_id=document_id,
                chunk_id=str(chunk_id),
                rank=rank,
                similarity_score=round(distance_to_similarity(distance), 6),
                chunk_hash=sha256_text(content),
                content=content,
            )
        )
    return hits


class ChromaRetriever:
    """A :class:`Retriever` over an existing persistent Chroma collection."""

    def __init__(
        self,
        *,
        persist_directory: str,
        collection_name: str,
        embedding_model_id: str,
        vector_store_id: str | None = None,
        embedding_function: Any | None = None,
        distance_to_similarity: Callable[[float], float] = cosine_distance_to_similarity,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_id = embedding_model_id
        # Stable identity reported for logging when the caller does not override.
        self.vector_store_id = vector_store_id or f"chroma:{collection_name}"
        self.embedding_function = embedding_function
        self.distance_to_similarity = distance_to_similarity
        self._client = None
        self._collection = None

    def load(self) -> "ChromaRetriever":
        """Lazily import ``chromadb`` and open the persistent collection."""
        import chromadb  # noqa: WPS433 (intentional lazy import)

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        kwargs: dict[str, Any] = {"name": self.collection_name}
        if self.embedding_function is not None:
            kwargs["embedding_function"] = self.embedding_function
        self._collection = self._client.get_collection(**kwargs)
        return self

    def retrieve(
        self,
        retrieval_query: str,
        *,
        top_k: int,
        vector_store_id: str,
        embedding_model_id: str,
    ) -> list[RetrievedHit]:
        if self._collection is None:
            self.load()
        result = self._collection.query(
            query_texts=[retrieval_query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        # Chroma returns one nested list per query; we issue a single query.
        return map_query_result_to_hits(
            ids=result["ids"][0],
            documents=result["documents"][0],
            distances=result["distances"][0],
            metadatas=result["metadatas"][0],
            top_k=top_k,
            distance_to_similarity=self.distance_to_similarity,
        )


__all__ = [
    "ChromaRetriever",
    "map_query_result_to_hits",
    "cosine_distance_to_similarity",
]
