"""Build the controlled RAG corpus and (optionally) ingest it into Chroma.

This is a runnable operator script, not a library module. It:
  1. deterministically generates the controlled corpus and its ground truth
     (``forensic.corpus.build_controlled_corpus``);
  2. writes the corpus JSONL (retrievable content) and the ground-truth JSONL
     (evaluation labels, kept out of the forensic stream);
  3. optionally embeds the corpus into a persistent Chroma collection using a
     sentence-transformers embedding function, in cosine space, so that the
     ``ChromaRetriever`` similarity scores are well defined.

Heavy dependencies (``chromadb``, ``sentence_transformers``) are imported only
when ``--ingest`` is passed, so generating the data files works without the
vector-store stack installed.

Typical use inside the container::

    python tools/forensic/runners/build_rag_corpus.py \
        --corpus-out data/processed/rag_corpus.jsonl \
        --groundtruth-out groundtruth/rag_corpus_groundtruth.jsonl \
        --ingest \
        --persist-dir data/processed/chroma \
        --collection internal-handbook-v1 \
        --embedding-model sentence-transformers/all-mpnet-base-v2

The embedding model and the target LLM remain configuration choices; this
script hard-codes none of them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly: ensure the package is importable.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from forensic.corpus import CORPUS_ID, build_controlled_corpus  # noqa: E402


def _ingest_into_chroma(
    corpus,
    *,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
) -> int:
    """Embed the corpus into a persistent Chroma collection (cosine space)."""
    import chromadb  # lazy
    from chromadb.utils import embedding_functions  # lazy

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    client = chromadb.PersistentClient(path=persist_dir)
    # Recreate the collection to keep ingestion idempotent and reproducible.
    try:
        client.delete_collection(collection_name)
    except Exception:  # noqa: BLE001 — absent collection is fine
        pass
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "corpus_id": CORPUS_ID},
    )
    collection.add(
        ids=[doc.chunk_id for doc in corpus.documents],
        documents=[doc.content for doc in corpus.documents],
        metadatas=[
            {**doc.metadata, "document_id": doc.document_id} for doc in corpus.documents
        ],
    )
    return collection.count()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-out", required=True, help="output path for the corpus JSONL")
    parser.add_argument(
        "--groundtruth-out", required=True, help="output path for the ground-truth JSONL"
    )
    parser.add_argument("--ingest", action="store_true", help="also ingest into Chroma")
    parser.add_argument("--persist-dir", default="data/processed/chroma")
    parser.add_argument("--collection", default=CORPUS_ID)
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model id (MiniLM, matches traffic_rag)",
    )
    parser.add_argument(
        "--n-distractors",
        type=int,
        default=0,
        help="number of benign handbook distractor chunks for competitive retrieval",
    )
    parser.add_argument(
        "--seed", type=int, default=20260622, help="deterministic distractor seed"
    )
    args = parser.parse_args(argv)

    corpus = build_controlled_corpus(n_distractors=args.n_distractors, seed=args.seed)
    corpus_path = corpus.write_corpus_jsonl(args.corpus_out)
    gt_path = corpus.write_groundtruth_jsonl(args.groundtruth_out)

    n_docs = len(corpus.documents)
    n_secret = sum(1 for r in corpus.groundtruth if r.is_secret_bearing)
    n_poisoned = sum(1 for r in corpus.groundtruth if r.is_poisoned)
    n_distractors = sum(
        1 for d in corpus.documents if d.document_id.startswith("handbook-note-")
    )
    print(f"corpus_id={CORPUS_ID}")
    print(
        f"documents={n_docs} secret_bearing={n_secret} poisoned={n_poisoned} "
        f"distractors={n_distractors}"
    )
    print(f"corpus_jsonl={corpus_path}")
    print(f"groundtruth_jsonl={gt_path}")

    if args.ingest:
        count = _ingest_into_chroma(
            corpus,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
        )
        print(f"chroma_collection={args.collection} ingested={count} persist_dir={args.persist_dir}")
    else:
        print("ingest=skipped (pass --ingest to embed into Chroma)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
