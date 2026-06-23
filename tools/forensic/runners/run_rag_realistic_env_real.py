"""Build the REPRODUCIBLE, SEPARATE environment for the RAG campaign.

This is a self-contained operator script that constructs a dedicated, deterministic
RAG environment so the campaign can be regenerated exactly and kept apart from the
baseline collection (``internal-handbook-v1`` / ``chroma_real``), which stays untouched
and reproducible.

What it does (CPU only — no LLM is loaded here)
-----------------------------------------------
1. builds the controlled corpus with ``n_distractors`` benign handbook chunks
   (deterministic, seeded) so retrieval is competitive;
2. writes the corpus JSONL and the ground-truth JSONL to dedicated paths;
3. ingests the corpus into a SEPARATE persistent Chroma collection
   (``internal-handbook-realistic`` at ``data/processed/chroma_rag_realistic``),
   in cosine space, with the MiniLM embedder (NOT mpnet);
4. runs a retrieval-competitiveness SMOKE: for on-topic queries the matching
   secret/poisoned document should enter top-k; for off-topic benign queries the
   sensitive documents should stay dormant amid the distractors. This is the
   empirical baseline for backdoor gating (gating measured, never assumed);
5. writes a reproducibility MANIFEST (params, counts, content hashes, library
   versions, smoke metrics) so the environment is auditable and regenerable.

Anti-circularity note
---------------------
The smoke reads the sensitive document ids from the ground truth ONLY to *score*
the retrieval competitiveness offline; the labels never enter the forensic stream
or any detector feature. This is an environment-validation utility, not a detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

ROOT = Path(__file__).resolve().parents[3]

# Target generator recorded for provenance (NOT loaded by this CPU script).
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"

# Dedicated, SEPARATE environment for the RAG campaign (baseline untouched).
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
COLLECTION = "internal-handbook-realistic"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma_rag_realistic"
CORPUS_OUT = ROOT / "data" / "processed" / "rag_realistic_corpus.jsonl"
GROUNDTRUTH_OUT = ROOT / "groundtruth" / "rag_realistic_corpus_groundtruth.jsonl"
MANIFEST_OUT = ROOT / "results" / "rag_realistic_env" / "manifest.json"

# SEPARATE backdoor env (the baseline collection stays untouched).
COLLECTION_BD = "internal-handbook-realistic-bd"
CHROMA_DIR_BD = ROOT / "data" / "processed" / "chroma_rag_realistic_bd"
CORPUS_OUT_BD = ROOT / "data" / "processed" / "rag_realistic_bd_corpus.jsonl"
GROUNDTRUTH_OUT_BD = ROOT / "groundtruth" / "rag_realistic_bd_corpus_groundtruth.jsonl"
MANIFEST_OUT_BD = ROOT / "results" / "rag_realistic_env" / "manifest_bd.json"

DEFAULT_N_DISTRACTORS = 300
DEFAULT_SEED = 20260622
TOP_K = 3

# Off-topic benign queries (recon register): the sensitive docs should NOT surface.
_OFF_TOPIC_QUERIES = (
    "incident response procedure",
    "password rotation policy",
    "data backup routine",
    "employee onboarding process",
    "change management workflow",
    "vpn usage guidance",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_corpus_jsonl(corpus, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(
        json.dumps({"document_id": d.document_id, "content": d.content}, sort_keys=True) + "\n"
        for d in corpus.documents
    )
    path.write_text(blob, encoding="utf-8")
    return _sha256_text(blob)


def _ingest(corpus, *, persist_dir: Path, collection: str, embedding_model: str) -> int:
    import chromadb
    from chromadb.utils import embedding_functions

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection)   # idempotent rebuild
    except Exception:  # noqa: BLE001 — absent collection is fine
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    coll = client.create_collection(
        name=collection, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    coll.add(
        ids=[d.document_id for d in corpus.documents],
        documents=[d.content for d in corpus.documents],
        metadatas=[{"document_id": d.document_id} for d in corpus.documents],
    )
    return coll.count()


def _competitiveness_smoke(corpus, *, persist_dir: Path, collection: str) -> dict:
    """Measure on-topic retrievability vs off-topic dormancy of sensitive docs.

    When a trigger-gated backdoor document is present, also verify that the
    TRIGGER query retrieves it while ordinary benign queries leave it DORMANT —
    the empirical, held-out gating measurement (never assumed)."""
    from forensic.corpus import PLANTED_SECRETS
    from forensic.rag_chroma import ChromaRetriever

    from chromadb.utils import embedding_functions

    sensitive_ids = {
        r.document_id for r in corpus.groundtruth if r.is_secret_bearing or r.is_poisoned
    }
    poisoned_id = next(
        r.document_id for r in corpus.groundtruth
        if r.is_poisoned and r.injection_type == "indirect_prompt_injection"
    )
    backdoor = next(
        (r for r in corpus.groundtruth if r.injection_type == "backdoor_trigger"), None
    )

    # Use the SAME sentence-transformers MiniLM for queries as for ingest, so the
    # query and document embeddings live in one space AND stay offline (Chroma's
    # default ONNX embedder would otherwise be downloaded and silently used).
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_ID)
    retriever = ChromaRetriever(
        persist_directory=str(persist_dir),
        collection_name=collection,
        embedding_model_id=EMBEDDING_MODEL_ID,
        vector_store_id=collection,
        embedding_function=ef,
    ).load()

    def _ids(query: str) -> list[str]:
        hits = retriever.retrieve(
            query, top_k=TOP_K, vector_store_id=collection, embedding_model_id=EMBEDDING_MODEL_ID
        )
        return [h.document_id for h in hits]

    # on-topic: querying a secret's own context should retrieve its doc
    on_topic = []
    for s in PLANTED_SECRETS:
        ids = _ids(s["context"])
        target = f"secret-doc-{s['secret_id']}"
        on_topic.append({"query": s["context"], "target": target, "topk": ids, "hit": target in ids})
    ids = _ids("service runbook maintenance windows")
    on_topic.append(
        {"query": "service runbook maintenance windows", "target": poisoned_id,
         "topk": ids, "hit": poisoned_id in ids}
    )

    # off-topic benign: sensitive docs (incl. the backdoor) should stay dormant
    off_topic = []
    for q in _OFF_TOPIC_QUERIES:
        ids = _ids(q)
        surfaced = sorted(set(ids) & sensitive_ids)
        off_topic.append({"query": q, "topk": ids, "sensitive_surfaced": surfaced,
                          "dormant": len(surfaced) == 0})

    out = {
        "top_k": TOP_K,
        "on_topic_hit_rate": round(sum(c["hit"] for c in on_topic) / len(on_topic), 3),
        "off_topic_dormancy_rate": round(sum(c["dormant"] for c in off_topic) / len(off_topic), 3),
        "on_topic": on_topic,
        "off_topic": off_topic,
    }

    # backdoor gating: trigger retrieves it; benign queries leave it dormant
    if backdoor is not None:
        trig_ids = _ids(backdoor.trigger_phrase)
        bd_dormant = all(backdoor.document_id not in c["topk"] for c in off_topic)
        out["backdoor_gating"] = {
            "trigger_phrase": backdoor.trigger_phrase,
            "document_id": backdoor.document_id,
            "trigger_topk": trig_ids,
            "trigger_retrieves_backdoor": backdoor.document_id in trig_ids,
            "dormant_on_benign": bd_dormant,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-distractors", type=int, default=DEFAULT_N_DISTRACTORS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-backdoor", action="store_true",
                        help="add the trigger-gated backdoor doc into a SEPARATE collection")
    parser.add_argument("--skip-smoke", action="store_true", help="ingest only, no smoke")
    args = parser.parse_args(argv)

    # The backdoor lands in a SEPARATE collection/persist-dir (the baseline env stays untouched).
    if args.include_backdoor:
        collection, persist_dir = COLLECTION_BD, CHROMA_DIR_BD
        corpus_out, gt_out, manifest_out = CORPUS_OUT_BD, GROUNDTRUTH_OUT_BD, MANIFEST_OUT_BD
    else:
        collection, persist_dir = COLLECTION, CHROMA_DIR
        corpus_out, gt_out, manifest_out = CORPUS_OUT, GROUNDTRUTH_OUT, MANIFEST_OUT

    from forensic.corpus import build_controlled_corpus

    corpus = build_controlled_corpus(
        n_distractors=args.n_distractors, seed=args.seed, include_backdoor=args.include_backdoor
    )
    n_secret = sum(1 for r in corpus.groundtruth if r.is_secret_bearing)
    n_poison = sum(1 for r in corpus.groundtruth if r.is_poisoned)
    n_backdoor = sum(1 for r in corpus.groundtruth if r.injection_type == "backdoor_trigger")
    n_distract = sum(1 for d in corpus.documents if d.document_id.startswith("handbook-note-"))

    corpus_hash = _write_corpus_jsonl(corpus, corpus_out)
    gt_path = corpus.write_groundtruth_jsonl(gt_out)
    gt_hash = _sha256_text(Path(gt_path).read_text(encoding="utf-8"))

    print(f"corpus: documents={len(corpus.documents)} secret={n_secret} poisoned={n_poison} "
          f"backdoor={n_backdoor} distractors={n_distract}")
    print(f"ingesting into '{collection}' at {persist_dir} (embedder={EMBEDDING_MODEL_ID}) ...")
    ingested = _ingest(
        corpus, persist_dir=persist_dir, collection=collection, embedding_model=EMBEDDING_MODEL_ID
    )
    print(f"ingested={ingested}")

    smoke = None
    if not args.skip_smoke:
        smoke = _competitiveness_smoke(corpus, persist_dir=persist_dir, collection=collection)
        print(f"smoke: on_topic_hit_rate={smoke['on_topic_hit_rate']} "
              f"off_topic_dormancy_rate={smoke['off_topic_dormancy_rate']}")
        if "backdoor_gating" in smoke:
            bg = smoke["backdoor_gating"]
            print(f"backdoor gating: trigger_retrieves={bg['trigger_retrieves_backdoor']} "
                  f"dormant_on_benign={bg['dormant_on_benign']}")

    import chromadb
    import sentence_transformers

    manifest = {
        "campaign": "rag-b3bis-realistic" + ("-bd" if args.include_backdoor else ""),
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "target_model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                         "quantization": "4bit-nf4"},
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "vector_store": {"collection": collection, "persist_dir": str(persist_dir.relative_to(ROOT)),
                         "space": "cosine", "ingested": ingested},
        "corpus": {
            "n_distractors": args.n_distractors, "seed": args.seed,
            "include_backdoor": args.include_backdoor,
            "documents": len(corpus.documents), "secret_bearing": n_secret,
            "poisoned": n_poison, "backdoor": n_backdoor, "distractors": n_distract,
            "corpus_jsonl": str(corpus_out.relative_to(ROOT)), "corpus_sha256": corpus_hash,
            "groundtruth_jsonl": str(Path(gt_path).relative_to(ROOT)), "groundtruth_sha256": gt_hash,
        },
        "library_versions": {"chromadb": chromadb.__version__,
                             "sentence_transformers": sentence_transformers.__version__},
        "competitiveness_smoke": smoke,
        "reproduce": (
            "docker compose run --rm -e HF_HUB_OFFLINE=1 "
            "-e PYTHONPATH=/workspace/tools:/workspace/tools/forensic thesis "
            f"python3 tools/forensic/runners/run_rag_realistic_env_real.py "
            f"--n-distractors {args.n_distractors} --seed {args.seed}"
            + (" --include-backdoor" if args.include_backdoor else "")
        ),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
