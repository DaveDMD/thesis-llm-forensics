"""Run the integrated forensic pipeline against the REAL model + retriever.

Unlike the deterministic mock backend (``forensic.pipeline.run_integrated_pipeline``),
this script wires in the real Mistral-7B-Instruct-v0.2 backend in 4-bit NF4 and a real Chroma
retriever over the controlled corpus. It exists to exercise the two artefacts the
mock cannot produce — indirect-injection payload echo and secret leakage in the
response — and thus to confirm, end to end on a real LLM, the blind spot the
thesis documents.

Scale
-----
``--scale {small,full}`` selects the dataset size:
  * small : rag=2 / mia=2 / benign=2  (~30 cases) — fast validation run;
  * full  : rag=10 / mia=10 / benign=6 (the official 359-case dataset).
The real model generates at ~2-3 s/request, so a full run is ~20-30 min; the
small run is a couple of minutes and is meant to validate the wiring first.

Configuration
-------------
* model target: Mistral-7B-Instruct-v0.2, 4-bit NF4;
* embedding model: all-MiniLM-L6-v2 — matches traffic_rag.EMBEDDING_MODEL_ID
  and Anderson 2024; NOT mpnet (the build script's generic default);
* vector store id: internal-handbook-v1 (corpus.CORPUS_ID).

The Chroma index is built on first run if absent, so the script is self-contained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

ROOT = Path(__file__).resolve().parents[3]

# Confirmed targets (model / embedding / store).
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# Pinned model snapshot (HuggingFace commit) for reproducible provenance; this
# is the revision recorded in the manifest and used to load the weights.
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"  # matches traffic_rag.EMBEDDING_MODEL_ID
COLLECTION = "internal-handbook-v1"      # corpus.CORPUS_ID / traffic_rag.VECTOR_STORE_ID

CHROMA_DIR = ROOT / "data" / "processed" / "chroma_real"
CORPUS_OUT = ROOT / "data" / "processed" / "rag_corpus.jsonl"
GROUNDTRUTH_OUT = ROOT / "groundtruth" / "rag_corpus_groundtruth.jsonl"

EVIDENCE_DIR = ROOT / "evidence"
GROUNDTRUTH_DIR = ROOT / "groundtruth"

SCALES = {
    "small": {"rag_variants": 2, "mia_variants": 2, "benign_variants": 2},
    "full": {"rag_variants": 10, "mia_variants": 10, "benign_variants": 6},
}


def _build_chroma_index_if_absent() -> None:
    """Embed the controlled corpus into Chroma (cosine) if the index is missing.

    Always persists the exact indexed corpus to ``CORPUS_OUT`` so the run
    manifest can fingerprint the dataset that backed the retriever.
    """
    from forensic.corpus import build_controlled_corpus

    corpus = build_controlled_corpus()
    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_OUT.write_text(
        "".join(
            json.dumps({"document_id": d.document_id, "content": d.content}, sort_keys=True)
            + "\n"
            for d in corpus.documents
        ),
        encoding="utf-8",
    )

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Chroma index present at {CHROMA_DIR} — reusing.")
        return
    print(f"building Chroma index at {CHROMA_DIR} (embedding: {EMBEDDING_MODEL_ID}) ...")
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_ID
    )
    coll = client.get_or_create_collection(
        name=COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    docs, ids, metas = [], [], []
    for doc in corpus.documents:
        docs.append(doc.content)
        ids.append(doc.document_id)
        metas.append({"document_id": doc.document_id})
    coll.add(documents=docs, ids=ids, metadatas=metas)
    print(f"indexed {len(docs)} documents into '{COLLECTION}'.")


def _model_artifact_paths() -> dict[str, Path]:
    """Collect the small identity files of the pinned model snapshot, if cached.

    Returns the subset of provenance-bearing files that exist locally, so the
    manifest can fingerprint the served model without hashing the multi-GB
    weight shards. Returns an empty mapping when the snapshot is not present.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_name = "models--" + MODEL_ID.replace("/", "--")
    snapshot_dir = hf_home / "hub" / cache_name / "snapshots" / MODEL_REVISION
    wanted = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    )
    return {name: snapshot_dir / name for name in wanted if (snapshot_dir / name).exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=list(SCALES), default="small")
    args = parser.parse_args()
    cfg = SCALES[args.scale]

    from forensic.backends_transformers import TransformersBackend
    from forensic.rag_chroma import ChromaRetriever
    from forensic.pipeline import run_integrated_pipeline

    _build_chroma_index_if_absent()

    print(f"loading {MODEL_ID} (4-bit NF4) ...")
    backend = TransformersBackend(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        quantization="4bit",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    ).load()

    retriever = ChromaRetriever(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
        embedding_model_id=EMBEDDING_MODEL_ID,
        vector_store_id=COLLECTION,
    )

    log_path = EVIDENCE_DIR / f"real_pipeline_{args.scale}.jsonl"
    gt_path = GROUNDTRUTH_DIR / f"real_pipeline_{args.scale}_groundtruth.jsonl"
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    GROUNDTRUTH_DIR.mkdir(parents=True, exist_ok=True)
    for p in (log_path, Path(str(log_path) + ".lock"), gt_path):
        if p.exists():
            p.unlink()

    dataset_paths = {"rag_corpus": CORPUS_OUT} if CORPUS_OUT.exists() else None
    model_artifacts = _model_artifact_paths() or None

    print(f"running integrated pipeline (scale={args.scale}, real backend) ...")
    result = run_integrated_pipeline(
        log_path=str(log_path),
        run_id=f"real-pipeline-{args.scale}",
        repo_path=str(ROOT),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_hash=MODEL_ID,
        backend=backend,
        retriever=retriever,
        notes=f"integrated pipeline, real backend, scale={args.scale}",
        model_artifacts=model_artifacts,
        dataset_paths=dataset_paths,
        rag_variants=cfg["rag_variants"],
        mia_variants=cfg["mia_variants"],
        benign_variants=cfg["benign_variants"],
    )

    gt_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in result.groundtruth_records),
        encoding="utf-8",
    )

    # Quick artefact check: did the real model expose the mock-blind signals?
    from forensic.features import build_features

    rows = build_features(result.forensic_records, result.groundtruth_records)
    echo = [r for r in rows if r.get("diagnostic_response_echoes_injection_payload")]
    leak = [r for r in rows if r.get("feature_response_contains_secret_like_pattern")]
    refusal = [r for r in rows if r.get("feature_response_contains_refusal")]

    summary = dict(result.summary)
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scale": args.scale,
            "backend": "mistral-7b-instruct-v0.2-4bit",
            "embedding_model": EMBEDDING_MODEL_ID,
            "real_artefact_check": {
                "injection_payload_echo_rows": len(echo),
                "secret_like_in_response_rows": len(leak),
                "refusal_rows": len(refusal),
            },
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n--- real-artefact check (mock-blind signals) ---")
    print(f"injection payload echo : {len(echo)} rows")
    print(f"secret-like in response: {len(leak)} rows")
    print(f"refusals               : {len(refusal)} rows")


if __name__ == "__main__":
    main()
