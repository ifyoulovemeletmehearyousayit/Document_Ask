from __future__ import annotations
import logging
from functools import lru_cache
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "documents"
BATCH_SIZE = 64

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)

@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    client = chromadb.HttpClient(host="chromadb", port=8000)
    return client.get_or_create_collection(name=COLLECTION_NAME)

def embed_and_store(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    model = _get_model()
    collection = _get_collection()
    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}_{c['chunk_index']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(vecs.tolist())

    collection.upsert(ids=ids, embeddings=all_embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)

def query_similar(question: str, n_results: int = 5, source_filter: Optional[str] = None) -> list[dict]:
    model = _get_model()
    collection = _get_collection()
    vector = model.encode([question], normalize_embeddings=True).tolist()
    where = {"source": source_filter} if source_filter else None
    try:
        results = collection.query(query_embeddings=vector, n_results=n_results, where=where)
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"text": doc, "source": meta.get("source", ""), "chunk_index": meta.get("chunk_index", -1)} for doc, meta in zip(docs, metas)]

def get_all_chunks(source_filter: Optional[str] = None) -> list[dict]:
    collection = _get_collection()
    kwargs = {"where": {"source": source_filter}} if source_filter else {}
    results = collection.get(**kwargs)
    return [{"text": doc, "source": meta.get("source", ""), "chunk_index": meta.get("chunk_index", -1)} for doc, meta in zip(results["documents"], results["metadatas"])]

def list_sources() -> list[str]:
    collection = _get_collection()
    results = collection.get(include=["metadatas"])
    sources = {m.get("source", "") for m in results["metadatas"] if m.get("source")}
    return sorted(sources)

def delete_source(filename: str) -> int:
    collection = _get_collection()
    results = collection.get(where={"source": filename}, include=["metadatas"])
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)