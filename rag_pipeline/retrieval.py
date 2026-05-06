"""First dense retrieval helper over Qdrant."""

from __future__ import annotations

from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embeddings import Embedder
from rag_pipeline.qdrant_store import QdrantStore


def search_chunks(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 10,
    config: WorkerConfig | None = None,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and return normalized chunk search results."""
    cfg = config or WorkerConfig.from_env()
    active_embedder = embedder or Embedder(
        provider=cfg.embedding_provider,
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key,
        gemini_api_key=cfg.gemini_api_key,
        batch_size=cfg.embedding_batch_size,
    )
    active_store = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=cfg.qdrant_collection,
    )
    vector = active_embedder.embed([query])[0]
    hits = active_store.search_chunks(vector, user_id, source_types, top_k)
    return [_normalize_hit(hit) for hit in hits]


def _normalize_hit(hit: Any) -> dict[str, Any]:
    payload = getattr(hit, "payload", None) or hit.get("payload", {})
    score = getattr(hit, "score", None)
    if score is None and isinstance(hit, dict):
        score = hit.get("score")
    return {
        "chunk_id": payload.get("chunk_id"),
        "text": payload.get("text"),
        "score": score,
        "source_type": payload.get("source_type"),
        "source_id": payload.get("source_id"),
        "page_index": payload.get("page_index"),
        "title": payload.get("title"),
        "heading": payload.get("heading"),
        "metadata": payload.get("metadata") or {},
    }
