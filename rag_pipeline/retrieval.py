"""First dense retrieval helper over Qdrant."""

from __future__ import annotations

from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embeddings import Embedder
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.source_types import contains_chat_memory
from rag_pipeline.sparse_embeddings import SparseEmbedder


def search_chunks(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 10,
    pdf_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    config: WorkerConfig | None = None,
    embedder: Embedder | None = None,
    store: QdrantStore | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and return normalized chunk search results."""
    if contains_chat_memory(source_types) and not source_ids:
        return []
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
    kwargs = {"pdf_ids": pdf_ids}
    if source_ids is not None:
        kwargs["source_ids"] = source_ids
    hits = active_store.search_chunks(vector, user_id, source_types, top_k, **kwargs)
    return [_normalize_hit(hit) for hit in hits]


def search_sparse_chunks(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 10,
    pdf_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    config: WorkerConfig | None = None,
    sparse_embedder: SparseEmbedder | None = None,
    store: QdrantStore | None = None,
) -> list[dict[str, Any]]:
    """Embed a query sparsely and return normalized chunk search results."""
    if contains_chat_memory(source_types) and not source_ids:
        return []
    cfg = config or WorkerConfig.from_env()
    active_sparse_embedder = sparse_embedder or SparseEmbedder(
        provider=cfg.sparse_provider,
        model=cfg.sparse_model,
    )
    active_store = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=cfg.qdrant_collection,
    )
    vector = active_sparse_embedder.embed([query])[0]
    kwargs = {"pdf_ids": pdf_ids}
    if source_ids is not None:
        kwargs["source_ids"] = source_ids
    hits = active_store.search_sparse_chunks(vector, user_id, source_types, top_k, **kwargs)
    return [_normalize_hit(hit) for hit in hits]


def search_hybrid_chunks(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 10,
    prefetch_limit: int = 30,
    pdf_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    config: WorkerConfig | None = None,
    embedder: Embedder | None = None,
    sparse_embedder: SparseEmbedder | None = None,
    store: QdrantStore | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid dense+sparse retrieval with RRF fusion."""
    if contains_chat_memory(source_types) and not source_ids:
        return []
    cfg = config or WorkerConfig.from_env()
    active_embedder = embedder or Embedder(
        provider=cfg.embedding_provider,
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key,
        gemini_api_key=cfg.gemini_api_key,
        batch_size=cfg.embedding_batch_size,
    )
    active_sparse_embedder = sparse_embedder or SparseEmbedder(
        provider=cfg.sparse_provider,
        model=cfg.sparse_model,
    )
    active_store = store or QdrantStore(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        collection_name=cfg.qdrant_collection,
    )
    dense_vector = active_embedder.embed([query])[0]
    sparse_vector = active_sparse_embedder.embed([query])[0]
    try:
        kwargs = {"pdf_ids": pdf_ids}
        if source_ids is not None:
            kwargs["source_ids"] = source_ids
        hits = active_store.search_hybrid_chunks(
            dense_vector,
            sparse_vector,
            user_id,
            source_types,
            top_k,
            prefetch_limit,
            **kwargs,
        )
    except NotImplementedError:
        kwargs = {"pdf_ids": pdf_ids}
        if source_ids is not None:
            kwargs["source_ids"] = source_ids
        dense_hits = active_store.search_chunks(
            dense_vector,
            user_id,
            source_types,
            prefetch_limit,
            **kwargs,
        )
        sparse_hits = active_store.search_sparse_chunks(
            sparse_vector,
            user_id,
            source_types,
            prefetch_limit,
            **kwargs,
        )
        hits = _local_rrf(dense_hits, sparse_hits, top_k)
    return [_normalize_hit(hit) for hit in hits]


def _local_rrf(
    dense_hits: list[Any],
    sparse_hits: list[Any],
    top_k: int,
    k: int = 60,
) -> list[Any]:
    by_id: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for hits in [dense_hits, sparse_hits]:
        for rank, hit in enumerate(hits, start=1):
            payload = getattr(hit, "payload", None) or hit.get("payload", {})
            chunk_id = str(payload.get("chunk_id"))
            by_id.setdefault(chunk_id, hit)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    fused = []
    for chunk_id in ordered_ids[:top_k]:
        hit = by_id[chunk_id]
        if hasattr(hit, "score"):
            hit.score = scores[chunk_id]
        elif isinstance(hit, dict):
            hit["score"] = scores[chunk_id]
        fused.append(hit)
    return fused


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
        "pdf_id": payload.get("pdf_id"),
    }
