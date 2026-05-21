"""First dense retrieval helper over Qdrant."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embedding_cache import EmbeddingCacheKey
from rag_pipeline.embedding_cache import default_query_embedding_cache
from rag_pipeline.embedding_cache import normalize_query_for_embedding
from rag_pipeline.embeddings import Embedder
from rag_pipeline.observability.timing import Timer
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.result_cache import build_filter_hash
from rag_pipeline.result_cache import build_source_version_hash
from rag_pipeline.result_cache import default_retrieval_result_cache
from rag_pipeline.result_cache import make_retrieval_cache_key
from rag_pipeline.source_types import contains_chat_memory
from rag_pipeline.source_types import contains_web
from rag_pipeline.sparse_embeddings import SparseEmbedder


logger = logging.getLogger(__name__)


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
    if contains_web(source_types):
        return []
    if contains_chat_memory(source_types) and not source_ids:
        return []
    explicit_config = config is not None
    cfg = config or WorkerConfig.from_env()
    cache_enabled = embedder is None or (
        explicit_config and hasattr(cfg, "embedding_provider")
    )
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
    retrieval_cache_allowed = (store is None and embedder is None) or (
        explicit_config and hasattr(cfg, "retrieval_result_cache_enabled")
    )
    cache_key = _retrieval_cache_key(
        cfg,
        query=query,
        user_id=user_id,
        source_types=source_types,
        pdf_ids=pdf_ids,
        source_ids=source_ids,
        top_k=top_k,
        candidate_k=top_k,
        retrieval_mode="dense",
        hybrid_strategy="dense",
    )
    cached = _get_cached_retrieval_results(cfg, cache_key) if retrieval_cache_allowed else None
    if cached is not None:
        return cached
    vector = _embed_dense_query_cached(active_embedder, query, cfg, cache_enabled=cache_enabled)
    kwargs = {"pdf_ids": pdf_ids}
    if source_ids is not None:
        kwargs["source_ids"] = source_ids
    with Timer("qdrant_retrieve"):
        hits = active_store.search_chunks(vector, user_id, source_types, top_k, **kwargs)
    results = [_normalize_hit(hit) for hit in hits]
    if retrieval_cache_allowed:
        _set_cached_retrieval_results(cfg, cache_key, results)
    return results


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
    if contains_web(source_types):
        return []
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
    with Timer("embed_sparse"):
        vector = active_sparse_embedder.embed([query])[0]
    kwargs = {"pdf_ids": pdf_ids}
    if source_ids is not None:
        kwargs["source_ids"] = source_ids
    with Timer("qdrant_retrieve"):
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
    if contains_web(source_types):
        return []
    if contains_chat_memory(source_types) and not source_ids:
        return []
    explicit_config = config is not None
    cfg = config or WorkerConfig.from_env()
    cache_enabled = embedder is None or (
        explicit_config and hasattr(cfg, "embedding_provider")
    )
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
    retrieval_cache_allowed = (store is None and embedder is None and sparse_embedder is None) or (
        explicit_config and hasattr(cfg, "retrieval_result_cache_enabled")
    )
    hybrid_strategy = "native" if getattr(cfg, "qdrant_native_hybrid_enabled", True) else "fallback"
    cache_key = _retrieval_cache_key(
        cfg,
        query=query,
        user_id=user_id,
        source_types=source_types,
        pdf_ids=pdf_ids,
        source_ids=source_ids,
        top_k=top_k,
        candidate_k=prefetch_limit,
        retrieval_mode="hybrid",
        hybrid_strategy=hybrid_strategy,
    )
    cached = _get_cached_retrieval_results(cfg, cache_key) if retrieval_cache_allowed else None
    if cached is not None:
        return cached
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            _embed_dense_query_cached,
            active_embedder,
            query,
            cfg,
            cache_enabled,
        )
        sparse_future = executor.submit(_embed_sparse_query, active_sparse_embedder, query)
        dense_vector = dense_future.result()
        sparse_vector = sparse_future.result()
    kwargs = {"pdf_ids": pdf_ids}
    if source_ids is not None:
        kwargs["source_ids"] = source_ids
    with Timer("qdrant_retrieve"):
        if cfg.qdrant_native_hybrid_enabled:
            try:
                hits = active_store.search_hybrid_chunks(
                    dense_vector,
                    sparse_vector,
                    user_id,
                    source_types,
                    top_k,
                    prefetch_limit,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "Native Qdrant hybrid search failed; falling back to local RRF.",
                    exc_info=True,
                )
                hits = _search_hybrid_chunks_fallback(
                    active_store,
                    dense_vector,
                    sparse_vector,
                    user_id,
                    source_types,
                    prefetch_limit,
                    top_k,
                    **kwargs,
                )
        else:
            hits = _search_hybrid_chunks_fallback(
                active_store,
                dense_vector,
                sparse_vector,
                user_id,
                source_types,
                prefetch_limit,
                top_k,
                **kwargs,
            )
    results = [_normalize_hit(hit) for hit in hits]
    if retrieval_cache_allowed:
        _set_cached_retrieval_results(cfg, cache_key, results)
    return results


def _search_hybrid_chunks_fallback(
    store: QdrantStore,
    dense_vector: list[float],
    sparse_vector: Any,
    user_id: str,
    source_types: list[str] | None,
    prefetch_limit: int,
    top_k: int,
    **kwargs: Any,
) -> list[Any]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            store.search_chunks,
            dense_vector,
            user_id,
            source_types,
            prefetch_limit,
            **kwargs,
        )
        sparse_future = executor.submit(
            store.search_sparse_chunks,
            sparse_vector,
            user_id,
            source_types,
            prefetch_limit,
            **kwargs,
        )
        dense_hits = dense_future.result()
        sparse_hits = sparse_future.result()
    return _local_rrf(dense_hits, sparse_hits, top_k)


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


def _embed_dense_query(embedder: Any, query: str) -> list[float]:
    with Timer("embed_dense"):
        return embedder.embed([query])[0]


def _embed_dense_query_cached(
    embedder: Any,
    query: str,
    config: WorkerConfig,
    cache_enabled: bool = True,
) -> list[float]:
    if not cache_enabled or not getattr(config, "query_embedding_cache_enabled", True):
        return _embed_dense_query(embedder, query)
    default_query_embedding_cache.max_entries = getattr(
        config,
        "query_embedding_cache_max_entries",
        512,
    )
    key = EmbeddingCacheKey(
        normalized_query=normalize_query_for_embedding(query),
        provider=config.embedding_provider,
        model=config.embedding_model,
        embedding_kind="dense",
    )
    with Timer("embed_dense"):
        return default_query_embedding_cache.get_or_compute(
            key,
            lambda: embedder.embed([query])[0],
        )


def _embed_sparse_query(sparse_embedder: Any, query: str) -> Any:
    with Timer("embed_sparse"):
        return sparse_embedder.embed([query])[0]


def _retrieval_cache_key(
    config: WorkerConfig,
    *,
    query: str,
    user_id: str,
    source_types: list[str] | None,
    pdf_ids: list[str] | None,
    source_ids: list[str] | None,
    top_k: int,
    candidate_k: int,
    retrieval_mode: str,
    hybrid_strategy: str,
) -> Any:
    filter_hash = build_filter_hash(
        source_types=source_types,
        pdf_ids=pdf_ids,
        source_ids=source_ids,
    )
    scoped_source_ids = source_ids or pdf_ids or []
    return make_retrieval_cache_key(
        user_id=user_id,
        query=query,
        source_types=source_types,
        source_ids=scoped_source_ids,
        source_version_hash=build_source_version_hash(source_ids=scoped_source_ids),
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        candidate_k=candidate_k,
        config=config,
        hybrid_strategy=hybrid_strategy,
        filter_hash=filter_hash,
    )


def _get_cached_retrieval_results(config: WorkerConfig, cache_key: Any) -> list[dict[str, Any]] | None:
    if not getattr(config, "retrieval_result_cache_enabled", True):
        return None
    default_retrieval_result_cache.max_entries = getattr(
        config,
        "retrieval_result_cache_max_entries",
        512,
    )
    default_retrieval_result_cache.ttl_s = getattr(
        config,
        "retrieval_result_cache_ttl_s",
        300,
    )
    return default_retrieval_result_cache.get(cache_key)


def _set_cached_retrieval_results(
    config: WorkerConfig,
    cache_key: Any,
    results: list[dict[str, Any]],
) -> None:
    if getattr(config, "retrieval_result_cache_enabled", True):
        default_retrieval_result_cache.set(cache_key, results)


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
