from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline import retrieval
from rag_pipeline.config import WorkerConfig
from rag_pipeline.result_cache import RetrievalResultCache
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseVectorData


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(self.calls)]]


class CountingSparseEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [SparseVectorData(indices=[self.calls], values=[1.0])]


class CountingStore:
    def __init__(self) -> None:
        self.hybrid_calls = 0

    def search_hybrid_chunks(self, *args, **kwargs):
        self.hybrid_calls += 1
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    "chunk_id": f"chunk-{self.hybrid_calls}",
                    "text": "cached text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "metadata": {"content_hash": "hash-1"},
                },
            )
        ]


def _config(**overrides) -> WorkerConfig:
    values = {
        "supabase_url": "https://example.supabase.co",
        "supabase_service_role_key": "key",
        "retrieval_result_cache_enabled": True,
        "retrieval_result_cache_max_entries": 8,
        "retrieval_result_cache_ttl_s": 300,
        "query_embedding_cache_enabled": False,
    }
    values.update(overrides)
    return WorkerConfig(**values)


def test_retrieval_result_cache_skips_embeddings_and_qdrant_on_hit(monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "default_retrieval_result_cache", RetrievalResultCache(max_entries=8))
    embedder = CountingEmbedder()
    sparse = CountingSparseEmbedder()
    store = CountingStore()

    first = search_hybrid_chunks(
        "Process Mining",
        "user-1",
        source_types=["pdf"],
        source_ids=["pdf-1"],
        config=_config(),
        embedder=embedder,
        sparse_embedder=sparse,
        store=store,
    )
    second = search_hybrid_chunks(
        " process   mining ",
        "user-1",
        source_types=["pdf"],
        source_ids=["pdf-1"],
        config=_config(),
        embedder=embedder,
        sparse_embedder=sparse,
        store=store,
    )

    assert first == second
    assert embedder.calls == 1
    assert sparse.calls == 1
    assert store.hybrid_calls == 1


def test_retrieval_result_cache_key_includes_user_scope_and_config(monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "default_retrieval_result_cache", RetrievalResultCache(max_entries=8))
    embedder = CountingEmbedder()
    sparse = CountingSparseEmbedder()
    store = CountingStore()

    search_hybrid_chunks("q", "user-1", source_ids=["pdf-1"], config=_config(), embedder=embedder, sparse_embedder=sparse, store=store)
    search_hybrid_chunks("q", "user-2", source_ids=["pdf-1"], config=_config(), embedder=embedder, sparse_embedder=sparse, store=store)
    search_hybrid_chunks("q", "user-1", source_ids=["pdf-2"], config=_config(), embedder=embedder, sparse_embedder=sparse, store=store)
    search_hybrid_chunks("q", "user-1", source_ids=["pdf-1"], config=_config(embedding_model="other"), embedder=embedder, sparse_embedder=sparse, store=store)

    assert store.hybrid_calls == 4


def test_retrieval_result_cache_ttl_and_lru(monkeypatch) -> None:
    cache = RetrievalResultCache(max_entries=1, ttl_s=0)
    monkeypatch.setattr(retrieval, "default_retrieval_result_cache", cache)
    store = CountingStore()

    search_hybrid_chunks("one", "user-1", source_ids=["pdf-1"], config=_config(retrieval_result_cache_ttl_s=0), embedder=CountingEmbedder(), sparse_embedder=CountingSparseEmbedder(), store=store)
    search_hybrid_chunks("one", "user-1", source_ids=["pdf-1"], config=_config(retrieval_result_cache_ttl_s=0), embedder=CountingEmbedder(), sparse_embedder=CountingSparseEmbedder(), store=store)
    assert store.hybrid_calls == 2

    cache = RetrievalResultCache(max_entries=1, ttl_s=300)
    monkeypatch.setattr(retrieval, "default_retrieval_result_cache", cache)
    store = CountingStore()
    search_hybrid_chunks("one", "user-1", source_ids=["pdf-1"], config=_config(retrieval_result_cache_max_entries=1), embedder=CountingEmbedder(), sparse_embedder=CountingSparseEmbedder(), store=store)
    search_hybrid_chunks("two", "user-1", source_ids=["pdf-1"], config=_config(retrieval_result_cache_max_entries=1), embedder=CountingEmbedder(), sparse_embedder=CountingSparseEmbedder(), store=store)
    search_hybrid_chunks("one", "user-1", source_ids=["pdf-1"], config=_config(retrieval_result_cache_max_entries=1), embedder=CountingEmbedder(), sparse_embedder=CountingSparseEmbedder(), store=store)
    assert store.hybrid_calls == 3
