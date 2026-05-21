from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embedding_cache import QueryEmbeddingCache
from rag_pipeline import retrieval
from rag_pipeline.retrieval import search_chunks


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(self.calls)]]


class FakeStore:
    def __init__(self) -> None:
        self.vectors = []

    def search_chunks(self, query_vector, user_id, source_types, top_k, pdf_ids=None, source_ids=None):
        self.vectors.append(query_vector)
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    "chunk_id": "chunk-1",
                    "text": "text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                },
            )
        ]


def _config() -> WorkerConfig:
    return WorkerConfig(
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="key",
        query_embedding_cache_enabled=True,
        query_embedding_cache_max_entries=8,
    )


def test_retrieval_reuses_dense_embedding_for_same_query_across_scopes(monkeypatch) -> None:
    cache = QueryEmbeddingCache(max_entries=8)
    monkeypatch.setattr(retrieval, "default_query_embedding_cache", cache)
    embedder = CountingEmbedder()
    store = FakeStore()

    search_chunks("Process Mining", "user-1", pdf_ids=["pdf-1"], config=_config(), embedder=embedder, store=store)
    search_chunks(" process   mining ", "user-1", pdf_ids=["pdf-2"], config=_config(), embedder=embedder, store=store)

    assert embedder.calls == 1
    assert store.vectors == [[1.0], [1.0]]


def test_retrieval_embeds_different_queries_separately(monkeypatch) -> None:
    cache = QueryEmbeddingCache(max_entries=8)
    monkeypatch.setattr(retrieval, "default_query_embedding_cache", cache)
    embedder = CountingEmbedder()

    search_chunks("query one", "user-1", config=_config(), embedder=embedder, store=FakeStore())
    search_chunks("query two", "user-1", config=_config(), embedder=embedder, store=FakeStore())

    assert embedder.calls == 2
