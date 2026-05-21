from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseVectorData


class QueryAwareEmbedder:
    def embed(self, texts):
        query_number = int(texts[0].split("-")[-1])
        return [[float(query_number)]]


class QueryAwareSparseEmbedder:
    def embed(self, texts):
        query_number = int(texts[0].split("-")[-1])
        return [SparseVectorData(indices=[query_number], values=[1.0])]


class DeterministicHybridStore:
    def __init__(self) -> None:
        self.native_calls = 0
        self.dense_calls = 0
        self.sparse_calls = 0

    def search_hybrid_chunks(
        self,
        dense_vector,
        sparse_vector,
        user_id,
        source_types,
        top_k,
        prefetch_limit,
        pdf_ids=None,
        source_ids=None,
    ):
        self.native_calls += 1
        query_number = int(dense_vector[0])
        return [_hit(f"source-{query_number}-{index}") for index in range(top_k)]

    def search_chunks(
        self,
        dense_vector,
        user_id,
        source_types,
        top_k,
        pdf_ids=None,
        source_ids=None,
    ):
        self.dense_calls += 1
        query_number = int(dense_vector[0])
        return [_hit(f"source-{query_number}-{index}", score=1.0 / (index + 1)) for index in range(top_k)]

    def search_sparse_chunks(
        self,
        sparse_vector,
        user_id,
        source_types,
        top_k,
        pdf_ids=None,
        source_ids=None,
    ):
        self.sparse_calls += 1
        query_number = int(sparse_vector.indices[0])
        return [_hit(f"source-{query_number}-{index}", score=1.0 / (index + 2)) for index in range(top_k)]


def _hit(source_id: str, score: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": f"chunk-{source_id}",
            "text": f"text for {source_id}",
            "source_type": "pdf",
            "source_id": source_id,
            "pdf_id": source_id,
            "metadata": {"content_hash": source_id},
        },
    )


def _config(native_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(qdrant_native_hybrid_enabled=native_enabled)


def test_native_and_fallback_hybrid_have_expected_set_overlap() -> None:
    """Ordering differences between native fusion and fallback merge are expected and not a regression."""
    queries = [f"query-{index}" for index in range(1, 6)]
    for query in queries:
        store = DeterministicHybridStore()
        native_results = search_hybrid_chunks(
            query,
            "user-1",
            top_k=5,
            prefetch_limit=8,
            source_types=["pdf"],
            config=_config(native_enabled=True),
            embedder=QueryAwareEmbedder(),
            sparse_embedder=QueryAwareSparseEmbedder(),
            store=store,
        )
        fallback_results = search_hybrid_chunks(
            query,
            "user-1",
            top_k=5,
            prefetch_limit=8,
            source_types=["pdf"],
            config=_config(native_enabled=False),
            embedder=QueryAwareEmbedder(),
            sparse_embedder=QueryAwareSparseEmbedder(),
            store=store,
        )

        native_ids = [result["source_id"] for result in native_results]
        fallback_ids = [result["source_id"] for result in fallback_results]
        overlap = len(set(native_ids) & set(fallback_ids)) / len(set(native_ids))

        assert len(native_ids) == len(set(native_ids))
        assert len(fallback_ids) == len(set(fallback_ids))
        assert all(result["chunk_id"] and result["text"] for result in native_results)
        assert all(result["chunk_id"] and result["text"] for result in fallback_results)
        assert overlap >= 0.8
