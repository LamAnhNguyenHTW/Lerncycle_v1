from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline.retrieval import search_chunks
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.retrieval import search_sparse_chunks
from rag_pipeline.sparse_embeddings import SparseVectorData


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts = []

    def embed(self, texts):
        self.texts.append(texts)
        return [[0.1, 0.2]]


class FakeStore:
    def __init__(self) -> None:
        self.calls = []

    def search_chunks(self, query_vector, user_id, source_types, top_k, pdf_ids=None):
        self.calls.append((query_vector, user_id, source_types, top_k, pdf_ids))
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    "chunk_id": "chunk-1",
                    "text": "RAG text",
                    "source_type": "note",
                    "source_id": "note-1",
                    "pdf_id": "pdf-note",
                    "page_index": 1,
                    "heading": "Intro",
                    "metadata": {"a": 1},
                },
            )
        ]

    def search_sparse_chunks(self, query_vector, user_id, source_types, top_k, pdf_ids=None):
        self.calls.append((query_vector, user_id, source_types, top_k, "sparse", pdf_ids))
        return [
            SimpleNamespace(
                score=0.8,
                payload={
                    "chunk_id": "chunk-2",
                    "text": "Sparse text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                    "page_index": 2,
                    "title": "Title",
                    "metadata": {"b": 2},
                },
            )
        ]

    def search_hybrid_chunks(
        self,
        dense_vector,
        sparse_vector,
        user_id,
        source_types,
        top_k,
        prefetch_limit,
        pdf_ids=None,
    ):
        self.calls.append(
            (
                dense_vector,
                sparse_vector,
                user_id,
                source_types,
                top_k,
                prefetch_limit,
                "hybrid",
                pdf_ids,
            )
        )
        return [
            SimpleNamespace(
                score=0.95,
                payload={
                    "chunk_id": "chunk-h",
                    "text": "Hybrid text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                    "page_index": 3,
                    "metadata": {},
                },
            )
        ]


class FakeSparseEmbedder:
    def __init__(self) -> None:
        self.texts = []

    def embed(self, texts):
        self.texts.append(texts)
        return [SparseVectorData(indices=[1, 2], values=[0.5, 0.7])]


def test_search_chunks_embeds_query_and_searches_qdrant() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()

    search_chunks("query", "user-1", embedder=embedder, store=store)

    assert embedder.texts == [["query"]]
    assert store.calls[0][0] == [0.1, 0.2]


def test_search_chunks_filters_by_user_id() -> None:
    store = FakeStore()

    search_chunks("query", "user-1", embedder=FakeEmbedder(), store=store)

    assert store.calls[0][1] == "user-1"


def test_search_chunks_filters_by_source_types() -> None:
    store = FakeStore()

    search_chunks(
        "query",
        "user-1",
        source_types=["note"],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert store.calls[0][2] == ["note"]


def test_search_chunks_returns_normalized_results() -> None:
    results = search_chunks(
        "query",
        "user-1",
        embedder=FakeEmbedder(),
        store=FakeStore(),
    )

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "RAG text",
            "score": 0.9,
            "source_type": "note",
            "source_id": "note-1",
            "pdf_id": "pdf-note",
            "page_index": 1,
            "title": None,
            "heading": "Intro",
            "metadata": {"a": 1},
            "pdf_id": "pdf-note",
        }
    ]


def test_search_chunks_filters_by_pdf_ids() -> None:
    store = FakeStore()

    search_chunks(
        "query",
        "user-1",
        pdf_ids=["pdf-1"],
        embedder=FakeEmbedder(),
        store=store,
    )

    assert store.calls[0][4] == ["pdf-1"]


def test_sparse_search_embeds_query_sparse() -> None:
    sparse_embedder = FakeSparseEmbedder()
    store = FakeStore()

    search_sparse_chunks(
        "query",
        "user-1",
        sparse_embedder=sparse_embedder,
        store=store,
    )

    assert sparse_embedder.texts == [["query"]]


def test_sparse_search_uses_sparse_vector_name() -> None:
    store = FakeStore()

    search_sparse_chunks(
        "query",
        "user-1",
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.calls[0][0] == SparseVectorData(
        indices=[1, 2],
        values=[0.5, 0.7],
    )
    assert store.calls[0][4] == "sparse"


def test_sparse_search_filters_by_user_id() -> None:
    store = FakeStore()

    search_sparse_chunks(
        "query",
        "user-1",
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.calls[0][1] == "user-1"


def test_sparse_search_filters_by_source_types() -> None:
    store = FakeStore()

    search_sparse_chunks(
        "query",
        "user-1",
        source_types=["pdf"],
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.calls[0][2] == ["pdf"]


def test_sparse_search_returns_normalized_results() -> None:
    results = search_sparse_chunks(
        "query",
        "user-1",
        sparse_embedder=FakeSparseEmbedder(),
        store=FakeStore(),
    )

    assert results == [
        {
            "chunk_id": "chunk-2",
            "text": "Sparse text",
            "score": 0.8,
            "source_type": "pdf",
            "source_id": "pdf-1",
            "pdf_id": "pdf-1",
            "page_index": 2,
            "title": "Title",
            "heading": None,
            "metadata": {"b": 2},
            "pdf_id": "pdf-1",
        }
    ]


def test_hybrid_search_embeds_dense_and_sparse_query() -> None:
    embedder = FakeEmbedder()
    sparse_embedder = FakeSparseEmbedder()

    search_hybrid_chunks(
        "query",
        "user-1",
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        store=FakeStore(),
    )

    assert embedder.texts == [["query"]]
    assert sparse_embedder.texts == [["query"]]


def test_hybrid_search_uses_prefetch_for_dense_and_sparse_when_supported() -> None:
    store = FakeStore()

    search_hybrid_chunks(
        "query",
        "user-1",
        prefetch_limit=33,
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    call = store.calls[0]
    assert call[0] == [0.1, 0.2]
    assert call[1] == SparseVectorData(indices=[1, 2], values=[0.5, 0.7])
    assert call[5] == 33
    assert call[6] == "hybrid"


def test_hybrid_search_uses_rrf_fusion() -> None:
    store = FakeStore()

    search_hybrid_chunks(
        "query",
        "user-1",
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.calls[0][6] == "hybrid"


def test_hybrid_search_filters_by_user_id_in_both_paths() -> None:
    store = FakeStore()

    search_hybrid_chunks(
        "query",
        "user-1",
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.calls[0][2] == "user-1"


def test_hybrid_search_filters_by_source_types() -> None:
    store = FakeStore()

    search_hybrid_chunks(
        "query",
        "user-1",
        source_types=["pdf"],
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.calls[0][3] == ["pdf"]


def test_hybrid_search_returns_normalized_results() -> None:
    results = search_hybrid_chunks(
        "query",
        "user-1",
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=FakeStore(),
    )

    assert results == [
        {
            "chunk_id": "chunk-h",
            "text": "Hybrid text",
            "score": 0.95,
            "source_type": "pdf",
            "source_id": "pdf-1",
            "pdf_id": "pdf-1",
            "page_index": 3,
            "title": None,
            "heading": None,
            "metadata": {},
            "pdf_id": "pdf-1",
        }
    ]


class FallbackStore(FakeStore):
    def search_hybrid_chunks(self, *args, **kwargs):
        raise NotImplementedError


def test_hybrid_search_can_use_local_rrf_fallback() -> None:
    results = search_hybrid_chunks(
        "query",
        "user-1",
        top_k=1,
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=FallbackStore(),
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] in {"chunk-1", "chunk-2"}
