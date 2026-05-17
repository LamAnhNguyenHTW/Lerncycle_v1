from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline import qdrant_store
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseVectorData


class FakeModels:
    class SparseVector:
        def __init__(self, indices, values):
            self.indices = indices
            self.values = values

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class MatchAny:
        def __init__(self, any):
            self.any = any

    class FieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must):
            self.must = must

    class Prefetch:
        def __init__(self, query, using, filter, limit):
            self.query = query
            self.using = using
            self.filter = filter
            self.limit = limit

    class Fusion:
        RRF = "rrf"

    class FusionQuery:
        def __init__(self, fusion):
            self.fusion = fusion


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2]]


class FakeSparseEmbedder:
    def embed(self, texts):
        return [SparseVectorData(indices=[1, 2], values=[0.5, 0.7])]


class NativeClient:
    def __init__(self) -> None:
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.95,
                    payload={
                        "chunk_id": "chunk-native",
                        "text": "Native hybrid text",
                        "source_type": "pdf",
                        "source_id": "pdf-1",
                        "pdf_id": "pdf-1",
                        "page_index": 1,
                        "metadata": {"path": "native"},
                    },
                )
            ]
        )


class NativeStore:
    def __init__(self, *, fail_native: bool = False) -> None:
        self.fail_native = fail_native
        self.native_calls = 0
        self.dense_calls = 0
        self.sparse_calls = 0

    def search_hybrid_chunks(self, *args, **kwargs):
        self.native_calls += 1
        if self.fail_native:
            raise RuntimeError("native unavailable")
        return [
            SimpleNamespace(
                score=0.95,
                payload={
                    "chunk_id": "chunk-native",
                    "text": "Native hybrid text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                    "page_index": 1,
                    "metadata": {"path": "native"},
                },
            )
        ]

    def search_chunks(self, *args, **kwargs):
        self.dense_calls += 1
        return [
            SimpleNamespace(
                score=0.8,
                payload={
                    "chunk_id": "chunk-dense",
                    "text": "Dense text",
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "pdf_id": "pdf-1",
                },
            )
        ]

    def search_sparse_chunks(self, *args, **kwargs):
        self.sparse_calls += 1
        return [
            SimpleNamespace(
                score=0.7,
                payload={
                    "chunk_id": "chunk-sparse",
                    "text": "Sparse text",
                    "source_type": "pdf",
                    "source_id": "pdf-2",
                    "pdf_id": "pdf-2",
                },
            )
        ]


def _config(native_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(qdrant_native_hybrid_enabled=native_enabled)


def test_qdrant_store_native_hybrid_uses_one_fusion_query(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_store, "_models", lambda: FakeModels)
    client = NativeClient()
    store = qdrant_store.QdrantStore(None, None, "learncycle_chunks", client=client)

    hits = store.search_hybrid_chunks(
        dense_vector=[0.1, 0.2],
        sparse_vector=SparseVectorData(indices=[1, 2], values=[0.5, 0.7]),
        user_id="user-1",
        source_types=["pdf"],
        top_k=5,
        prefetch_limit=30,
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert len(call["prefetch"]) == 2
    assert isinstance(call["query"], FakeModels.FusionQuery)
    assert call["query"].fusion == FakeModels.Fusion.RRF
    assert hits[0].payload["chunk_id"] == "chunk-native"


def test_retrieval_native_hybrid_returns_normalized_chunk_shape() -> None:
    store = NativeStore()

    results = search_hybrid_chunks(
        "query",
        "user-1",
        config=_config(native_enabled=True),
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.native_calls == 1
    assert store.dense_calls == 0
    assert store.sparse_calls == 0
    assert results == [
        {
            "chunk_id": "chunk-native",
            "text": "Native hybrid text",
            "score": 0.95,
            "source_type": "pdf",
            "source_id": "pdf-1",
            "page_index": 1,
            "title": None,
            "heading": None,
            "metadata": {"path": "native"},
            "pdf_id": "pdf-1",
        }
    ]


def test_retrieval_falls_back_when_native_hybrid_disabled() -> None:
    store = NativeStore()

    results = search_hybrid_chunks(
        "query",
        "user-1",
        top_k=1,
        config=_config(native_enabled=False),
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.native_calls == 0
    assert store.dense_calls == 1
    assert store.sparse_calls == 1
    assert results[0]["chunk_id"] in {"chunk-dense", "chunk-sparse"}


def test_retrieval_falls_back_when_native_hybrid_errors(caplog) -> None:
    store = NativeStore(fail_native=True)

    results = search_hybrid_chunks(
        "query",
        "user-1",
        top_k=1,
        config=_config(native_enabled=True),
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
    )

    assert store.native_calls == 1
    assert store.dense_calls == 1
    assert store.sparse_calls == 1
    assert results[0]["chunk_id"] in {"chunk-dense", "chunk-sparse"}
    assert "falling back to local RRF" in caplog.text
