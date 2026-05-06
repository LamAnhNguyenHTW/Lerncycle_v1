from __future__ import annotations

from types import SimpleNamespace

from rag_pipeline.retrieval import search_chunks


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts = []

    def embed(self, texts):
        self.texts.append(texts)
        return [[0.1, 0.2]]


class FakeStore:
    def __init__(self) -> None:
        self.calls = []

    def search_chunks(self, query_vector, user_id, source_types, top_k):
        self.calls.append((query_vector, user_id, source_types, top_k))
        return [
            SimpleNamespace(
                score=0.9,
                payload={
                    "chunk_id": "chunk-1",
                    "text": "RAG text",
                    "source_type": "note",
                    "source_id": "note-1",
                    "page_index": 1,
                    "heading": "Intro",
                    "metadata": {"a": 1},
                },
            )
        ]


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
            "page_index": 1,
            "title": None,
            "heading": "Intro",
            "metadata": {"a": 1},
        }
    ]
