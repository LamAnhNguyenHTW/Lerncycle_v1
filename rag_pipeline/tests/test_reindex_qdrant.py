from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rag_pipeline.reindex_qdrant import reindex_qdrant
from rag_pipeline.sparse_embeddings import SparseVectorData


class FakeQuery:
    def __init__(self, supabase: "FakeSupabase") -> None:
        self.supabase = supabase
        self.filters: dict[str, Any] = {}
        self.pending_update: dict[str, Any] | None = None

    def select(self, *_args):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        if self.pending_update is not None and key == "id":
            self.supabase.updates.append((value, self.pending_update))
        return self

    def update(self, values):
        self.pending_update = values
        return self

    def execute(self):
        if self.pending_update is not None:
            return SimpleNamespace(data=[])
        rows = self.supabase.rows
        for key, value in self.filters.items():
            rows = [row for row in rows if row.get(key) == value]
        self.supabase.read_filters.append(dict(self.filters))
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.read_filters: list[dict[str, Any]] = []
        self.updates: list[tuple[str, dict[str, Any]]] = []

    def table(self, name: str) -> FakeQuery:
        assert name == "rag_chunks"
        return FakeQuery(self)


class FakeEmbedder:
    dimension = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeSparseEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[SparseVectorData]:
        self.calls.append(texts)
        return [
            SparseVectorData(indices=[index], values=[float(index + 1)])
            for index, _ in enumerate(texts)
        ]


class FakeStore:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[int, bool]] = []
        self.recreate_calls: list[int] = []
        self.upserts: list[tuple[list[dict[str, Any]], bool]] = []

    def ensure_collection(self, dim: int, sparse_enabled: bool = False) -> None:
        self.ensure_calls.append((dim, sparse_enabled))

    def recreate_collection_for_hybrid(self, dim: int) -> None:
        self.recreate_calls.append(dim)

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        sparse_enabled: bool = False,
    ) -> None:
        self.upserts.append((chunks, sparse_enabled))


def _row(user_id: str = "user-1", source_type: str = "pdf") -> dict[str, Any]:
    return {
        "id": "chunk-1",
        "user_id": user_id,
        "source_type": source_type,
        "source_id": "source-1",
        "pdf_id": "pdf-1" if source_type == "pdf" else None,
        "note_id": "note-1" if source_type == "note" else None,
        "annotation_id": None,
        "rag_document_id": "document-1",
        "page_index": 0,
        "heading_path": ["Heading"],
        "content": "Process Mining text",
        "metadata": {"a": 1},
        "content_hash": "hash-1",
        "created_at": "created",
        "updated_at": "updated",
    }


def test_reindex_reads_existing_rag_chunks() -> None:
    supabase = FakeSupabase([_row()])

    reindex_qdrant(
        supabase=supabase,
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=FakeStore(),
        user_id="user-1",
    )

    assert supabase.read_filters == [{"user_id": "user-1"}]


def test_reindex_generates_sparse_vectors() -> None:
    sparse_embedder = FakeSparseEmbedder()

    reindex_qdrant(
        supabase=FakeSupabase([_row()]),
        embedder=FakeEmbedder(),
        sparse_embedder=sparse_embedder,
        store=FakeStore(),
        user_id="user-1",
    )

    assert sparse_embedder.calls == [["Process Mining text"]]


def test_reindex_upserts_hybrid_points() -> None:
    store = FakeStore()

    reindex_qdrant(
        supabase=FakeSupabase([_row()]),
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
        user_id="user-1",
    )

    chunks, sparse_enabled = store.upserts[0]
    assert sparse_enabled is True
    assert chunks[0]["id"] == "chunk-1"
    assert chunks[0]["embedding"] == [0.0, 1.0]
    assert chunks[0]["sparse_embedding"] == SparseVectorData(
        indices=[0],
        values=[1.0],
    )


def test_reindex_filters_by_user_id() -> None:
    supabase = FakeSupabase([_row(user_id="user-1"), _row(user_id="user-2")])
    store = FakeStore()

    reindex_qdrant(
        supabase=supabase,
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
        user_id="user-1",
    )

    assert len(store.upserts[0][0]) == 1
    assert store.upserts[0][0][0]["user_id"] == "user-1"


def test_reindex_does_not_cross_users() -> None:
    supabase = FakeSupabase([_row(user_id="user-2")])
    store = FakeStore()

    reindex_qdrant(
        supabase=supabase,
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
        user_id="user-1",
    )

    assert store.upserts == []


def test_reindex_recreate_collection_is_explicit() -> None:
    store = FakeStore()

    reindex_qdrant(
        supabase=FakeSupabase([_row()]),
        embedder=FakeEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        store=store,
        user_id="user-1",
        recreate_collection=True,
    )

    assert store.recreate_calls == [2]
    assert store.ensure_calls == []
