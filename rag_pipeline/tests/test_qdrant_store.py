from __future__ import annotations

import pytest

from types import SimpleNamespace

from rag_pipeline import qdrant_store
from rag_pipeline.qdrant_store import QdrantStore


class FakeModels:
    class Distance:
        COSINE = "cosine"

    class VectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class Modifier:
        IDF = "idf"

    class SparseVectorParams:
        def __init__(self, modifier=None):
            self.modifier = modifier

    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

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

    class FilterSelector:
        def __init__(self, filter):
            self.filter = filter


class FakeClient:
    def __init__(self, exists=False, collection_info=None):
        self.exists = exists
        self.collection_info = collection_info
        self.created = None
        self.recreated = None
        self.upserts = []
        self.deletes = []
        self.searches = []

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created = kwargs

    def recreate_collection(self, **kwargs):
        self.recreated = kwargs

    def get_collection(self, name):
        if self.collection_info is None:
            raise RuntimeError("missing collection info")
        return self.collection_info

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def query_points(self, **kwargs):
        self.searches.append(kwargs)
        return SimpleNamespace(points=[])


def _patch_models(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_store, "_models", lambda: FakeModels)


def test_ensure_collection_creates_when_missing(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=False)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.ensure_collection(1536)

    config = client.created["vectors_config"]["dense"]
    assert client.created["collection_name"] == "learncycle_chunks"
    assert config.size == 1536
    assert config.distance == "cosine"


def test_ensure_collection_is_idempotent_when_present(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=True)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.ensure_collection(1536)

    assert client.created is None


def test_ensure_collection_creates_dense_and_sparse_when_missing(
    monkeypatch,
) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=False)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.ensure_collection(1536, sparse_enabled=True)

    assert "dense" in client.created["vectors_config"]
    assert "sparse" in client.created["sparse_vectors_config"]
    assert client.created["sparse_vectors_config"]["sparse"].modifier == "idf"


def test_ensure_collection_detects_existing_hybrid_collection(monkeypatch) -> None:
    _patch_models(monkeypatch)
    info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": object()},
                sparse_vectors={"sparse": object()},
            )
        )
    )
    client = FakeClient(exists=True, collection_info=info)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.ensure_collection(1536, sparse_enabled=True)

    assert client.created is None


def test_ensure_collection_raises_clear_error_when_sparse_missing(
    monkeypatch,
) -> None:
    _patch_models(monkeypatch)
    info = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors={"dense": object()}, sparse_vectors={})
        )
    )
    client = FakeClient(exists=True, collection_info=info)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    with pytest.raises(RuntimeError, match="recreate_collection_for_hybrid"):
        store.ensure_collection(1536, sparse_enabled=True)


def test_recreate_collection_for_hybrid_is_explicit_only(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=True)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.recreate_collection_for_hybrid(1536)

    assert client.recreated["collection_name"] == "learncycle_chunks"
    assert "dense" in client.recreated["vectors_config"]
    assert "sparse" in client.recreated["sparse_vectors_config"]


def test_sparse_vector_created_with_idf_modifier(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=False)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.ensure_collection(1536, sparse_enabled=True)

    sparse_config = client.created["sparse_vectors_config"]["sparse"]
    assert sparse_config.modifier == FakeModels.Modifier.IDF


def test_upsert_builds_expected_payload(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.upsert_chunks(
        [
            {
                "id": "chunk-1",
                "embedding": [0.1],
                "user_id": "user-1",
                "source_type": "note",
                "source_id": "note-1",
                "page_index": 0,
                "text": "",
                "metadata": {},
                "pdf_id": None,
            }
        ]
    )

    point = client.upserts[0]["points"][0]
    assert point.id == "chunk-1"
    assert point.vector == {"dense": [0.1]}
    assert point.payload["chunk_id"] == "chunk-1"
    assert point.payload["page_index"] == 0
    assert point.payload["text"] == ""
    assert "pdf_id" not in point.payload


def test_upsert_chunks_includes_dense_and_sparse_vectors(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.upsert_chunks(
        [
            {
                "id": "chunk-1",
                "embedding": [0.1],
                "sparse_embedding": {"indices": [1, 2], "values": [0.5, 0.7]},
                "user_id": "user-1",
                "source_type": "note",
                "source_id": "note-1",
                "text": "text",
            }
        ],
        sparse_enabled=True,
    )

    vector = client.upserts[0]["points"][0].vector
    assert vector["dense"] == [0.1]
    assert vector["sparse"].indices == [1, 2]
    assert vector["sparse"].values == [0.5, 0.7]


def test_upsert_chunks_keeps_existing_payload_fields(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.upsert_chunks(
        [
            {
                "id": "chunk-1",
                "embedding": [0.1],
                "sparse_embedding": {"indices": [1], "values": [0.5]},
                "user_id": "user-1",
                "source_type": "note",
                "source_id": "note-1",
                "page_index": 0,
                "text": "",
                "metadata": {},
                "pdf_id": None,
            }
        ],
        sparse_enabled=True,
    )

    payload = client.upserts[0]["points"][0].payload
    assert payload["page_index"] == 0
    assert payload["text"] == ""
    assert "pdf_id" not in payload


def test_upsert_chunks_raises_if_dense_missing(monkeypatch) -> None:
    _patch_models(monkeypatch)
    store = QdrantStore(None, None, "learncycle_chunks", client=FakeClient())

    with pytest.raises(RuntimeError, match="dense embedding"):
        store.upsert_chunks([{"id": "chunk-1"}])


def test_upsert_chunks_raises_if_sparse_missing_when_hybrid_enabled(
    monkeypatch,
) -> None:
    _patch_models(monkeypatch)
    store = QdrantStore(None, None, "learncycle_chunks", client=FakeClient())

    with pytest.raises(RuntimeError, match="sparse embedding"):
        store.upsert_chunks(
            [{"id": "chunk-1", "embedding": [0.1]}],
            sparse_enabled=True,
        )


def test_upsert_chunks_still_supports_dense_only_mode(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.upsert_chunks([{"id": "chunk-1", "embedding": [0.1]}])

    assert client.upserts[0]["points"][0].vector == {"dense": [0.1]}


def test_upsert_chunks_handles_empty_input_without_qdrant_call(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.upsert_chunks([])

    assert client.upserts == []


def test_delete_by_source_uses_filter(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=True)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.delete_points_by_source("user-1", "note", "note-1")

    selector = client.deletes[0]["points_selector"]
    keys = [condition.key for condition in selector.filter.must]
    assert keys == ["user_id", "source_type", "source_id"]


def test_delete_by_source_is_noop_when_collection_missing(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient(exists=False)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.delete_points_by_source("user-1", "note", "note-1")

    assert client.deletes == []


def test_search_chunks_passes_user_and_source_filters(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.search_chunks([0.1], "user-1", ["note"], 5)

    search = client.searches[0]
    keys = [condition.key for condition in search["query_filter"].must]
    assert search["using"] == "dense"
    assert keys == ["user_id", "source_type"]


def test_search_chunks_supports_empty_source_types_as_all_sources(monkeypatch) -> None:
    _patch_models(monkeypatch)
    client = FakeClient()
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.search_chunks([0.1], "user-1", [], 5)

    keys = [condition.key for condition in client.searches[0]["query_filter"].must]
    assert keys == ["user_id"]
