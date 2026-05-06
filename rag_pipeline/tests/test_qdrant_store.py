from __future__ import annotations

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

    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

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
    def __init__(self, exists=False):
        self.exists = exists
        self.created = None
        self.upserts = []
        self.deletes = []
        self.searches = []

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created = kwargs

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
