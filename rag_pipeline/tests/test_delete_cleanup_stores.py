"""Store-level tests for source deletion cleanup (Qdrant + both Neo4j layers)."""

from __future__ import annotations

from rag_pipeline import qdrant_store
from rag_pipeline.learning_structure.neo4j_store import delete_learning_graph
from rag_pipeline.neo4j_graph_store import Neo4jGraphStore
from rag_pipeline.qdrant_store import QdrantStore


class FakeModels:
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


class FakeQdrantClient:
    def __init__(self, exists: bool = True) -> None:
        self.exists = exists
        self.deletes = []

    def collection_exists(self, name):
        return self.exists

    def delete(self, collection_name, points_selector):
        self.deletes.append((collection_name, points_selector))


class FakeSession:
    def __init__(self, driver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, statement, parameters):
        self.driver.calls.append((statement, parameters))
        return []


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []

    def session(self, database=None):
        return FakeSession(self)


def test_qdrant_delete_points_by_pdf_id_filters_on_user_and_pdf(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_store, "_models", lambda: FakeModels)
    client = FakeQdrantClient(exists=True)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.delete_points_by_pdf_id("user-1", "pdf-1")

    assert len(client.deletes) == 1
    _, selector = client.deletes[0]
    conditions = {cond.key: cond.match.value for cond in selector.filter.must}
    assert conditions == {"user_id": "user-1", "pdf_id": "pdf-1"}


def test_qdrant_delete_points_by_pdf_id_noop_without_collection(monkeypatch) -> None:
    monkeypatch.setattr(qdrant_store, "_models", lambda: FakeModels)
    client = FakeQdrantClient(exists=False)
    store = QdrantStore(None, None, "learncycle_chunks", client=client)

    store.delete_points_by_pdf_id("user-1", "pdf-1")

    assert client.deletes == []


def test_graph_store_delete_by_pdf_id_removes_rels_chunks_and_orphans() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore("bolt://x", "neo4j", "pw", database="neo4j", driver=driver)

    store.delete_by_pdf_id("user-1", "pdf-1")

    statements = [call[0] for call in driver.calls]
    assert any("RELATED" in stmt and "pdf_id" in stmt for stmt in statements)
    assert any("Chunk" in stmt and "DETACH DELETE" in stmt for stmt in statements)
    assert any("NOT (concept)--()" in stmt for stmt in statements)
    for _, params in driver.calls:
        assert params.get("user_id") == "user-1"


def test_graph_store_delete_by_source_detaches_chunks() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore("bolt://x", "neo4j", "pw", database="neo4j", driver=driver)

    store.delete_by_source("user-1", "pdf", "pdf-1")

    chunk_statements = [
        call[0] for call in driver.calls if "Chunk" in call[0] and "DELETE" in call[0]
    ]
    assert chunk_statements
    assert all("DETACH DELETE" in stmt for stmt in chunk_statements)


def test_delete_learning_graph_removes_layer_chunks_and_document() -> None:
    driver = FakeDriver()

    delete_learning_graph("user-1", "pdf-1", driver=driver, database="neo4j")

    statements = [call[0] for call in driver.calls]
    assert any("graph_scope: 'learning_structure'" in stmt and "DELETE r" in stmt for stmt in statements)
    assert any("LearningTopic" in stmt and "DETACH DELETE" in stmt for stmt in statements)
    assert any("Chunk" in stmt and "DETACH DELETE" in stmt for stmt in statements)
    assert any("Document" in stmt and "DETACH DELETE" in stmt for stmt in statements)
    for _, params in driver.calls:
        assert params.get("user_id") == "user-1"
        assert params.get("source_id") == "pdf-1"
        assert params.get("source_type") == "pdf"


def test_delete_learning_graph_is_idempotent_against_empty_graph() -> None:
    driver = FakeDriver()

    delete_learning_graph("user-1", "pdf-1", driver=driver)
    delete_learning_graph("user-1", "pdf-1", driver=driver)

    assert len(driver.calls) == 8
