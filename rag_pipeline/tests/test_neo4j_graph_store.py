from __future__ import annotations

from rag_pipeline.graph_schema import GraphEdge, GraphExtraction, GraphNode
from rag_pipeline.neo4j_graph_store import Neo4jGraphStore


class FakeRecord(dict):
    def data(self):
        return dict(self)


class FakeSession:
    def __init__(self, driver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, statement, parameters):
        self.driver.calls.append((statement, parameters))
        return [FakeRecord(name="Process Mining")]


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []

    def session(self, database=None):
        self.database = database
        return FakeSession(self)


def _store():
    driver = FakeDriver()
    return Neo4jGraphStore("bolt://x", "neo4j", "pw", database="neo4j", driver=driver), driver


def _extraction():
    return GraphExtraction(
        nodes=[GraphNode("Process Mining"), GraphNode("Event Logs")],
        edges=[GraphEdge("Process Mining", "Event Logs", "uses")],
    )


def test_neo4j_store_creates_constraints() -> None:
    store, driver = _store()

    store.ensure_constraints()

    assert any("CREATE CONSTRAINT" in call[0] for call in driver.calls)


def test_neo4j_store_upserts_concepts_chunks_and_relationships() -> None:
    store, driver = _store()

    stats = store.upsert_extraction(
        "user-1",
        {"chunk_id": "chunk-1", "source_type": "pdf", "source_id": "pdf-1", "text": "x" * 500},
        _extraction(),
    )

    joined = "\n".join(call[0] for call in driver.calls)
    assert "MERGE (chunk:Chunk" in joined
    assert "MERGE (concept:Concept" in joined
    assert "MERGE (source)-[rel:RELATED" in joined
    assert stats == {"nodes_upserted": 2, "relationships_upserted": 1}


def test_neo4j_store_delete_by_source() -> None:
    store, driver = _store()

    store.delete_by_source("user-1", "pdf", "pdf-1")

    assert all(call[1].get("user_id") == "user-1" for call in driver.calls)
    assert any("DELETE" in call[0] for call in driver.calls)


def test_neo4j_store_search_concepts_respects_source_filters() -> None:
    store, driver = _store()

    rows = store.search_concepts("user-1", "process", ["pdf"], ["pdf-1"])

    assert rows == [{"name": "Process Mining"}]
    assert driver.calls[-1][1]["source_types"] == ["pdf"]
    assert driver.calls[-1][1]["source_ids"] == ["pdf-1"]


def test_neo4j_store_neighborhood_respects_source_filters() -> None:
    store, driver = _store()

    store.get_neighborhood("user-1", ["process mining"], source_types=["pdf"], source_ids=["pdf-1"])

    assert driver.calls[-1][1]["user_id"] == "user-1"
    assert driver.calls[-1][1]["source_types"] == ["pdf"]


def test_neo4j_store_path_query_respects_user_id() -> None:
    store, driver = _store()

    store.find_path_between_concepts("user-1", "A", "B")

    assert driver.calls[-1][1]["user_id"] == "user-1"
