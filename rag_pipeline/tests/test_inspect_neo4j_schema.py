from __future__ import annotations

from rag_pipeline.scripts.inspect_neo4j_schema import inspect_schema, render_markdown


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

    def run(self, statement, parameters=None):
        self.driver.calls.append((statement, parameters or {}))
        if statement == "CALL db.labels()":
            return [FakeRecord(label="Concept"), FakeRecord(label="Chunk")]
        if statement == "CALL db.relationshipTypes()":
            return [FakeRecord(relationshipType="RELATED")]
        if statement == "SHOW CONSTRAINTS":
            return [FakeRecord(name="chunk_user_id", labelsOrTypes=["Chunk"])]
        if statement == "SHOW INDEXES":
            return [FakeRecord(name="chunk_source", labelsOrTypes=["Chunk"])]
        if "MATCH (n:`Concept`)" in statement:
            return [FakeRecord(keys=["name", "user_id"])]
        if "MATCH (n:`Chunk`)" in statement:
            return [FakeRecord(keys=["chunk_id", "user_id"])]
        return []


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []

    def session(self, database=None):
        self.database = database
        return FakeSession(self)


def test_inspect_schema_collects_labels_relationships_constraints_indexes_and_properties() -> None:
    driver = FakeDriver()

    inventory = inspect_schema(driver, database="neo4j")

    assert inventory["labels"] == ["Concept", "Chunk"]
    assert inventory["relationship_types"] == ["RELATED"]
    assert inventory["constraints"][0]["name"] == "chunk_user_id"
    assert inventory["indexes"][0]["name"] == "chunk_source"
    assert inventory["property_keys_by_label"] == {
        "Concept": ["name", "user_id"],
        "Chunk": ["chunk_id", "user_id"],
    }
    assert any(call[0] == "CALL db.labels()" for call in driver.calls)
    assert any("MATCH (n:`Concept`)" in call[0] for call in driver.calls)


def test_render_markdown_handles_non_json_native_neo4j_values() -> None:
    class Neo4jLikeDate:
        def __str__(self) -> str:
            return "2026-05-15T10:00:00Z"

    markdown = render_markdown(
        {
            "labels": ["Chunk"],
            "relationship_types": [],
            "constraints": [],
            "indexes": [{"name": "chunk_source", "lastRead": Neo4jLikeDate()}],
            "property_keys_by_label": {"Chunk": ["chunk_id"]},
        }
    )

    assert "2026-05-15T10:00:00Z" in markdown
