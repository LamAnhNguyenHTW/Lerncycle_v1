from __future__ import annotations

from rag_pipeline.learning_structure.models import ChunkForExtraction, LearningTreeNode
import pytest

from rag_pipeline.learning_structure.neo4j_store import LearningGraphValidationError, ensure_constraints, write_learning_graph


class FakeSession:
    def __init__(self, driver) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, statement, parameters=None):
        self.driver.calls.append((statement, parameters or {}))
        return []


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []

    def session(self, database=None):
        self.database = database
        return FakeSession(self)


def _tree() -> LearningTreeNode:
    concept = LearningTreeNode(
        id="concept-1",
        label="Event Log",
        type="concept",
        summary="Definition",
        confidence=0.8,
        chunk_ids=["chunk-1"],
        children=[],
    )
    objective = LearningTreeNode(
        id="objective-1",
        label="Explain event logs",
        type="objective",
        confidence=0.7,
        chunk_ids=["chunk-1"],
        children=[],
    )
    topic = LearningTreeNode(
        id="topic-1",
        label="Process Mining",
        type="topic",
        summary="Process mining teaches how event data supports discovery and improvement of process behavior.",
        page_start=1,
        page_end=1,
        confidence=0.9,
        order_index=0,
        chunk_ids=["chunk-1"],
        children=[concept, objective],
    )
    return LearningTreeNode(id="document:source-1", label="Document", type="document", chunk_ids=[], children=[topic])


def _chunks() -> list[ChunkForExtraction]:
    return [
        ChunkForExtraction(
            chunk_id="chunk-1",
            text="not stored in neo4j",
            page_index=1,
            heading_path=["Process Mining"],
            content_hash="hash-1",
        )
    ]


def test_ensure_constraints_reuses_composite_chunk_constraint() -> None:
    driver = FakeDriver()

    ensure_constraints(driver)

    joined = "\n".join(statement for statement, _ in driver.calls)
    assert "LearningTopic" in joined
    assert "LearningConcept" in joined
    assert "LearningObjective" in joined
    assert "REQUIRE (c.user_id, c.chunk_id) IS UNIQUE" in joined
    assert "REQUIRE c.chunk_id IS UNIQUE" not in joined


def test_write_learning_graph_deletes_only_learning_scope_and_merges_shared_nodes() -> None:
    driver = FakeDriver()

    write_learning_graph(
        "user-1",
        {"source_id": "source-1", "source_type": "pdf", "title": "Doc"},
        _tree(),
        _chunks(),
        driver=driver,
    )

    joined = "\n".join(statement for statement, _ in driver.calls)
    assert "MATCH ()-[r {graph_scope: 'learning_structure'}]->()" in joined
    assert "LearningTopic OR n:LearningConcept OR n:LearningObjective" in joined
    assert "n:Document" not in joined
    assert "n:Chunk" not in joined
    assert "n:Concept" not in joined
    assert "MERGE (doc:Document" in joined
    assert "MERGE (chunk:Chunk {user_id: $user_id, chunk_id: chunk_payload.chunk_id})" in joined
    assert "DETACH DELETE" in joined
    assert all(params.get("user_id") in {"user-1", None} for _, params in driver.calls)


def test_write_learning_graph_tags_relationships_and_requires_evidence() -> None:
    driver = FakeDriver()

    write_learning_graph(
        "user-1",
        {"source_id": "source-1", "source_type": "pdf", "title": "Doc"},
        _tree(),
        _chunks(),
        driver=driver,
    )

    joined = "\n".join(statement for statement, _ in driver.calls)
    assert "HAS_LEARNING_TOPIC" in joined
    assert "HAS_LEARNING_CONCEPT" in joined
    assert "HAS_OBJECTIVE" in joined
    assert "SUPPORTED_BY" in joined
    assert "graph_scope: 'learning_structure'" in joined
    node_payloads = [params.get("node") for _, params in driver.calls if params.get("node")]
    assert all(payload["chunk_ids"] for payload in node_payloads)
    topic_payloads = [payload for payload in node_payloads if payload["title"]]
    assert topic_payloads[0]["level"] == "topic"


def test_write_learning_graph_sets_subtopic_level() -> None:
    driver = FakeDriver()
    tree = _tree()
    subtopic = LearningTreeNode(
        id="subtopic-1",
        label="Event Logs",
        type="subtopic",
        summary="Event logs teach how recorded process events become evidence for process mining analysis.",
        confidence=0.8,
        chunk_ids=["chunk-1"],
        children=[],
    )
    tree.children[0].children.insert(0, subtopic)

    write_learning_graph("user-1", {"source_id": "source-1", "source_type": "pdf", "title": "Doc"}, tree, _chunks(), driver=driver)

    topic_payloads = [params["node"] for _, params in driver.calls if params.get("node", {}).get("title")]
    assert [payload["level"] for payload in topic_payloads] == ["topic", "subtopic"]


def test_write_learning_graph_rejects_invalid_tree_before_mutation() -> None:
    driver = FakeDriver()
    tree = _tree()
    tree.children[0] = tree.children[0].model_copy(update={"chunk_ids": []})

    with pytest.raises(LearningGraphValidationError):
        write_learning_graph("user-1", {"source_id": "source-1", "source_type": "pdf", "title": "Doc"}, tree, _chunks(), driver=driver)

    assert driver.calls == []


def test_write_learning_graph_rejects_shape_level_mismatch_before_mutation() -> None:
    driver = FakeDriver()
    tree = _tree()
    tree.children[0] = tree.children[0].model_copy(update={"type": "subtopic"})

    with pytest.raises(LearningGraphValidationError):
        write_learning_graph("user-1", {"source_id": "source-1", "source_type": "pdf", "title": "Doc"}, tree, _chunks(), driver=driver)

    assert driver.calls == []


def test_write_learning_graph_is_statement_idempotent_for_same_input() -> None:
    first = FakeDriver()
    second = FakeDriver()
    args = ("user-1", {"source_id": "source-1", "source_type": "pdf", "title": "Doc"}, _tree(), _chunks())

    write_learning_graph(*args, driver=first)
    write_learning_graph(*args, driver=second)

    assert _without_updated_at(first.calls) == _without_updated_at(second.calls)


def _without_updated_at(calls):
    normalized = []
    for statement, params in calls:
        params = dict(params)
        params.pop("updated_at", None)
        if isinstance(params.get("node"), dict):
            node = dict(params["node"])
            node.pop("updated_at", None)
            params["node"] = node
        normalized.append((statement, params))
    return normalized
