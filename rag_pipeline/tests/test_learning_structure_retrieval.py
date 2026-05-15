from __future__ import annotations

from rag_pipeline.learning_structure.retrieval import get_document_learning_tree


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
        return [FakeRecord(tree=self.driver.tree)] if self.driver.tree else []


class FakeDriver:
    def __init__(self, tree=None) -> None:
        self.tree = tree
        self.calls = []

    def session(self, database=None):
        return FakeSession(self)


def test_get_document_learning_tree_filters_by_user_source_and_scope() -> None:
    driver = FakeDriver(
        tree={
            "id": "document:source-1",
            "label": "Document",
            "type": "document",
            "chunk_ids": [],
            "children": [
                {
                    "id": "topic-1",
                    "label": "Process Mining",
                    "type": "topic",
                    "chunk_ids": ["chunk-1"],
                    "children": [],
                }
            ],
        }
    )

    tree = get_document_learning_tree("user-1", "source-1", driver=driver)

    assert tree is not None
    assert tree.children[0].label == "Process Mining"
    statement, params = driver.calls[0]
    assert "graph_scope: 'learning_structure'" in statement
    assert "user_id: $user_id" in statement
    assert params == {"user_id": "user-1", "source_id": "source-1"}


def test_get_document_learning_tree_returns_none_when_missing() -> None:
    assert get_document_learning_tree("user-2", "source-1", driver=FakeDriver()) is None


def test_get_document_learning_tree_builds_hierarchy_from_flat_neo4j_record() -> None:
    driver = FakeDriver()
    driver.tree = None

    class FlatSession(FakeSession):
        def run(self, statement, parameters=None):
            self.driver.calls.append((statement, parameters or {}))
            return [
                FakeRecord(
                    document={
                        "id": "document:source-1",
                        "label": "Document",
                        "type": "document",
                        "chunk_ids": [],
                        "children": [],
                    },
                    roots=[{"id": "topic-1", "order_index": 1}],
                    nodes=[
                        {
                            "id": "topic-1",
                            "title": "Process Mining",
                            "labels": ["LearningTopic"],
                            "confidence": 0.9,
                            "chunk_ids": ["chunk-1"],
                        },
                        {
                            "id": "subtopic-1",
                            "title": "Event Logs",
                            "labels": ["LearningTopic"],
                            "confidence": 0.8,
                            "chunk_ids": ["chunk-2"],
                        },
                        {
                            "id": "concept-1",
                            "name": "XES",
                            "labels": ["LearningConcept"],
                            "confidence": 0.7,
                            "chunk_ids": ["chunk-3"],
                        },
                    ],
                    relationships=[
                        {
                            "source_id": "topic-1",
                            "target_id": "subtopic-1",
                            "type": "HAS_SUBTOPIC",
                            "order_index": 1,
                        },
                        {
                            "source_id": "subtopic-1",
                            "target_id": "concept-1",
                            "type": "HAS_LEARNING_CONCEPT",
                            "order_index": 1,
                        },
                    ],
                )
            ]

    class FlatDriver(FakeDriver):
        def session(self, database=None):
            return FlatSession(self)

    tree = get_document_learning_tree("user-1", "source-1", driver=FlatDriver())

    assert tree is not None
    assert tree.children[0].label == "Process Mining"
    assert tree.children[0].children[0].type == "subtopic"
    assert tree.children[0].children[0].children[0].label == "XES"
