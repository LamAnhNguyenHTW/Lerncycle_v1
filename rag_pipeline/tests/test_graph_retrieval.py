from __future__ import annotations

from rag_pipeline.graph_retrieval import detect_graph_intent, retrieve_graph_context


class FakeGraphStore:
    def __init__(self) -> None:
        self.calls = []

    def search_concepts(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [{"name": "Process Mining", "normalized_name": "process mining"}]

    def get_neighborhood(self, **kwargs):
        self.calls.append(("neighborhood", kwargs))
        return {
            "relationships": [
                {
                    "source": "Process Mining",
                    "target": "Event Logs",
                    "relation_type": "uses",
                    "description": "uses event logs as input",
                    "chunk_id": "chunk-1",
                    "page_index": 2,
                    "source_type": "pdf",
                    "source_id": "pdf-1",
                    "snippet": "Process Mining uses Event Logs.",
                }
            ]
        }


def test_detect_graph_intent_relationship_question() -> None:
    assert detect_graph_intent("Wie hängt Process Mining mit Event Logs zusammen?")


def test_detect_graph_intent_mindmap_question() -> None:
    assert detect_graph_intent("Erstelle eine Konzeptkarte zu Process Mining")


def test_detect_graph_intent_normal_factual_false() -> None:
    assert detect_graph_intent("Was ist Process Mining?") is False


def test_retrieve_graph_context_filters_user_id() -> None:
    store = FakeGraphStore()

    retrieve_graph_context("zusammenhang", "user-1", graph_store=store)

    assert store.calls[0][1]["user_id"] == "user-1"


def test_retrieve_graph_context_respects_source_scope() -> None:
    store = FakeGraphStore()

    retrieve_graph_context(
        "zusammenhang",
        "user-1",
        source_types=["pdf"],
        source_ids=["pdf-1"],
        graph_store=store,
    )

    assert store.calls[0][1]["source_types"] == ["pdf"]
    assert store.calls[0][1]["source_ids"] == ["pdf-1"]
    assert store.calls[1][1]["source_types"] == ["pdf"]
    assert store.calls[1][1]["source_ids"] == ["pdf-1"]


def test_retrieve_graph_context_returns_context_sources_nodes_edges() -> None:
    response = retrieve_graph_context("zusammenhang", "user-1", graph_store=FakeGraphStore())

    assert "Process Mining --uses--> Event Logs" in response["context_text"]
    assert response["sources"][0]["source_type"] == "knowledge_graph"
    assert response["nodes"]
    assert response["relationships"]


def test_retrieve_graph_context_empty_when_no_matches() -> None:
    class EmptyStore(FakeGraphStore):
        def search_concepts(self, **kwargs):
            return []

    response = retrieve_graph_context("zusammenhang", "user-1", graph_store=EmptyStore())

    assert response["context_text"] == ""
    assert response["sources"] == []


def test_retrieve_graph_context_caps_max_chars() -> None:
    response = retrieve_graph_context("zusammenhang", "user-1", max_chars=20, graph_store=FakeGraphStore())

    assert len(response["context_text"]) <= 20


def test_retrieve_graph_context_does_not_return_raw_neo4j_objects() -> None:
    response = retrieve_graph_context("zusammenhang", "user-1", graph_store=FakeGraphStore())

    assert "path" not in response
