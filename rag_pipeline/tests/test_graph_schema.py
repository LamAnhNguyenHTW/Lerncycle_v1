from __future__ import annotations

import pytest

from rag_pipeline.graph_schema import (
    GraphEdge,
    GraphExtraction,
    GraphNode,
    normalize_node_name,
    normalize_node_type,
    normalize_relation_type,
)


def test_normalize_node_name() -> None:
    assert normalize_node_name("  Process   Mining ") == "process mining"


def test_normalize_relation_type() -> None:
    assert normalize_relation_type("Depends On / Uses") == "depends_on_uses"


def test_graph_extraction_rejects_empty_node_name() -> None:
    with pytest.raises(ValueError, match="name"):
        GraphNode(name="")


def test_graph_extraction_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        GraphNode(name="A", confidence=1.2)


def test_graph_extraction_rejects_edge_with_unknown_node() -> None:
    with pytest.raises(ValueError, match="unknown node"):
        GraphExtraction(
            nodes=[GraphNode(name="A")],
            edges=[GraphEdge(source="A", target="B", relation_type="related_to")],
        )


def test_graph_extraction_rejects_self_edge() -> None:
    with pytest.raises(ValueError, match="self"):
        GraphEdge(source="A", target=" a ", relation_type="related_to")


def test_graph_extraction_deduplicates_nodes() -> None:
    extraction = GraphExtraction(
        nodes=[GraphNode(name="Process Mining"), GraphNode(name=" process mining ")],
        edges=[],
    )

    assert len(extraction.nodes) == 1


def test_graph_extraction_deduplicates_edges() -> None:
    extraction = GraphExtraction(
        nodes=[GraphNode(name="A"), GraphNode(name="B")],
        edges=[
            GraphEdge(source="A", target="B", relation_type="Related To"),
            GraphEdge(source=" a ", target=" b ", relation_type="related_to"),
        ],
    )

    assert len(extraction.edges) == 1


def test_normalize_node_type_lowercases() -> None:
    assert normalize_node_type("Concept") == "concept"
    assert normalize_node_type("PROCESS") == "process"


def test_normalize_node_type_aliases() -> None:
    assert normalize_node_type("Person") == "person_role"
    assert normalize_node_type("Course Module") == "other"
    assert normalize_node_type("source") == "other"


def test_normalize_node_type_unknown_falls_back_to_other() -> None:
    assert normalize_node_type("SomethingWeird") == "other"


def test_from_payload_normalizes_node_types() -> None:
    extraction = GraphExtraction.from_payload(
        {
            "nodes": [
                {"name": "Deckungsbeitrag", "node_type": "Concept"},
                {"name": "Karl Marx", "node_type": "Person"},
                {"name": "Modul 1", "node_type": "Course Module"},
            ],
            "edges": [],
        }
    )
    types = {n.normalized_name: n.node_type for n in extraction.nodes}
    assert types["deckungsbeitrag"] == "concept"
    assert types["karl marx"] == "person_role"
    assert types["modul 1"] == "other"


def test_graph_extraction_accepts_valid_payload() -> None:
    extraction = GraphExtraction.from_payload(
        {
            "nodes": [
                {"name": "Process Mining", "node_type": "concept"},
                {"name": "Event Log", "node_type": "data_object"},
            ],
            "edges": [
                {
                    "source": "Process Mining",
                    "target": "Event Log",
                    "relation_type": "uses",
                    "confidence": 0.9,
                }
            ],
        }
    )

    assert extraction.nodes[0].normalized_name == "process mining"
    assert extraction.edges[0].relation_type == "uses"
