"""Tests for vector plus Concept-Graph contribution accounting."""

from __future__ import annotations

from rag_pipeline.eval.graph_comparison import evaluate_graph_contribution
from rag_pipeline.eval.ground_truth import parse_query


def test_graph_contribution_counts_new_relevant_pages() -> None:
    query = parse_query(
        {
            "query_id": "q1",
            "question": "How are A and B related?",
            "language": "en",
            "question_type": "relational",
            "expected_source_ids": ["pdf-1"],
            "expected_pages": [2, 4],
        }
    )
    result = evaluate_graph_contribution(
        query,
        vector_hits=[{"source_id": "pdf-1", "page_index": 1}],
        graph_result={
            "nodes": [{"name": "A"}],
            "relationships": [
                {
                    "source": "A",
                    "target": "B",
                    "relation_type": "RELATED",
                    "source_id": "pdf-1",
                    "page_index": 3,
                    "chunk_id": "c2",
                }
            ],
            "context_text": "A --RELATED--> B",
        },
    )

    assert result["graph_context_available"] is True
    assert result["relevant_graph_relationships"] == 1
    assert result["vector_relevant_pages"] == [2]
    assert result["graph_relevant_pages"] == [4]
    assert result["added_relevant_pages"] == [4]
