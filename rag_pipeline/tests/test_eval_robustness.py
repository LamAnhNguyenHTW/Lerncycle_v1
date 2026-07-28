from __future__ import annotations

from rag_pipeline.eval.derive_robustness import (
    derive_graph_subset,
    derive_repeated_subset,
)


def _mode(rows, negative=()):
    return {"per_query": list(rows), "negative_controls": list(negative)}


def test_derive_repeated_subset_recomputes_metrics_and_latency() -> None:
    final_row = {
        "query_id": "final-q1",
        "language": "de",
        "question_type": "fact",
        "hit_at_1": 1.0,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "reciprocal_rank": 1.0,
        "precision_at_5": 0.2,
        "recall_at_5": 1.0,
        "ndcg_at_5": 1.0,
        "latency_ms": 20.0,
    }
    pilot_row = {
        **final_row,
        "query_id": "pilot-q1",
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "reciprocal_rank": 0.0,
        "recall_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "latency_ms": 999.0,
    }
    negative = {
        "query_id": "final-q2",
        "language": "de",
        "question_type": "unanswerable",
        "latency_ms": 40.0,
    }
    report = {
        "evaluation": {
            "runs": [{
                "repetition": 1,
                "variant_order": ["variant"],
                "results": {"variant": _mode([pilot_row, final_row], [negative])},
            }]
        }
    }

    subset = derive_repeated_subset(report, {"final-q1", "final-q2"})

    assert subset["query_count"] == 2
    assert subset["answerable_query_count"] == 1
    assert subset["negative_control_query_count"] == 1
    variant = subset["variants"]["variant"]
    assert variant["metrics"]["hit_at_1"]["mean"] == 1.0
    assert variant["latency_ms"]["mean"] == 30.0
    assert variant["latency_ms"]["count"] == 2


def test_derive_graph_subset_filters_pilot_rows() -> None:
    report = {
        "runs": [{
            "results": [
                {
                    "query_id": "pilot-q",
                    "graph_context_available": True,
                    "added_relevant_pages": [1],
                    "relevant_graph_relationships": 9,
                    "vector_latency_ms": 100.0,
                    "graph_latency_ms": 10.0,
                },
                {
                    "query_id": "final-q",
                    "graph_context_available": True,
                    "added_relevant_pages": [2],
                    "relevant_graph_relationships": 3,
                    "vector_latency_ms": 200.0,
                    "graph_latency_ms": 20.0,
                },
            ]
        }]
    }

    subset = derive_graph_subset(report, {"final-q"})

    assert subset["questions"] == 1
    assert subset["observations"] == 1
    assert subset["relevant_graph_relationships"] == 3
    assert subset["query_ids_with_added_relevant_pages"] == ["final-q"]

