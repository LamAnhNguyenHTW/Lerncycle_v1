"""Tests for repeated, order-rotated retrieval evaluation."""

from __future__ import annotations

from rag_pipeline.eval.ground_truth import parse_query
from rag_pipeline.eval.repeated import run_repeated_evaluation


def _queries():
    return [
        parse_query(
            {
                "query_id": "q1",
                "question": "first",
                "language": "de",
                "question_type": "fact",
                "expected_source_ids": ["pdf-1"],
            }
        ),
        parse_query(
            {
                "query_id": "q2",
                "question": "second",
                "language": "en",
                "question_type": "semantic_paraphrase",
                "expected_source_ids": ["pdf-2"],
            }
        ),
    ]


def test_repeated_evaluation_warms_rotates_and_aggregates() -> None:
    calls: list[tuple[str, str]] = []

    def searcher(label):
        def search(question):
            calls.append((label, question))
            source = "pdf-1" if question == "first" else "pdf-2"
            return [{"source_id": source}]

        return search

    report = run_repeated_evaluation(
        _queries(),
        {"a": searcher("a"), "b": searcher("b")},
        repetitions=3,
        k=5,
        warmup=True,
    )

    assert report["repetitions"] == 3
    assert report["warmup_enabled"] is True
    assert set(report["variants"]) == {"a", "b"}
    assert len(report["runs"]) == 3
    assert report["variants"]["a"]["metrics"]["hit_at_5"]["mean"] == 1.0
    assert report["variants"]["a"]["metrics"]["hit_at_5"]["stddev"] == 0.0
    assert report["variants"]["a"]["latency_ms"]["count"] == 6
    # 2 warm-ups plus 2 variants * 2 queries * 3 repetitions.
    assert len(calls) == 14
    assert report["runs"][0]["variant_order"] == ["a", "b"]
    assert report["runs"][1]["variant_order"] == ["b", "a"]
    assert report["runs"][0]["query_order"] == ["q1", "q2"]
    assert report["runs"][1]["query_order"] == ["q2", "q1"]
