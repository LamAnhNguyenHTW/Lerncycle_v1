"""Tests for evaluation metric computation."""

from __future__ import annotations

import math

from rag_pipeline.eval.ground_truth import parse_query
from rag_pipeline.eval.metrics import (
    aggregate_metrics,
    breakdown_by,
    evaluate_query,
    percentile,
)


def _query(**overrides):
    data = {
        "query_id": "q1",
        "question": "q",
        "language": "de",
        "question_type": "factual",
        "expected_pages": [3],
    }
    data.update(overrides)
    return parse_query(data)


def test_hit_at_k_and_reciprocal_rank() -> None:
    query = _query(expected_pages=[], expected_source_ids=["s2"])
    hits = [{"source_id": "s1"}, {"source_id": "s2"}, {"source_id": "s3"}]

    result = evaluate_query(query, hits, k=5)

    assert result["hit_at_1"] == 0.0
    assert result["hit_at_3"] == 1.0
    assert result["hit_at_5"] == 1.0
    assert result["reciprocal_rank"] == 0.5


def test_no_relevant_hits_zeroes_all_metrics() -> None:
    query = _query(expected_pages=[], expected_source_ids=["s9"])
    hits = [{"source_id": "s1"}, {"source_id": "s2"}]

    result = evaluate_query(query, hits, k=5)

    assert result["hit_at_1"] == 0.0
    assert result["hit_at_5"] == 0.0
    assert result["reciprocal_rank"] == 0.0
    assert result["precision_at_5"] == 0.0
    assert result["recall_at_5"] == 0.0
    assert result["ndcg_at_5"] == 0.0


def test_source_and_page_labels_must_both_match() -> None:
    query = _query(expected_source_ids=["s1"], expected_pages=[3])
    hits = [
        {"source_id": "s1", "page_index": 8},
        {"source_id": "other", "page_index": 2},
        {"source_id": "s1", "page_index": 2},
    ]

    result = evaluate_query(query, hits, k=5)

    assert result["hit_at_1"] == 0.0
    assert result["hit_at_3"] == 1.0
    assert result["reciprocal_rank"] == 1 / 3
    assert result["precision_at_5"] == 1 / 5


def test_recall_over_multiple_relevant_pages() -> None:
    # Two answer pages (7, 8); top-5 covers only page 7 -> recall 0.5.
    query = _query(expected_pages=[7, 8])
    hits = [
        {"page_index": 6},  # page 7 -> relevant
        {"page_index": 0},
        {"page_index": 1},
        {"page_index": 2},
        {"page_index": 3},
    ]

    result = evaluate_query(query, hits, k=5)

    assert result["recall_at_5"] == 0.5
    assert result["precision_at_5"] == 1 / 5


def test_recall_full_when_all_pages_covered() -> None:
    query = _query(expected_pages=[7, 8])
    hits = [{"page_index": 6}, {"page_index": 7}]

    result = evaluate_query(query, hits, k=5)

    assert result["recall_at_5"] == 1.0
    assert result["precision_at_5"] == 2 / 5


def test_precision_counts_relevant_in_topk() -> None:
    query = _query(expected_pages=[], expected_source_ids=["s1"])
    hits = [
        {"source_id": "s1"},
        {"source_id": "s1"},
        {"source_id": "s2"},
        {"source_id": "s2"},
        {"source_id": "s2"},
    ]

    result = evaluate_query(query, hits, k=5)

    assert result["precision_at_5"] == 2 / 5


def test_ndcg_rewards_relevant_near_top() -> None:
    query = _query(expected_pages=[], expected_source_ids=["s1"])
    top_ranked = [{"source_id": "s1"}, {"source_id": "x"}, {"source_id": "y"}]
    bottom_ranked = [{"source_id": "x"}, {"source_id": "y"}, {"source_id": "s1"}]

    ndcg_top = evaluate_query(query, top_ranked, k=5)["ndcg_at_5"]
    ndcg_bottom = evaluate_query(query, bottom_ranked, k=5)["ndcg_at_5"]

    assert ndcg_top == 1.0
    # single relevant hit at rank 3: DCG = 1/log2(4)=0.5, IDCG=1 -> 0.5
    assert math.isclose(ndcg_bottom, 0.5, rel_tol=1e-9)
    assert ndcg_top > ndcg_bottom


def test_ndcg_two_relevant_ideal_ordering() -> None:
    query = _query(expected_pages=[], expected_source_ids=["s1"])
    hits = [{"source_id": "s1"}, {"source_id": "s1"}, {"source_id": "x"}]

    # Both relevant already at the front -> perfect nDCG.
    assert evaluate_query(query, hits, k=5)["ndcg_at_5"] == 1.0


def test_percentile_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert math.isclose(percentile(values, 95), 38.5, rel_tol=1e-9)
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0


def test_aggregate_includes_latency_percentiles() -> None:
    per_query = [
        evaluate_query(_query(expected_source_ids=["s1"], expected_pages=[]),
                       [{"source_id": "s1"}], k=5),
        evaluate_query(_query(query_id="q2", expected_source_ids=["s2"], expected_pages=[]),
                       [{"source_id": "x"}], k=5),
    ]
    agg = aggregate_metrics(per_query, latencies_ms=[100.0, 300.0], k=5)

    assert agg["evaluated_queries"] == 2
    assert agg["hit_at_1"] == 0.5
    assert agg["mrr"] == 0.5
    assert agg["latency_ms"]["p50"] == 200.0
    assert agg["latency_ms"]["max"] == 300.0


def test_breakdown_by_language_and_type() -> None:
    per_query = [
        {"language": "de", "question_type": "factual", "hit_at_1": 1.0,
         "hit_at_3": 1.0, "hit_at_5": 1.0, "reciprocal_rank": 1.0,
         "precision_at_5": 0.2, "recall_at_5": 1.0, "ndcg_at_5": 1.0},
        {"language": "en", "question_type": "table", "hit_at_1": 0.0,
         "hit_at_3": 0.0, "hit_at_5": 0.0, "reciprocal_rank": 0.0,
         "precision_at_5": 0.0, "recall_at_5": 0.0, "ndcg_at_5": 0.0},
    ]
    by_lang = breakdown_by(per_query, "language", k=5)

    assert set(by_lang) == {"de", "en"}
    assert by_lang["de"]["hit_at_1"] == 1.0
    assert by_lang["en"]["hit_at_1"] == 0.0
