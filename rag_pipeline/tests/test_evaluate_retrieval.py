from __future__ import annotations

from rag_pipeline.evaluate_retrieval import evaluate_queries


def test_evaluation_computes_hit_at_k() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_source_id": "source-2"}],
        searchers={
            "dense": lambda _query: [
                {"source_id": "source-1"},
                {"source_id": "source-2"},
            ]
        },
    )

    assert results["dense"]["hit_at_1"] == 0.0
    assert results["dense"]["hit_at_3"] == 1.0
    assert results["dense"]["hit_at_5"] == 1.0


def test_evaluation_compares_dense_sparse_hybrid() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_source_id": "source-1"}],
        searchers={
            "dense": lambda _query: [{"source_id": "source-1"}],
            "sparse": lambda _query: [{"source_id": "source-2"}],
            "hybrid": lambda _query: [{"source_id": "source-1"}],
        },
    )

    assert set(results) == {"dense", "sparse", "hybrid"}
    assert results["dense"]["hit_at_1"] == 1.0
    assert results["sparse"]["hit_at_1"] == 0.0
    assert results["hybrid"]["hit_at_1"] == 1.0


def test_evaluation_handles_missing_expected_source() -> None:
    results = evaluate_queries(
        [{"query": "q"}],
        searchers={"dense": lambda _query: [{"source_id": "source-1"}]},
    )

    assert results["dense"]["evaluated_queries"] == 0
    assert results["dense"]["hit_at_1"] == 0.0
