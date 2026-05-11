from __future__ import annotations

from typing import Any

from rag_pipeline.evaluate_retrieval import create_searchers
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


def test_mrr_computes_correctly() -> None:
    results = evaluate_queries(
        [
            {"query": "q1", "expected_source_id": "source-2"},
            {"query": "q2", "expected_source_id": "source-3"},
        ],
        searchers={
            "dense": lambda _query: [
                {"source_id": "source-1"},
                {"source_id": "source-2"},
                {"source_id": "source-3"},
            ]
        },
    )

    assert results["dense"]["mrr"] == (1 / 2 + 1 / 3) / 2


def test_mrr_zero_when_no_hits() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_source_id": "source-2"}],
        searchers={"dense": lambda _query: [{"source_id": "source-1"}]},
    )

    assert results["dense"]["mrr"] == 0.0


def test_evaluation_matches_expected_page() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_page": 3}],
        searchers={"dense": lambda _query: [{"page_index": 2}]},
    )

    assert results["dense"]["hit_at_1"] == 1.0


def test_evaluation_matches_expected_contains() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_contains": "Process Mining"}],
        searchers={
            "dense": lambda _query: [{"text": "A short Process Mining definition"}]
        },
    )

    assert results["dense"]["hit_at_1"] == 1.0


def test_evaluation_matches_expected_chunk_id() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_chunk_id": "chunk-2"}],
        searchers={"dense": lambda _query: [{"chunk_id": "chunk-2"}]},
    )

    assert results["dense"]["hit_at_1"] == 1.0


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


def test_evaluation_compares_modes() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_source_id": "source-1"}],
        searchers={
            "dense": lambda _query: [{"source_id": "source-1"}],
            "sparse": lambda _query: [{"source_id": "source-1"}],
            "hybrid": lambda _query: [{"source_id": "source-1"}],
            "hybrid_reranked": lambda _query: [{"source_id": "source-1"}],
        },
    )

    assert set(results) == {"dense", "sparse", "hybrid", "hybrid_reranked"}


def test_evaluation_handles_no_results() -> None:
    results = evaluate_queries(
        [{"query": "q", "expected_source_id": "source-1"}],
        searchers={"dense": lambda _query: []},
    )

    assert results["dense"]["evaluated_queries"] == 1
    assert results["dense"]["hit_at_1"] == 0.0
    assert results["dense"]["hit_at_3"] == 0.0
    assert results["dense"]["hit_at_5"] == 0.0
    assert results["dense"]["mrr"] == 0.0


def test_evaluation_handles_missing_expected_source() -> None:
    results = evaluate_queries(
        [{"query": "q"}],
        searchers={"dense": lambda _query: [{"source_id": "source-1"}]},
    )

    assert results["dense"]["evaluated_queries"] == 0
    assert results["dense"]["hit_at_1"] == 0.0


def test_evaluation_supports_hybrid_reranked_mode() -> None:
    class FakeReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[dict[str, Any]], int]] = []

        def rerank(
            self, query: str, results: list[dict[str, Any]], top_k: int
        ) -> list[dict[str, Any]]:
            self.calls.append((query, results, top_k))
            return list(reversed(results))[:top_k]

    reranker = FakeReranker()
    searchers = create_searchers(
        user_id="user-1",
        top_k=1,
        mode="hybrid_reranked",
        reranker=reranker,
        hybrid_fn=lambda _query, **_kwargs: [
            {"chunk_id": "chunk-1"},
            {"chunk_id": "chunk-2"},
        ],
    )

    results = evaluate_queries(
        [{"query": "q", "expected_chunk_id": "chunk-2"}],
        searchers=searchers,
    )

    assert set(results) == {"hybrid_reranked"}
    assert results["hybrid_reranked"]["hit_at_1"] == 1.0
    assert reranker.calls[0][2] == 1
