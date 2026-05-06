"""Evaluate dense, sparse, and hybrid retrieval against expected sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from rag_pipeline.retrieval import search_chunks
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.retrieval import search_sparse_chunks


SearchFn = Callable[[str], list[dict[str, Any]]]


def evaluate_queries(
    queries: list[dict[str, Any]],
    searchers: dict[str, SearchFn],
) -> dict[str, dict[str, float | int]]:
    """Compute Hit@1/3/5 for each retrieval mode."""
    results: dict[str, dict[str, float | int]] = {}
    evaluable = [
        query for query in queries if query.get("expected_source_id") is not None
    ]
    for mode, searcher in searchers.items():
        hit_counts = {1: 0, 3: 0, 5: 0}
        for item in evaluable:
            hits = searcher(str(item["query"]))
            expected_source_id = str(item["expected_source_id"])
            expected_source_type = item.get("expected_source_type")
            expected_page_index = item.get("expected_page_index")
            for k in hit_counts:
                if _hits_expected(
                    hits[:k],
                    expected_source_id=expected_source_id,
                    expected_source_type=expected_source_type,
                    expected_page_index=expected_page_index,
                ):
                    hit_counts[k] += 1
        denominator = len(evaluable)
        results[mode] = {
            "evaluated_queries": denominator,
            "hit_at_1": _rate(hit_counts[1], denominator),
            "hit_at_3": _rate(hit_counts[3], denominator),
            "hit_at_5": _rate(hit_counts[5], denominator),
        }
    return results


def _hits_expected(
    hits: list[dict[str, Any]],
    expected_source_id: str,
    expected_source_type: str | None,
    expected_page_index: int | None,
) -> bool:
    for hit in hits:
        if str(hit.get("source_id")) != expected_source_id:
            continue
        if expected_source_type and hit.get("source_type") != expected_source_type:
            continue
        if (
            expected_page_index is not None
            and hit.get("page_index") != expected_page_index
        ):
            continue
        return True
    return False


def _rate(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return count / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = json.loads(args.input.read_text(encoding="utf-8"))
    searchers = {
        "dense": lambda query: search_chunks(
            query,
            user_id=args.user_id,
            top_k=args.top_k,
        ),
        "sparse": lambda query: search_sparse_chunks(
            query,
            user_id=args.user_id,
            top_k=args.top_k,
        ),
        "hybrid": lambda query: search_hybrid_chunks(
            query,
            user_id=args.user_id,
            top_k=args.top_k,
        ),
    }
    print(json.dumps(evaluate_queries(queries, searchers), indent=2))


if __name__ == "__main__":
    main()
