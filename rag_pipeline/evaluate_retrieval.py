"""Evaluate dense, sparse, hybrid, and reranked retrieval against expected hits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from rag_pipeline.config import WorkerConfig
from rag_pipeline.retrieval import search_chunks
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.retrieval import search_sparse_chunks
from rag_pipeline.reranker import BaseReranker
from rag_pipeline.reranker import create_reranker


SearchFn = Callable[[str], list[dict[str, Any]]]
RetrievalFn = Callable[..., list[dict[str, Any]]]
EXPECTED_FIELDS = (
    "expected_source_id",
    "expected_source_type",
    "expected_page_index",
    "expected_page",
    "expected_contains",
    "expected_chunk_id",
)


def evaluate_queries(
    queries: list[dict[str, Any]],
    searchers: dict[str, SearchFn],
) -> dict[str, dict[str, float | int]]:
    """Compute Hit@1/3/5 and MRR for each retrieval mode."""
    results: dict[str, dict[str, float | int]] = {}
    evaluable = [query for query in queries if _has_expected_fields(query)]
    for mode, searcher in searchers.items():
        hit_counts = {1: 0, 3: 0, 5: 0}
        reciprocal_rank_sum = 0.0
        for item in evaluable:
            hits = searcher(str(item["query"]))
            for k in hit_counts:
                if _hits_expected(hits[:k], item):
                    hit_counts[k] += 1
            reciprocal_rank_sum += _reciprocal_rank(hits, item)
        denominator = len(evaluable)
        results[mode] = {
            "evaluated_queries": denominator,
            "hit_at_1": _rate(hit_counts[1], denominator),
            "hit_at_3": _rate(hit_counts[3], denominator),
            "hit_at_5": _rate(hit_counts[5], denominator),
            "mrr": _rate(reciprocal_rank_sum, denominator),
        }
    return results


def create_searchers(
    *,
    user_id: str,
    top_k: int,
    mode: str = "all",
    source_types: list[str] | None = None,
    pdf_ids: list[str] | None = None,
    reranker: BaseReranker | None = None,
    candidate_k: int = 30,
    dense_fn: RetrievalFn = search_chunks,
    sparse_fn: RetrievalFn = search_sparse_chunks,
    hybrid_fn: RetrievalFn = search_hybrid_chunks,
) -> dict[str, SearchFn]:
    """Build retrieval mode callables for CLI and tests."""
    selected_modes = _selected_modes(mode)
    searchers: dict[str, SearchFn] = {}
    if "dense" in selected_modes:
        searchers["dense"] = lambda query: dense_fn(
            query,
            user_id=user_id,
            source_types=source_types,
            top_k=top_k,
            pdf_ids=pdf_ids,
        )
    if "sparse" in selected_modes:
        searchers["sparse"] = lambda query: sparse_fn(
            query,
            user_id=user_id,
            source_types=source_types,
            top_k=top_k,
            pdf_ids=pdf_ids,
        )
    if "hybrid" in selected_modes:
        searchers["hybrid"] = lambda query: hybrid_fn(
            query,
            user_id=user_id,
            source_types=source_types,
            top_k=top_k,
            pdf_ids=pdf_ids,
        )
    if "hybrid_reranked" in selected_modes:
        active_reranker = reranker or create_reranker(
            provider="noop",
            model="",
            enabled=False,
        )
        searchers["hybrid_reranked"] = lambda query: active_reranker.rerank(
            query,
            hybrid_fn(
                query,
                user_id=user_id,
                source_types=source_types,
                top_k=candidate_k,
                pdf_ids=pdf_ids,
            ),
            top_k=top_k,
        )
    return searchers


def _selected_modes(mode: str) -> list[str]:
    valid_modes = ["dense", "sparse", "hybrid", "hybrid_reranked"]
    if mode == "all":
        return valid_modes
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {', '.join(valid_modes)}, all")
    return [mode]


def _has_expected_fields(item: dict[str, Any]) -> bool:
    return any(item.get(field) is not None for field in EXPECTED_FIELDS)


def _hits_expected(hits: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    for hit in hits:
        if _hit_matches_expected(hit, expected):
            return True
    return False


def _reciprocal_rank(hits: list[dict[str, Any]], expected: dict[str, Any]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if _hit_matches_expected(hit, expected):
            return 1 / rank
    return 0.0


def _hit_matches_expected(hit: dict[str, Any], expected: dict[str, Any]) -> bool:
    expected_source_id = expected.get("expected_source_id")
    if expected_source_id is not None and str(hit.get("source_id")) == str(
        expected_source_id
    ):
        return True
    expected_source_type = expected.get("expected_source_type")
    if expected_source_type is not None and hit.get("source_type") == expected_source_type:
        return True
    expected_page_index = expected.get("expected_page_index")
    if expected_page_index is not None and hit.get("page_index") == expected_page_index:
        return True
    expected_page = expected.get("expected_page")
    if expected_page is not None and hit.get("page_index") == int(expected_page) - 1:
        return True
    expected_contains = expected.get("expected_contains")
    if expected_contains is not None and expected_contains in str(hit.get("text") or ""):
        return True
    expected_chunk_id = expected.get("expected_chunk_id")
    if expected_chunk_id is not None and str(hit.get("chunk_id")) == str(
        expected_chunk_id
    ):
        return True
    return False


def _rate(count: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return count / denominator


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid", "hybrid_reranked", "all"],
        default="all",
    )
    parser.add_argument("--source-types")
    parser.add_argument("--pdf-ids")
    args = parser.parse_args()

    input_path = args.file or args.input
    if input_path is None:
        parser.error("Provide an input path or --file")

    config = WorkerConfig.from_env()
    reranker = create_reranker(
        provider=config.reranking_provider,
        model=config.reranking_model,
        enabled=config.reranking_enabled,
    )
    queries = json.loads(input_path.read_text(encoding="utf-8"))
    searchers = create_searchers(
        user_id=args.user_id,
        top_k=args.top_k,
        mode=args.mode,
        source_types=_split_csv(args.source_types),
        pdf_ids=_split_csv(args.pdf_ids),
        reranker=reranker,
        candidate_k=config.reranking_candidate_k,
    )
    print(json.dumps(evaluate_queries(queries, searchers), indent=2))


if __name__ == "__main__":
    main()
