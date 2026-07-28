"""Evaluate dense, sparse, hybrid, and reranked retrieval against expected hits.

Two layers live here:
  - The original lightweight `evaluate_queries` / `create_searchers` API (Hit@K,
    MRR over an expected-field dict). Kept unchanged for backward compatibility.
  - A metric-rich labeled layer (`evaluate_labeled_queries`, `build_eval_report`,
    `write_report`) that adds Recall@5, Precision@5, nDCG@5, per-query latency
    with p50/p95, per-language / per-question-type breakdowns, and reproducible
    JSON reports labeled with strategy, collection, and configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from rag_pipeline.config import WorkerConfig
from rag_pipeline.eval.ground_truth import GroundTruthQuery, load_ground_truth
from rag_pipeline.eval.metrics import aggregate_metrics, breakdown_by, evaluate_query
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
    retrieval_kwargs: dict[str, Any] | None = None,
) -> dict[str, SearchFn]:
    """Build retrieval mode callables for CLI and tests.

    `retrieval_kwargs` is forwarded to every underlying retrieval call. Real
    runs pass e.g. ``{"config": eval_config_with_caches_disabled(cfg),
    "store": <eval-collection store>}`` so per-query latency is measured without
    cache hits and against the correct evaluation collection.
    """
    selected_modes = _selected_modes(mode)
    extra = dict(retrieval_kwargs or {})
    searchers: dict[str, SearchFn] = {}
    if "dense" in selected_modes:
        searchers["dense"] = lambda query: dense_fn(
            query,
            user_id=user_id,
            source_types=source_types,
            top_k=top_k,
            pdf_ids=pdf_ids,
            **extra,
        )
    if "sparse" in selected_modes:
        searchers["sparse"] = lambda query: sparse_fn(
            query,
            user_id=user_id,
            source_types=source_types,
            top_k=top_k,
            pdf_ids=pdf_ids,
            **extra,
        )
    if "hybrid" in selected_modes:
        searchers["hybrid"] = lambda query: hybrid_fn(
            query,
            user_id=user_id,
            source_types=source_types,
            top_k=top_k,
            pdf_ids=pdf_ids,
            **extra,
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
                **extra,
            ),
            top_k=top_k,
        )
    return searchers


def eval_config_with_caches_disabled(base_config: WorkerConfig) -> WorkerConfig:
    """Return a config copy with all retrieval caches disabled for clean timing.

    Real evaluation runs must not serve repeated queries from the process-local
    query-embedding, retrieval-result, or reranker caches, otherwise latency and
    ordering measurements are contaminated.
    """
    from dataclasses import replace

    return replace(
        base_config,
        query_embedding_cache_enabled=False,
        retrieval_result_cache_enabled=False,
        reranker_cache_enabled=False,
    )


def evaluate_labeled_queries(
    queries: list[GroundTruthQuery],
    searchers: dict[str, SearchFn],
    *,
    k: int = 5,
    capture_latency: bool = True,
) -> dict[str, dict[str, Any]]:
    """Evaluate labeled ground-truth queries per retrieval mode.

    For each mode returns per-query metrics, an aggregate (Hit@K, MRR,
    Precision@k, Recall@k, nDCG@k, latency percentiles), and breakdowns by
    language and question type.

    Args:
        queries: Labeled queries with at least one relevance signal.
        searchers: Mode name to a callable taking the query string and returning
            ranked hit dicts.
        k: Cutoff for @k metrics.
        capture_latency: Measure wall-clock time per searcher call.
    """
    results: dict[str, dict[str, Any]] = {}
    for mode, searcher in searchers.items():
        per_query: list[dict[str, Any]] = []
        negative_controls: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        for query in queries:
            latency_ms: float | None = None
            if capture_latency:
                start = time.perf_counter()
                hits = searcher(query.question)
                latency_ms = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(latency_ms)
            else:
                hits = searcher(query.question)
            if query.has_relevance_labels():
                row = evaluate_query(query, hits, k=k)
                row["retrieved_hits"] = _summarize_hits(hits, k=k)
                if latency_ms is not None:
                    row["latency_ms"] = latency_ms
                per_query.append(row)
                continue

            negative_row: dict[str, Any] = {
                "query_id": query.query_id,
                "language": query.language,
                "question_type": query.question_type,
                "num_retrieved": len(hits),
                "returned_any": bool(hits),
                "metric_status": "not_applicable_no_relevance_labels",
                "retrieved_hits": _summarize_hits(hits, k=k),
            }
            if latency_ms is not None:
                negative_row["latency_ms"] = latency_ms
            negative_controls.append(negative_row)
        aggregate = aggregate_metrics(
            per_query, latencies_ms if capture_latency else None, k=k
        )
        aggregate["total_queries"] = len(queries)
        aggregate["negative_control_queries"] = len(negative_controls)
        results[mode] = {
            "aggregate": aggregate,
            "by_language": breakdown_by(per_query, "language", k=k),
            "by_question_type": breakdown_by(per_query, "question_type", k=k),
            "per_query": per_query,
            "negative_controls": negative_controls,
        }
    return results


def _summarize_hits(
    hits: list[dict[str, Any]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    """Return audit-safe ranking metadata without persisting chunk text."""
    summarized: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits[:k], start=1):
        page_index = hit.get("page_index")
        score = hit.get("score")
        summarized.append(
            {
                "rank": rank,
                "source_id": (
                    str(hit["source_id"]) if hit.get("source_id") is not None else None
                ),
                "page": int(page_index) + 1 if page_index is not None else None,
                "chunk_id": (
                    str(hit["chunk_id"]) if hit.get("chunk_id") is not None else None
                ),
                "score": float(score) if score is not None else None,
            }
        )
    return summarized


def build_eval_report(
    *,
    mode_results: dict[str, dict[str, Any]],
    strategy: str,
    collection: str,
    config: dict[str, Any],
    k: int = 5,
    ground_truth_path: str | None = None,
) -> dict[str, Any]:
    """Wrap per-mode results with reproducibility metadata.

    `config` must contain only non-secret settings (embedding model, top_k,
    cache flags, ...). Secret-like keys and credentialed URLs are rejected
    before the report is built. Secret values are never included in errors.
    """
    _validate_non_secret_config(config)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "collection": collection,
        "k": k,
        "ground_truth_path": ground_truth_path,
        "config": config,
        "modes": mode_results,
    }


_SECRET_KEY_FRAGMENTS = (
    'api_key',
    'apikey',
    'credential',
    'password',
    'secret',
    'service_role',
    'token',
)


def _validate_non_secret_config(value: Any, *, path: str = 'config') -> None:
    '''Reject credentials before reproducibility metadata reaches a report.'''
    if isinstance(value, dict):
        for raw_key, nested_value in value.items():
            key = str(raw_key)
            normalized_key = key.lower().replace('-', '_')
            if any(fragment in normalized_key for fragment in _SECRET_KEY_FRAGMENTS):
                raise ValueError(
                    f'Evaluation report {path}.{key} must contain only non-secret settings.'
                )
            _validate_non_secret_config(nested_value, path=f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _validate_non_secret_config(nested_value, path=f'{path}[{index}]')
        return
    if isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc and (
            parsed.username is not None or parsed.password is not None
        ):
            raise ValueError(
                f'Evaluation report {path} must contain only non-secret settings.'
            )


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write an evaluation report to JSON (UTF-8, pretty-printed)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path


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
