"""Run the 15-question chunking pilot against all evaluation collections.

The runner deliberately fixes retrieval to hybrid dense+sparse without
reranking. It executes unanswerable questions as explicit negative controls,
but excludes them from relevance metrics because they have no relevant set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embeddings import Embedder
from rag_pipeline.eval.ground_truth import GroundTruthQuery, load_ground_truth
from rag_pipeline.eval.index_corpus import STRATEGIES, STRATEGY_COLLECTIONS
from rag_pipeline.evaluate_retrieval import (
    build_eval_report,
    create_searchers,
    eval_config_with_caches_disabled,
    evaluate_labeled_queries,
    write_report,
)
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.sparse_embeddings import SparseEmbedder


StoreFactory = Callable[[str], Any]


def _safe_retrieval_configuration(
    config: WorkerConfig,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Return reproducibility settings with no endpoints or credentials."""
    return {
        "mode": "hybrid",
        "reranking_enabled": False,
        "top_k": top_k,
        "hybrid_prefetch_limit": config.hybrid_prefetch_limit,
        "hybrid_fusion": config.hybrid_fusion,
        "qdrant_native_hybrid_enabled": config.qdrant_native_hybrid_enabled,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "sparse_provider": config.sparse_provider,
        "sparse_model": config.sparse_model,
        "caches_disabled": True,
        "query_embedding_cache_enabled": False,
        "retrieval_result_cache_enabled": False,
        "reranker_cache_enabled": False,
    }


def _sanitize_extraction_errors(
    errors: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Copy only allow-listed diagnostic fields into durable reports."""
    safe: list[dict[str, str]] = []
    for error in errors or []:
        safe.append(
            {
                "strategy": str(error.get("strategy") or "unknown"),
                "source_id": str(error.get("source_id") or "unknown"),
                "stage": str(error.get("stage") or "extraction"),
                "error_type": str(error.get("error_type") or "unknown"),
            }
        )
    return safe


def _sanitize_extraction_diagnostics(
    diagnostics: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Allow-list Docling page coverage fields for the pilot report."""
    safe: list[dict[str, Any]] = []
    for item in diagnostics or []:
        safe.append(
            {
                "strategy": str(item.get("strategy") or "unknown"),
                "source_id": str(item.get("source_id") or "unknown"),
                "total_pages": int(item.get("total_pages") or 0),
                "chunked_pages": list(item.get("chunked_pages") or []),
                "missing_pages": list(item.get("missing_pages") or []),
                "fallback_pages": list(item.get("fallback_pages") or []),
                "retry_error_keys": list(item.get("retry_error_keys") or []),
                "page_batch_size": int(item.get("page_batch_size") or 0),
            }
        )
    return safe


def run_pilot_retrieval(
    queries: list[GroundTruthQuery],
    *,
    base_config: WorkerConfig,
    user_id: str,
    top_k: int = 5,
    embedder: Any | None = None,
    sparse_embedder: Any | None = None,
    store_factory: StoreFactory | None = None,
    hybrid_fn: Callable[..., list[dict[str, Any]]] = search_hybrid_chunks,
    extraction_errors: list[dict[str, Any]] | None = None,
    extraction_diagnostics: list[dict[str, Any]] | None = None,
    ground_truth_path: str = "rag_pipeline/eval/queries_pilot.json",
) -> dict[str, Any]:
    """Evaluate the same queries and retrieval config against all strategies."""
    config = replace(
        eval_config_with_caches_disabled(base_config),
        reranking_enabled=False,
    )
    active_embedder = embedder or Embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        openai_api_key=config.openai_api_key,
        gemini_api_key=config.gemini_api_key,
        batch_size=config.embedding_batch_size,
    )
    active_sparse_embedder = sparse_embedder or SparseEmbedder(
        provider=config.sparse_provider,
        model=config.sparse_model,
    )
    make_store = store_factory or (
        lambda collection: QdrantStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=collection,
        )
    )
    retrieval_config = _safe_retrieval_configuration(config, top_k=top_k)
    runs: dict[str, dict[str, Any]] = {}

    for strategy in STRATEGIES:
        collection = STRATEGY_COLLECTIONS[strategy]
        searchers = create_searchers(
            user_id=user_id,
            top_k=top_k,
            mode="hybrid",
            source_types=["pdf"],
            hybrid_fn=hybrid_fn,
            retrieval_kwargs={
                "config": config,
                "store": make_store(collection),
                "embedder": active_embedder,
                "sparse_embedder": active_sparse_embedder,
                "prefetch_limit": config.hybrid_prefetch_limit,
            },
        )
        mode_results = evaluate_labeled_queries(
            queries,
            searchers,
            k=top_k,
            capture_latency=True,
        )
        runs[strategy] = build_eval_report(
            mode_results=mode_results,
            strategy=strategy,
            collection=collection,
            config=retrieval_config,
            k=top_k,
            ground_truth_path=ground_truth_path,
        )

    metric_evaluable = sum(query.has_relevance_labels() for query in queries)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "chunking_pilot",
        "interpretation_status": "pilot_only_no_strategy_decision",
        "query_count": len(queries),
        "metric_evaluable_queries": metric_evaluable,
        "negative_control_queries": len(queries) - metric_evaluable,
        "retrieval_configuration": retrieval_config,
        "extraction_errors": _sanitize_extraction_errors(extraction_errors),
        "extraction_diagnostics": _sanitize_extraction_diagnostics(
            extraction_diagnostics
        ),
        "runs": runs,
    }


def write_pilot_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the combined pilot JSON using the existing safe report writer."""
    return write_report(report, path)


def _load_extraction_errors(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Index stats must be a JSON object.")
    errors = payload.get("extraction_errors", [])
    if not isinstance(errors, list):
        raise ValueError("Index stats extraction_errors must be a list.")
    return [error for error in errors if isinstance(error, dict)]


def _load_extraction_diagnostics(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Index stats must be a JSON object.")
    diagnostics = payload.get("extraction_diagnostics", [])
    if not isinstance(diagnostics, list):
        raise ValueError("Index stats extraction_diagnostics must be a list.")
    return [item for item in diagnostics if isinstance(item, dict)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("rag_pipeline/eval/queries_pilot.json"),
    )
    parser.add_argument("--index-stats", type=Path)
    parser.add_argument("--user-id", default="eval-pilot-v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    queries = load_ground_truth(args.queries)
    config = WorkerConfig.from_env()
    report = run_pilot_retrieval(
        queries,
        base_config=config,
        user_id=args.user_id,
        top_k=args.top_k,
        extraction_errors=_load_extraction_errors(args.index_stats),
        extraction_diagnostics=_load_extraction_diagnostics(args.index_stats),
        ground_truth_path=args.queries.as_posix(),
    )
    output = write_pilot_report(report, args.out)
    print(output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
