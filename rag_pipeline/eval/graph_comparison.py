"""Compare dense vector support with additional production Concept-Graph context."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embeddings import Embedder
from rag_pipeline.eval.ground_truth import GroundTruthQuery, is_hit_relevant, load_ground_truth
from rag_pipeline.eval.index_corpus import STRATEGIES, STRATEGY_COLLECTIONS
from rag_pipeline.eval.metrics import evaluate_query, percentile
from rag_pipeline.evaluate_retrieval import eval_config_with_caches_disabled, write_report
from rag_pipeline.graph_retrieval import retrieve_graph_context
from rag_pipeline.graph_store_factory import create_graph_store
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.retrieval import search_chunks


GRAPH_QUESTION_TYPES = {"relational", "multi_hop"}


def _relevant_pages(
    query: GroundTruthQuery,
    hits: list[dict[str, Any]],
) -> list[int]:
    pages = {
        int(hit["page_index"]) + 1
        for hit in hits
        if hit.get("page_index") is not None and is_hit_relevant(hit, query)
    }
    return sorted(pages)


def evaluate_graph_contribution(
    query: GroundTruthQuery,
    *,
    vector_hits: list[dict[str, Any]],
    graph_result: dict[str, Any],
) -> dict[str, Any]:
    """Measure whether graph relationships add grounded relevant pages."""
    graph_hits = [
        {
            "source_id": relationship.get("source_id"),
            "page_index": relationship.get("page_index"),
            "chunk_id": relationship.get("chunk_id"),
        }
        for relationship in graph_result.get("relationships", [])
    ]
    vector_pages = _relevant_pages(query, vector_hits)
    graph_pages = _relevant_pages(query, graph_hits)
    return {
        "query_id": query.query_id,
        "question_type": query.question_type,
        "language": query.language,
        "graph_context_available": bool(graph_result.get("context_text")),
        "graph_node_count": len(graph_result.get("nodes", [])),
        "graph_relationship_count": len(graph_hits),
        "relevant_graph_relationships": sum(
            1 for hit in graph_hits if is_hit_relevant(hit, query)
        ),
        "vector_relevant_pages": vector_pages,
        "graph_relevant_pages": graph_pages,
        "added_relevant_pages": sorted(set(graph_pages) - set(vector_pages)),
        "relationships": [
            {
                "source": relationship.get("source"),
                "target": relationship.get("target"),
                "relation_type": relationship.get("relation_type"),
                "source_id": relationship.get("source_id"),
                "page": (
                    int(relationship["page_index"]) + 1
                    if relationship.get("page_index") is not None
                    else None
                ),
            }
            for relationship in graph_result.get("relationships", [])
        ],
    }


def _latency(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "count": len(values),
    }


def _inject_system_truststore() -> None:
    try:
        from pip._vendor import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def run_graph_comparison(
    *,
    queries_path: Path,
    strategy: str,
    user_id: str,
    repetitions: int,
    top_k: int,
) -> dict[str, Any]:
    """Run dense vector retrieval and the production graph branch side by side."""
    base = WorkerConfig.from_env()
    config = replace(
        eval_config_with_caches_disabled(base),
        reranking_enabled=False,
    )
    queries = [
        query
        for query in load_ground_truth(queries_path)
        if query.question_type in GRAPH_QUESTION_TYPES
    ]
    embedder = Embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        openai_api_key=config.openai_api_key,
        gemini_api_key=config.gemini_api_key,
        batch_size=config.embedding_batch_size,
    )
    store = QdrantStore(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=STRATEGY_COLLECTIONS[strategy],
    )
    graph_store = create_graph_store(config)
    if graph_store is None:
        raise RuntimeError("Concept Graph store is not enabled.")

    # Excluded warm-up for both branches.
    first = queries[0]
    search_chunks(
        first.question,
        user_id=user_id,
        source_types=["pdf"],
        top_k=top_k,
        config=config,
        embedder=embedder,
        store=store,
    )
    retrieve_graph_context(
        first.question,
        user_id=user_id,
        source_types=["pdf"],
        source_ids=first.expected_source_ids,
        graph_store=graph_store,
    )

    runs: list[dict[str, Any]] = []
    vector_latencies: list[float] = []
    graph_latencies: list[float] = []
    for repetition in range(repetitions):
        offset = repetition % len(queries)
        ordered = queries[offset:] + queries[:offset]
        rows = []
        for query in ordered:
            started = time.perf_counter()
            vector_hits = search_chunks(
                query.question,
                user_id=user_id,
                source_types=["pdf"],
                top_k=top_k,
                config=config,
                embedder=embedder,
                store=store,
            )
            vector_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            graph_result = retrieve_graph_context(
                query.question,
                user_id=user_id,
                source_types=["pdf"],
                source_ids=query.expected_source_ids,
                graph_store=graph_store,
            )
            graph_ms = (time.perf_counter() - started) * 1000.0
            vector_latencies.append(vector_ms)
            graph_latencies.append(graph_ms)
            contribution = evaluate_graph_contribution(
                query,
                vector_hits=vector_hits,
                graph_result=graph_result,
            )
            contribution["vector_metrics"] = evaluate_query(
                query, vector_hits, k=top_k
            )
            contribution["vector_latency_ms"] = vector_ms
            contribution["graph_latency_ms"] = graph_ms
            rows.append(contribution)
        runs.append({"repetition": repetition + 1, "results": rows})

    all_rows = [row for run in runs for row in run["results"]]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "vector_plus_concept_graph",
        "ground_truth_path": queries_path.as_posix(),
        "strategy": strategy,
        "collection": STRATEGY_COLLECTIONS[strategy],
        "configuration": {
            "vector_mode": "dense",
            "graph_mode": "production_retrieve_graph_context",
            "question_types": sorted(GRAPH_QUESTION_TYPES),
            "top_k": top_k,
            "repetitions": repetitions,
            "warmup_enabled": True,
            "caches_disabled": True,
        },
        "summary": {
            "questions": len(queries),
            "observations": len(all_rows),
            "graph_context_available_observations": sum(
                bool(row["graph_context_available"]) for row in all_rows
            ),
            "observations_with_added_relevant_pages": sum(
                bool(row["added_relevant_pages"]) for row in all_rows
            ),
            "relevant_graph_relationships": sum(
                int(row["relevant_graph_relationships"]) for row in all_rows
            ),
            "vector_latency_ms": _latency(vector_latencies),
            "graph_latency_ms": _latency(graph_latencies),
        },
        "runs": runs,
    }
    if hasattr(graph_store, "close"):
        graph_store.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    parser.add_argument("--user-id", default="eval-pilot-v1")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    _inject_system_truststore()
    report = run_graph_comparison(
        queries_path=args.queries,
        strategy=args.strategy,
        user_id=args.user_id,
        repetitions=args.repetitions,
        top_k=args.top_k,
    )
    write_report(report, args.out)
    print(args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
