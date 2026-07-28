"""Run repeated chunking, retrieval, Cross-Encoder, or LLM comparisons."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embeddings import Embedder
from rag_pipeline.eval.ground_truth import load_ground_truth
from rag_pipeline.eval.index_corpus import STRATEGIES, STRATEGY_COLLECTIONS
from rag_pipeline.eval.repeated import run_repeated_evaluation
from rag_pipeline.evaluate_retrieval import eval_config_with_caches_disabled, write_report
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.reranker import BaseReranker, create_reranker
from rag_pipeline.retrieval import search_chunks, search_hybrid_chunks, search_sparse_chunks
from rag_pipeline.sparse_embeddings import SparseEmbedder


RetrievalFn = Callable[..., list[dict[str, Any]]]
DEFAULT_CROSS_ENCODER = "jinaai/jina-reranker-v2-base-multilingual"
RERANKER_PROVIDERS = ("fastembed", "llm")


def make_phase_searchers(
    phase: str,
    *,
    config: Any,
    user_id: str,
    top_k: int,
    store: Any,
    embedder: Any,
    sparse_embedder: Any,
    reranker: BaseReranker | None = None,
    candidate_k: int = 30,
    base_retrieval: str = "hybrid",
    reranked_label: str = "cross_encoder_reranking",
    dense_fn: RetrievalFn = search_chunks,
    sparse_fn: RetrievalFn = search_sparse_chunks,
    hybrid_fn: RetrievalFn = search_hybrid_chunks,
) -> dict[str, Callable[[str], list[dict[str, Any]]]]:
    """Build searchers with identical scope, store, models, and cache config."""
    common = {
        "user_id": user_id,
        "source_types": ["pdf"],
        "config": config,
        "store": store,
    }
    dense = lambda query: dense_fn(
        query, top_k=top_k, embedder=embedder, **common
    )
    sparse = lambda query: sparse_fn(
        query, top_k=top_k, sparse_embedder=sparse_embedder, **common
    )
    hybrid = lambda query: hybrid_fn(
        query,
        top_k=top_k,
        prefetch_limit=config.hybrid_prefetch_limit,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        **common,
    )
    if phase == "retrieval":
        return {"dense": dense, "sparse": sparse, "hybrid": hybrid}
    if phase == "reranking":
        if reranker is None:
            raise ValueError("reranker is required for the reranking phase")
        if base_retrieval not in {"dense", "sparse", "hybrid"}:
            raise ValueError("base_retrieval must be dense, sparse, or hybrid")

        baselines = {"dense": dense, "sparse": sparse, "hybrid": hybrid}

        def reranked(query: str) -> list[dict[str, Any]]:
            if base_retrieval == "dense":
                candidates = dense_fn(
                    query, top_k=candidate_k, embedder=embedder, **common
                )
            elif base_retrieval == "sparse":
                candidates = sparse_fn(
                    query,
                    top_k=candidate_k,
                    sparse_embedder=sparse_embedder,
                    **common,
                )
            else:
                candidates = hybrid_fn(
                    query,
                    top_k=candidate_k,
                    prefetch_limit=max(config.hybrid_prefetch_limit, candidate_k),
                    embedder=embedder,
                    sparse_embedder=sparse_embedder,
                    **common,
                )
            return reranker.rerank(query, candidates, top_k=top_k)

        return {
            "without_reranking": baselines[base_retrieval],
            reranked_label: reranked,
        }
    raise ValueError("phase must be retrieval or reranking")


def _hybrid_searcher(
    *,
    config: WorkerConfig,
    user_id: str,
    top_k: int,
    store: QdrantStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
) -> Callable[[str], list[dict[str, Any]]]:
    return lambda query: search_hybrid_chunks(
        query,
        user_id=user_id,
        source_types=["pdf"],
        top_k=top_k,
        prefetch_limit=config.hybrid_prefetch_limit,
        config=config,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        store=store,
    )


def _safe_config(
    config: WorkerConfig,
    *,
    phase: str,
    top_k: int,
    repetitions: int,
    collection_strategy: str | None,
    reranker_model: str | None,
    reranker_provider: str,
    base_retrieval: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "top_k": top_k,
        "repetitions": repetitions,
        "warmup_enabled": True,
        "order_rotation": "cyclic",
        "collection_strategy": collection_strategy,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "sparse_provider": config.sparse_provider,
        "sparse_model": config.sparse_model,
        "hybrid_fusion": config.hybrid_fusion,
        "hybrid_prefetch_limit": config.hybrid_prefetch_limit,
        "qdrant_native_hybrid_enabled": config.qdrant_native_hybrid_enabled,
        "reranker_provider": reranker_provider if phase == "reranking" else "disabled",
        "reranker_model": reranker_model if phase == "reranking" else None,
        "base_retrieval": base_retrieval if phase == "reranking" else None,
        "reranking_candidate_k": config.reranking_candidate_k,
        "caches_disabled": True,
    }


def run_comparison(
    *,
    phase: str,
    queries_path: Path,
    user_id: str,
    repetitions: int,
    top_k: int,
    collection_strategy: str | None,
    reranker_model: str,
    reranker_provider: str,
    base_retrieval: str,
) -> dict[str, Any]:
    """Construct real clients and execute one repeated comparison phase."""
    base_config = WorkerConfig.from_env()
    config = replace(
        eval_config_with_caches_disabled(base_config),
        reranking_enabled=False,
    )
    queries = load_ground_truth(queries_path)
    embedder = Embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        openai_api_key=config.openai_api_key,
        gemini_api_key=config.gemini_api_key,
        batch_size=config.embedding_batch_size,
    )
    sparse_embedder = SparseEmbedder(
        provider=config.sparse_provider,
        model=config.sparse_model,
    )

    if phase == "chunking":
        searchers = {
            strategy: _hybrid_searcher(
                config=config,
                user_id=user_id,
                top_k=top_k,
                store=QdrantStore(
                    url=config.qdrant_url,
                    api_key=config.qdrant_api_key,
                    collection_name=STRATEGY_COLLECTIONS[strategy],
                ),
                embedder=embedder,
                sparse_embedder=sparse_embedder,
            )
            for strategy in STRATEGIES
        }
    else:
        if collection_strategy not in STRATEGIES:
            raise ValueError("collection_strategy is required for this phase")
        store = QdrantStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=STRATEGY_COLLECTIONS[collection_strategy],
        )
        reranker = (
            create_reranker(reranker_provider, reranker_model, enabled=True)
            if phase == "reranking"
            else None
        )
        searchers = make_phase_searchers(
            phase,
            config=config,
            user_id=user_id,
            top_k=top_k,
            store=store,
            embedder=embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            candidate_k=config.reranking_candidate_k,
            base_retrieval=base_retrieval,
            reranked_label=(
                "cross_encoder_reranking"
                if reranker_provider == "fastembed"
                else "llm_reranking"
            ),
        )

    evaluation = run_repeated_evaluation(
        queries,
        searchers,
        repetitions=repetitions,
        k=top_k,
        warmup=True,
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": f"{phase}_comparison",
        "ground_truth_path": queries_path.as_posix(),
        "configuration": _safe_config(
            config,
            phase=phase,
            top_k=top_k,
            repetitions=repetitions,
            collection_strategy=collection_strategy,
            reranker_model=reranker_model,
            reranker_provider=reranker_provider,
            base_retrieval=base_retrieval,
        ),
        "evaluation": evaluation,
    }


def _inject_system_truststore() -> None:
    """Use the OS trust store for local Windows HTTPS without disabling TLS."""
    try:
        from pip._vendor import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["chunking", "retrieval", "reranking"], required=True)
    parser.add_argument("--queries", type=Path, default=Path("rag_pipeline/eval/queries_pilot.json"))
    parser.add_argument("--user-id", default="eval-pilot-v1")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection-strategy", choices=list(STRATEGIES))
    parser.add_argument("--reranker-model", default=DEFAULT_CROSS_ENCODER)
    parser.add_argument(
        "--reranker-provider",
        choices=RERANKER_PROVIDERS,
        default="fastembed",
    )
    parser.add_argument(
        "--base-retrieval",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    _inject_system_truststore()
    report = run_comparison(
        phase=args.phase,
        queries_path=args.queries,
        user_id=args.user_id,
        repetitions=args.repetitions,
        top_k=args.top_k,
        collection_strategy=args.collection_strategy,
        reranker_model=args.reranker_model,
        reranker_provider=args.reranker_provider,
        base_retrieval=args.base_retrieval,
    )
    write_report(report, args.out)
    print(args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
