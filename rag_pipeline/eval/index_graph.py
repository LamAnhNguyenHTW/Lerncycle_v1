"""Build an isolated Concept Graph from one Qdrant evaluation collection."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.eval.index_corpus import STRATEGIES, STRATEGY_COLLECTIONS, load_manifest
from rag_pipeline.evaluate_retrieval import write_report
from rag_pipeline.graph_extractor import GraphExtractor
from rag_pipeline.graph_store_factory import create_graph_store


def load_retry_chunk_ids(report_path: Path) -> set[str]:
    """Load only failed chunk IDs from a sanitized prior index report."""
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    failures = raw.get("stats", {}).get("failures", [])
    if not isinstance(failures, list):
        raise ValueError("Retry report must contain stats.failures as a list.")
    chunk_ids = {
        str(item["chunk_id"])
        for item in failures
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
    }
    if not chunk_ids:
        raise ValueError("Retry report contains no failed chunk IDs.")
    return chunk_ids


def index_graph_chunks(
    chunks: list[dict[str, Any]],
    *,
    user_id: str,
    extractor: Any,
    store: Any,
    concurrency: int = 8,
) -> dict[str, Any]:
    """Extract and persist graph data, returning sanitized aggregate diagnostics."""
    started = time.perf_counter()
    chunks_indexed = 0
    nodes_upserted = 0
    relationships_upserted = 0
    failures: list[dict[str, str]] = []

    def process(chunk: dict[str, Any]) -> tuple[dict[str, int] | None, dict[str, str] | None]:
        try:
            extraction = extractor.extract_from_chunk(chunk)
            stats = store.upsert_extraction(user_id, chunk, extraction)
            return stats, None
        except Exception as exc:
            return None, {
                "chunk_id": str(chunk.get("chunk_id") or "unknown"),
                "error_type": type(exc).__name__,
            }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(process, chunk) for chunk in chunks]
        for future in as_completed(futures):
            stats, failure = future.result()
            if failure is not None:
                failures.append(failure)
                continue
            chunks_indexed += 1
            nodes_upserted += int((stats or {}).get("nodes_upserted", 0))
            relationships_upserted += int(
                (stats or {}).get("relationships_upserted", 0)
            )

    return {
        "chunks_requested": len(chunks),
        "chunks_indexed": chunks_indexed,
        "nodes_upserted": nodes_upserted,
        "relationships_upserted": relationships_upserted,
        "failures": sorted(failures, key=lambda item: item["chunk_id"]),
        "indexing_seconds": time.perf_counter() - started,
    }


def _load_qdrant_chunks(
    config: WorkerConfig,
    *,
    collection: str,
    user_id: str,
) -> list[dict[str, Any]]:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            )
        ]
    )
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            chunks.append(
                {
                    "chunk_id": str(payload.get("chunk_id") or point.id),
                    "source_type": payload.get("source_type") or "pdf",
                    "source_id": payload.get("source_id"),
                    "pdf_id": payload.get("pdf_id"),
                    "page_index": payload.get("page_index"),
                    "heading": payload.get("heading"),
                    "text": payload.get("text") or "",
                }
            )
        if offset is None:
            break
    return chunks


def _inject_system_truststore() -> None:
    try:
        from pip._vendor import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--user-id", default="eval-pilot-v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--retry-report",
        type=Path,
        help="Retry only sanitized failed chunk IDs without deleting successful graph data.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    _inject_system_truststore()
    config = WorkerConfig.from_env()
    collection = STRATEGY_COLLECTIONS[args.strategy]
    chunks = _load_qdrant_chunks(
        config,
        collection=collection,
        user_id=args.user_id,
    )
    retry_ids = None
    if args.retry_report is not None:
        retry_ids = load_retry_chunk_ids(args.retry_report)
        chunks = [chunk for chunk in chunks if chunk["chunk_id"] in retry_ids]
        missing = retry_ids - {chunk["chunk_id"] for chunk in chunks}
        if missing:
            raise ValueError("Retry report references chunk IDs outside the collection.")
    store = create_graph_store(config)
    if store is None:
        raise RuntimeError("Concept Graph store is not enabled.")
    if retry_ids is None:
        documents = load_manifest(args.manifest)
        for document in documents:
            store.delete_by_source(args.user_id, document.source_type, document.source_id)

    extractor = GraphExtractor(
        max_nodes=config.graph_max_nodes_per_chunk,
        max_edges=config.graph_max_edges_per_chunk,
    )
    stats = index_graph_chunks(
        chunks,
        user_id=args.user_id,
        extractor=extractor,
        store=store,
        concurrency=args.concurrency,
    )
    report = {
        "schema_version": 1,
        "evaluation_kind": "concept_graph_index",
        "strategy": args.strategy,
        "collection": collection,
        "user_scope": "isolated_eval_user",
        "configuration": {
            "model": "gpt-4o-mini",
            "max_nodes_per_chunk": config.graph_max_nodes_per_chunk,
            "max_edges_per_chunk": config.graph_max_edges_per_chunk,
            "concurrency": args.concurrency,
            "retry_mode": retry_ids is not None,
            "retry_of": args.retry_report.name if args.retry_report else None,
        },
        "stats": stats,
    }
    write_report(report, args.out)
    if hasattr(store, "close"):
        store.close()
    print(args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
