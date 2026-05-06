"""Explicit maintenance command to rebuild Qdrant hybrid points."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.embeddings import Embedder
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.sparse_embeddings import SparseEmbedder


def reindex_qdrant(
    *,
    supabase: Any,
    embedder: Any,
    sparse_embedder: Any,
    store: Any,
    user_id: str | None = None,
    all_users: bool = False,
    source_type: str | None = None,
    source_id: str | None = None,
    recreate_collection: bool = False,
    embedding_model: str = "text-embedding-3-small",
    sparse_model: str = "Qdrant/bm25",
    qdrant_collection: str = "learncycle_chunks",
) -> int:
    """Read existing chunks and upsert dense+sparse Qdrant points."""
    if not user_id and not all_users:
        raise RuntimeError("Reindex requires --user-id or explicit --all.")

    query = supabase.table("rag_chunks").select("*")
    if user_id:
        query = query.eq("user_id", user_id)
    if source_type:
        query = query.eq("source_type", source_type)
    if source_id:
        query = query.eq("source_id", source_id)

    rows = query.execute().data or []
    if not rows:
        return 0

    texts = [str(row.get("content") or "") for row in rows]
    dense_vectors = embedder.embed(texts)
    sparse_vectors = sparse_embedder.embed(texts)
    if len(dense_vectors) != len(rows):
        raise RuntimeError("Dense embedding count does not match chunk count.")
    if len(sparse_vectors) != len(rows):
        raise RuntimeError("Sparse embedding count does not match chunk count.")

    dim = getattr(embedder, "dimension", None) or len(dense_vectors[0])
    if recreate_collection:
        store.recreate_collection_for_hybrid(dim)
    else:
        store.ensure_collection(dim, sparse_enabled=True)

    qdrant_chunks = [
        _qdrant_chunk_from_row(row, dense_vectors[index], sparse_vectors[index])
        for index, row in enumerate(rows)
    ]
    store.upsert_chunks(qdrant_chunks, sparse_enabled=True)
    _mark_rows_completed(
        supabase=supabase,
        rows=rows,
        embedding_model=embedding_model,
        sparse_model=sparse_model,
        qdrant_collection=qdrant_collection,
    )
    return len(rows)


def _qdrant_chunk_from_row(
    row: dict[str, Any],
    dense_vector: list[float],
    sparse_vector: Any,
) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    heading_path = row.get("heading_path") or []
    heading = " > ".join(heading_path) if heading_path else None
    return {
        "id": str(row["id"]),
        "embedding": dense_vector,
        "sparse_embedding": sparse_vector,
        "chunk_id": str(row["id"]),
        "user_id": str(row["user_id"]),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "pdf_id": row.get("pdf_id"),
        "note_id": row.get("note_id"),
        "annotation_id": row.get("annotation_id"),
        "document_id": row.get("rag_document_id"),
        "page_index": row.get("page_index"),
        "heading": heading,
        "text": row.get("content"),
        "metadata": metadata,
        "content_hash": row.get("content_hash"),
        "chunk_index": metadata.get("chunk_index"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _mark_rows_completed(
    *,
    supabase: Any,
    rows: list[dict[str, Any]],
    embedding_model: str,
    sparse_model: str,
    qdrant_collection: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        chunk_id = str(row["id"])
        supabase.table("rag_chunks").update(
            {
                "embedding_status": "completed",
                "embedding_model": embedding_model,
                "embedded_at": now,
                "qdrant_collection": qdrant_collection,
                "qdrant_point_id": chunk_id,
                "embedding_error": None,
                "sparse_embedding_status": "completed",
                "sparse_embedding_model": sparse_model,
                "sparse_embedded_at": now,
                "sparse_embedding_error": None,
                "updated_at": now,
            }
        ).eq("id", chunk_id).execute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id")
    parser.add_argument("--all", action="store_true", dest="all_users")
    parser.add_argument("--source-type")
    parser.add_argument("--source-id")
    parser.add_argument("--recreate-collection", action="store_true")
    args = parser.parse_args()

    config = WorkerConfig.from_env()
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("supabase is required for reindexing.") from exc

    supabase = create_client(
        config.supabase_url,
        config.supabase_service_role_key,
    )
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
    store = QdrantStore(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection,
    )
    count = reindex_qdrant(
        supabase=supabase,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        store=store,
        user_id=args.user_id,
        all_users=args.all_users,
        source_type=args.source_type,
        source_id=args.source_id,
        recreate_collection=args.recreate_collection,
        embedding_model=config.embedding_model,
        sparse_model=config.sparse_model,
        qdrant_collection=config.qdrant_collection,
    )
    print(f"reindexed_chunks={count}")


if __name__ == "__main__":
    main()
