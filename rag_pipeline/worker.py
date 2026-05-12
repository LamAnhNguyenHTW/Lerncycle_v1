"""Supabase-backed RAG indexing worker."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import tempfile
from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.docling_ingestion import process_pdf
from rag_pipeline.embeddings import Embedder
from rag_pipeline.models import RagChunk, SourceRef
from rag_pipeline.qdrant_store import QdrantStore
from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.sparse_embeddings import SparseEmbedder
from rag_pipeline.sparse_embeddings import SparseVectorData
from rag_pipeline.source_ingestion import chunks_from_annotation
from rag_pipeline.source_ingestion import chunk_from_chat_memory
from rag_pipeline.source_ingestion import chunks_from_note


LOGGER = logging.getLogger(__name__)
PDF_BUCKET = "pdfs"


class RagWorker:
    """Worker that claims RAG jobs and writes final chunks to Supabase."""

    def __init__(
        self,
        config: WorkerConfig,
        embedder: Embedder | None = None,
        qdrant_store: QdrantStore | None = None,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        """Create a worker.

        Args:
            config: Environment-derived worker config.
        """
        try:
            from supabase import create_client
        except ImportError as exc:
            raise RuntimeError(
                "supabase is required in the Python worker environment."
            ) from exc

        self._config = config
        self._supabase = create_client(
            config.supabase_url,
            config.supabase_service_role_key,
        )
        self._refiner = SemanticRefiner(
            openai_api_key=config.openai_api_key,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            gemini_api_key=config.gemini_api_key,
            gemini_output_dimensionality=config.gemini_output_dimensionality,
        )
        self._embedder = embedder or Embedder(
            provider=config.embedding_provider,
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,
            gemini_api_key=config.gemini_api_key,
            batch_size=config.embedding_batch_size,
        )
        self._qdrant_store = qdrant_store or QdrantStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
        )
        self._sparse_embedder = sparse_embedder or SparseEmbedder(
            provider=config.sparse_provider,
            model=config.sparse_model,
        )

    def run_once(self) -> bool:
        """Claim and process at most one job.

        Returns:
            True when a job was processed, otherwise False.
        """
        job = self._claim_job()
        if not job:
            return False

        try:
            self._process_job(job)
        except Exception as exc:
            LOGGER.exception("RAG job failed: %s", job.get("id"))
            self._mark_job_failed(str(job["id"]), str(exc))
        return True

    def _claim_job(self) -> dict[str, Any] | None:
        response = self._supabase.rpc("claim_rag_index_job").execute()
        jobs = response.data or []
        return jobs[0] if jobs else None

    def _process_job(self, job: dict[str, Any]) -> None:
        source_type = str(job.get("source_type") or "pdf")
        if source_type == "pdf":
            self._process_pdf_job(job)
            return
        if source_type == "note":
            self._process_note_job(job)
            return
        if source_type == "annotation_comment":
            self._process_annotation_job(job)
            return
        if source_type == "chat_memory":
            self._process_chat_memory_job(job)
            return
        raise RuntimeError(f"Unsupported RAG job source_type: {source_type}")

    def _process_pdf_job(self, job: dict[str, Any]) -> None:
        pdf_id = str(job.get("pdf_id") or job["source_id"])
        pdf = self._fetch_pdf_row(pdf_id)
        source = SourceRef(
            user_id=str(job["user_id"]),
            source_type="pdf",
            source_id=pdf_id,
            pdf_id=pdf_id,
        )

        document_id = self._upsert_document(source, status="processing")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            pdf_path = Path(tmp_dir) / pdf["name"]
            pdf_path.write_bytes(
                self._supabase.storage.from_(PDF_BUCKET).download(
                    pdf["storage_path"]
                )
            )
            chunks, docling_version, coverage_metadata = process_pdf(
                pdf_path=pdf_path,
                source=source,
                refiner=self._refiner,
                chunking_strategy=self._config.chunking_strategy,
                chunking_version=self._config.chunking_version,
            )

        self._replace_source_chunks(document_id, source, chunks)
        self._upsert_document(
            source,
            status="completed",
            docling_version=docling_version,
            metadata={
                "coverage": coverage_metadata,
                "chunk_count": len(chunks),
            },
        )
        self._mark_job_completed(str(job["id"]))

    def _process_note_job(self, job: dict[str, Any]) -> None:
        note_id = str(job.get("note_id") or job["source_id"])
        try:
            note = self._fetch_note_row(note_id, str(job["user_id"]))
        except RuntimeError:
            source = SourceRef(
                user_id=str(job["user_id"]),
                source_type="note",
                source_id=note_id,
                pdf_id=str(job["pdf_id"]) if job.get("pdf_id") else None,
                note_id=note_id,
            )
            self._delete_source_chunks(source)
            self._upsert_document(
                source,
                status="completed",
                metadata={"skipped": "source_deleted"},
            )
            self._mark_job_completed(str(job["id"]))
            return

        source = SourceRef(
            user_id=str(note["user_id"]),
            source_type="note",
            source_id=str(note["id"]),
            pdf_id=str(note["pdf_id"]) if note.get("pdf_id") else None,
            note_id=str(note["id"]),
        )

        document_id = self._upsert_document(source, status="processing")
        chunks = chunks_from_note(
            user_id=source.user_id,
            note_id=source.source_id,
            pdf_id=source.pdf_id,
            content=note.get("content"),
            refiner=self._refiner,
            chunking_strategy=self._config.chunking_strategy,
            chunking_version=self._config.chunking_version,
        )
        self._replace_source_chunks(document_id, source, chunks)
        self._upsert_document(
            source,
            status="completed",
            metadata={"chunk_count": len(chunks)},
        )
        self._mark_job_completed(str(job["id"]))

    def _process_annotation_job(self, job: dict[str, Any]) -> None:
        annotation_id = str(job.get("annotation_id") or job["source_id"])
        try:
            annotation = self._fetch_annotation_row(
                annotation_id,
                str(job["user_id"]),
            )
        except RuntimeError:
            source = SourceRef(
                user_id=str(job["user_id"]),
                source_type="annotation_comment",
                source_id=annotation_id,
                pdf_id=str(job["pdf_id"]) if job.get("pdf_id") else None,
                annotation_id=annotation_id,
            )
            self._delete_source_chunks(source)
            self._upsert_document(
                source,
                status="completed",
                metadata={"skipped": "source_deleted"},
            )
            self._mark_job_completed(str(job["id"]))
            return

        source = SourceRef(
            user_id=str(annotation["user_id"]),
            source_type="annotation_comment",
            source_id=str(annotation["id"]),
            pdf_id=str(annotation["pdf_id"]),
            annotation_id=str(annotation["id"]),
        )

        document_id = self._upsert_document(source, status="processing")
        chunks = chunks_from_annotation(
            user_id=source.user_id,
            annotation_id=source.source_id,
            pdf_id=source.pdf_id or "",
            quote=annotation.get("quote"),
            comment=annotation.get("comment"),
            metadata={
                "page_index": annotation.get("page_index"),
                "highlight_areas": annotation.get("highlight_areas") or [],
                "color": annotation.get("color"),
            },
            refiner=self._refiner,
            chunking_strategy=self._config.chunking_strategy,
            chunking_version=self._config.chunking_version,
        )
        self._replace_source_chunks(document_id, source, chunks)
        self._upsert_document(
            source,
            status="completed",
            metadata={"chunk_count": len(chunks)},
        )
        self._mark_job_completed(str(job["id"]))

    def _process_chat_memory_job(self, job: dict[str, Any]) -> None:
        user_id = str(job["user_id"])
        session_id = str(job["source_id"])
        source = SourceRef(
            user_id=user_id,
            source_type="chat_memory",
            source_id=session_id,
        )
        summary = self._load_chat_memory_summary(user_id, session_id)
        if not summary or not str(summary.get("summary") or "").strip():
            self._delete_source_chunks(source)
            self._update_chat_memory_summary(
                user_id,
                session_id,
                {
                    "rag_chunk_id": None,
                    "qdrant_point_id": None,
                    "embedding_status": "skipped",
                    "embedded_at": None,
                    "indexing_error": None,
                    "updated_at": _utc_now(),
                },
            )
            self._upsert_document(
                source,
                status="completed",
                metadata={"skipped": "summary_missing"},
            )
            self._mark_job_completed(str(job["id"]))
            return

        document_id = self._upsert_document(source, status="processing")
        chunk = chunk_from_chat_memory(
            user_id=user_id,
            session_id=session_id,
            summary_id=str(summary["id"]),
            summary=str(summary["summary"]),
            represented_message_count=int(summary.get("represented_message_count") or 0),
            updated_at=summary.get("updated_at"),
            chunking_strategy=self._config.chunking_strategy,
            chunking_version=self._config.chunking_version,
        )
        try:
            self._replace_source_chunks(document_id, source, [chunk])
            row = self._fetch_latest_chunk_row(user_id, "chat_memory", session_id)
            self._update_chat_memory_summary(
                user_id,
                session_id,
                {
                    "rag_job_id": str(job["id"]),
                    "rag_chunk_id": row.get("id"),
                    "qdrant_point_id": row.get("qdrant_point_id") or row.get("id"),
                    "embedding_status": "completed",
                    "embedded_at": _utc_now(),
                    "indexing_error": None,
                    "updated_at": _utc_now(),
                },
            )
            self._upsert_document(
                source,
                status="completed",
                metadata={"chunk_count": 1},
            )
            self._mark_job_completed(str(job["id"]))
        except Exception as exc:
            self._update_chat_memory_summary(
                user_id,
                session_id,
                {
                    "embedding_status": "failed",
                    "indexing_error": str(exc)[:4000],
                    "updated_at": _utc_now(),
                },
            )
            raise

    def _fetch_pdf_row(self, pdf_id: str) -> dict[str, Any]:
        response = (
            self._supabase.table("pdfs")
            .select("id, user_id, name, storage_path")
            .eq("id", pdf_id)
            .single()
            .execute()
        )
        if not response.data:
            raise RuntimeError(f"PDF not found: {pdf_id}")
        return response.data

    def _fetch_note_row(self, note_id: str, user_id: str) -> dict[str, Any]:
        response = (
            self._supabase.table("notes")
            .select("id, user_id, pdf_id, content")
            .eq("id", note_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not response.data:
            raise RuntimeError(f"Note not found: {note_id}")
        return response.data

    def _fetch_annotation_row(
        self,
        annotation_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        response = (
            self._supabase.table("pdf_annotations")
            .select(
                "id, user_id, pdf_id, page_index, highlight_areas, "
                "quote, comment, color"
            )
            .eq("id", annotation_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not response.data:
            raise RuntimeError(f"Annotation not found: {annotation_id}")
        return response.data

    def _load_chat_memory_summary(
        self,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        response = (
            self._supabase.table("chat_memory_summaries")
            .select(
                "id, user_id, session_id, summary, represented_message_count, updated_at"
            )
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return response.data

    def _fetch_latest_chunk_row(
        self,
        user_id: str,
        source_type: str,
        source_id: str,
    ) -> dict[str, Any]:
        response = (
            self._supabase.table("rag_chunks")
            .select("id, qdrant_point_id")
            .eq("user_id", user_id)
            .eq("source_type", source_type)
            .eq("source_id", source_id)
            .order("updated_at", desc=True)
            .limit(1)
            .single()
            .execute()
        )
        if not response.data:
            raise RuntimeError("Chat memory chunk row not found after indexing.")
        return response.data

    def _update_chat_memory_summary(
        self,
        user_id: str,
        session_id: str,
        values: dict[str, Any],
    ) -> None:
        self._supabase.table("chat_memory_summaries").update(values).eq(
            "user_id",
            user_id,
        ).eq("session_id", session_id).execute()

    def _upsert_document(
        self,
        source: SourceRef,
        status: str,
        docling_version: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = _utc_now()
        row = {
            "user_id": source.user_id,
            "source_type": source.source_type,
            "source_id": source.source_id,
            "pdf_id": source.pdf_id,
            "note_id": source.note_id,
            "annotation_id": source.annotation_id,
            "status": status,
            "error_message": error_message,
            "docling_version": docling_version,
            "chunking_strategy": self._config.chunking_strategy,
            "chunking_version": self._config.chunking_version,
            "metadata": metadata or {},
            "updated_at": now,
        }
        if status == "processing":
            row["started_at"] = now
        if status in {"completed", "failed"}:
            row["completed_at"] = now

        response = (
            self._supabase.table("rag_documents")
            .upsert(
                row,
                on_conflict=(
                    "user_id,source_type,source_id,"
                    "chunking_strategy,chunking_version"
                ),
            )
            .execute()
        )
        data = response.data or []
        if not data:
            lookup = (
                self._supabase.table("rag_documents")
                .select("id")
                .eq("user_id", source.user_id)
                .eq("source_type", source.source_type)
                .eq("source_id", source.source_id)
                .eq("chunking_strategy", self._config.chunking_strategy)
                .eq("chunking_version", self._config.chunking_version)
                .single()
                .execute()
            )
            return str(lookup.data["id"])
        return str(data[0]["id"])

    def _upsert_chunks(
        self,
        rag_document_id: str,
        chunks: list[RagChunk],
    ) -> None:
        if not chunks:
            return

        rows = []
        now = _utc_now()
        for index, chunk in enumerate(chunks):
            source = chunk.source
            client_chunk_key = _client_chunk_key(chunk, index)
            metadata = dict(chunk.metadata)
            metadata["client_chunk_key"] = client_chunk_key
            metadata["chunk_index"] = index
            rows.append(
                {
                    "user_id": source.user_id,
                    "rag_document_id": rag_document_id,
                    "source_type": source.source_type,
                    "source_id": source.source_id,
                    "pdf_id": source.pdf_id,
                    "note_id": source.note_id,
                    "annotation_id": source.annotation_id,
                    "page_index": chunk.page_index,
                    "heading_path": chunk.heading_path,
                    "chunk_kind": chunk.chunk_kind,
                    "content": chunk.content,
                    "metadata": metadata,
                    "content_hash": chunk.content_hash,
                    "chunking_strategy": self._config.chunking_strategy,
                    "chunking_version": self._config.chunking_version,
                    "embedding_status": "pending",
                    "embedding_model": None,
                    "embedded_at": None,
                    "qdrant_collection": self._config.qdrant_collection,
                    "qdrant_point_id": None,
                    "sparse_embedding_status": "pending",
                    "sparse_embedding_model": None,
                    "sparse_embedded_at": None,
                    "sparse_embedding_error": None,
                    "updated_at": now,
                }
            )

        response = self._supabase.table("rag_chunks").upsert(
            rows,
            on_conflict="user_id,source_type,source_id,content_hash",
            returning="representation",
        ).execute()
        returned_rows = response.data or []
        indexed_rows = _match_chunk_rows(chunks, returned_rows)

        try:
            embeddings = self._embedder.embed([chunk.content for chunk in chunks])
            dim = self._embedder.dimension
            if dim is None and embeddings:
                dim = len(embeddings[0])
            if dim is None:
                return
            sparse_enabled = bool(
                getattr(self._config, "sparse_enabled", False)
            )
            sparse_embeddings: list[SparseVectorData | None] = [None] * len(chunks)
            if sparse_enabled:
                sparse_embeddings = self._sparse_embedder.embed(
                    [chunk.content for chunk in chunks]
                )
                if len(sparse_embeddings) != len(chunks):
                    raise RuntimeError(
                        "Sparse embedding provider returned "
                        f"{len(sparse_embeddings)} vectors for {len(chunks)} inputs."
                    )
            self._qdrant_store.ensure_collection(dim, sparse_enabled=sparse_enabled)
            qdrant_chunks = [
                _qdrant_chunk_payload(
                    row,
                    chunk,
                    index,
                    embeddings[index],
                    sparse_embeddings[index],
                )
                for index, (chunk, row) in enumerate(zip(chunks, indexed_rows))
            ]
            self._qdrant_store.upsert_chunks(
                qdrant_chunks,
                sparse_enabled=sparse_enabled,
            )
            self._mark_chunks_embedding_completed(
                [row["id"] for row in indexed_rows],
                sparse_completed=sparse_enabled,
            )
        except Exception as exc:
            self._mark_chunks_embedding_failed(
                [row["id"] for row in indexed_rows],
                str(exc),
                sparse_failed=bool(getattr(self._config, "sparse_enabled", False)),
            )
            raise

    def _replace_source_chunks(
        self,
        rag_document_id: str,
        source: SourceRef,
        chunks: list[RagChunk],
    ) -> None:
        self._qdrant_store.delete_points_by_source(
            source.user_id,
            source.source_type,
            source.source_id,
        )
        self._supabase.table("rag_chunks").delete().eq(
            "user_id",
            source.user_id,
        ).eq("source_type", source.source_type).eq(
            "source_id",
            source.source_id,
        ).eq(
            "chunking_version",
            self._config.chunking_version,
        ).execute()

        self._upsert_chunks(rag_document_id, chunks)

    def _delete_source_chunks(self, source: SourceRef) -> None:
        self._qdrant_store.delete_points_by_source(
            source.user_id,
            source.source_type,
            source.source_id,
        )
        self._supabase.table("rag_chunks").delete().eq(
            "user_id",
            source.user_id,
        ).eq("source_type", source.source_type).eq(
            "source_id",
            source.source_id,
        ).eq(
            "chunking_version",
            self._config.chunking_version,
        ).execute()

    def _mark_chunks_embedding_completed(
        self,
        chunk_ids: list[str],
        sparse_completed: bool = False,
    ) -> None:
        now = _utc_now()
        for chunk_id in chunk_ids:
            update = {
                "embedding_status": "completed",
                "embedding_model": self._config.embedding_model,
                "embedded_at": now,
                "qdrant_collection": self._config.qdrant_collection,
                "qdrant_point_id": chunk_id,
                "embedding_error": None,
                "updated_at": now,
            }
            if sparse_completed:
                update.update(
                    {
                        "sparse_embedding_status": "completed",
                        "sparse_embedding_model": self._config.sparse_model,
                        "sparse_embedded_at": now,
                        "sparse_embedding_error": None,
                    }
                )
            self._supabase.table("rag_chunks").update(update).eq(
                "id",
                chunk_id,
            ).execute()

    def _mark_chunks_embedding_failed(
        self,
        chunk_ids: list[str],
        error_message: str,
        sparse_failed: bool = False,
    ) -> None:
        now = _utc_now()
        for chunk_id in chunk_ids:
            update = {
                "embedding_status": "failed",
                "embedding_error": error_message[:4000],
                "updated_at": now,
            }
            if sparse_failed:
                update.update(
                    {
                        "sparse_embedding_status": "failed",
                        "sparse_embedding_error": error_message[:4000],
                    }
                )
            self._supabase.table("rag_chunks").update(update).eq(
                "id",
                chunk_id,
            ).execute()


    def _mark_job_completed(self, job_id: str) -> None:
        self._supabase.table("rag_index_jobs").update(
            {
                "status": "completed",
                "locked_at": None,
                "completed_at": _utc_now(),
                "updated_at": _utc_now(),
                "error_message": None,
            }
        ).eq("id", job_id).execute()

    def _mark_job_failed(self, job_id: str, error_message: str) -> None:
        self._supabase.table("rag_index_jobs").update(
            {
                "status": "failed",
                "locked_at": None,
                "completed_at": _utc_now(),
                "updated_at": _utc_now(),
                "error_message": error_message[:4000],
            }
        ).eq("id", job_id).execute()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _match_chunk_rows(
    chunks: list[RagChunk],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        _row_client_chunk_key(row): row
        for row in rows
    }
    matched = []
    for index, chunk in enumerate(chunks):
        key = _client_chunk_key(chunk, index)
        row = by_key.get(key)
        if row is None:
            raise RuntimeError("Could not map Supabase chunk row after upsert.")
        matched.append(row)
    return matched


def _client_chunk_key(chunk: RagChunk, chunk_index: int) -> str:
    source = chunk.source
    return "|".join(
        [
            source.source_type,
            source.source_id,
            str(chunk_index),
            chunk.content_hash,
        ]
    )


def _row_client_chunk_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("client_chunk_key"):
        return str(metadata["client_chunk_key"])
    return "|".join(
        [
            str(row.get("source_type")),
            str(row.get("source_id")),
            str(metadata.get("chunk_index", "")),
            str(row.get("content_hash")),
        ]
    )


def _qdrant_chunk_payload(
    row: dict[str, Any],
    chunk: RagChunk,
    chunk_index: int,
    embedding: list[float],
    sparse_embedding: SparseVectorData | None = None,
) -> dict[str, Any]:
    source = chunk.source
    heading = " > ".join(chunk.heading_path) if chunk.heading_path else None
    metadata = dict(row.get("metadata") or chunk.metadata)
    payload = {
        "id": str(row["id"]),
        "embedding": embedding,
        "chunk_id": str(row["id"]),
        "user_id": source.user_id,
        "source_type": source.source_type,
        "source_id": source.source_id,
        "pdf_id": source.pdf_id,
        "note_id": source.note_id,
        "annotation_id": source.annotation_id,
        "document_id": row.get("rag_document_id"),
        "page_index": chunk.page_index,
        "heading": heading,
        "text": chunk.content,
        "metadata": metadata,
        "content_hash": chunk.content_hash,
        "chunk_index": chunk_index,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    if sparse_embedding is not None:
        payload["sparse_embedding"] = sparse_embedding
    return payload


def main() -> None:
    """Run one worker iteration from CLI."""
    logging.basicConfig(level=logging.INFO)
    worker = RagWorker(WorkerConfig.from_env())
    processed = worker.run_once()
    LOGGER.info("processed_job=%s", processed)


if __name__ == "__main__":
    main()
