"""Supabase-backed RAG indexing worker."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import tempfile
from typing import Any

from rag_pipeline.config import WorkerConfig
from rag_pipeline.docling_ingestion import process_pdf
from rag_pipeline.models import RagChunk, SourceRef
from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.source_ingestion import chunks_from_annotation
from rag_pipeline.source_ingestion import chunks_from_note


LOGGER = logging.getLogger(__name__)
PDF_BUCKET = "pdfs"


class RagWorker:
    """Worker that claims RAG jobs and writes final chunks to Supabase."""

    def __init__(self, config: WorkerConfig) -> None:
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
        with tempfile.TemporaryDirectory() as tmp_dir:
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

        self._upsert_chunks(document_id, chunks)
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
        for chunk in chunks:
            source = chunk.source
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
                    "metadata": chunk.metadata,
                    "content_hash": chunk.content_hash,
                    "chunking_strategy": self._config.chunking_strategy,
                    "chunking_version": self._config.chunking_version,
                    "embedding_status": "pending",
                    "embedding_model": None,
                    "embedded_at": None,
                    "qdrant_collection": self._config.qdrant_collection,
                    "qdrant_point_id": None,
                    "updated_at": now,
                }
            )

        self._supabase.table("rag_chunks").upsert(
            rows,
            on_conflict="user_id,source_type,source_id,content_hash",
        ).execute()

    def _replace_source_chunks(
        self,
        rag_document_id: str,
        source: SourceRef,
        chunks: list[RagChunk],
    ) -> None:
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


def main() -> None:
    """Run one worker iteration from CLI."""
    logging.basicConfig(level=logging.INFO)
    worker = RagWorker(WorkerConfig.from_env())
    processed = worker.run_once()
    LOGGER.info("processed_job=%s", processed)


if __name__ == "__main__":
    main()
