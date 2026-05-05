"""Tests for worker source branches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.worker import RagWorker


class RecordingWorker(RagWorker):
    """Worker test double that records writes without Supabase."""

    def __init__(self) -> None:
        self._config = SimpleNamespace(
            chunking_strategy="test_strategy",
            chunking_version="v1",
            qdrant_collection=None,
        )
        self._refiner = SemanticRefiner(openai_api_key=None)
        self.completed_jobs: list[str] = []
        self.documents: list[tuple[str, str, dict[str, Any] | None]] = []
        self.replaced_chunks: dict[tuple[str, str, str], list[Any]] = {}

    def _fetch_note_row(self, note_id: str, user_id: str) -> dict[str, Any]:
        return {
            "id": note_id,
            "user_id": user_id,
            "pdf_id": "pdf-1",
            "content": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Note text"}],
                    }
                ],
            },
        }

    def _fetch_annotation_row(
        self,
        annotation_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        return {
            "id": annotation_id,
            "user_id": user_id,
            "pdf_id": "pdf-1",
            "page_index": 2,
            "highlight_areas": [{"pageIndex": 2}],
            "quote": "Highlighted quote",
            "comment": "User comment",
            "color": "yellow",
        }

    def _upsert_document(
        self,
        source,
        status: str,
        docling_version: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.documents.append((source.source_type, status, metadata))
        return "rag-document-1"

    def _replace_source_chunks(self, rag_document_id, source, chunks) -> None:
        self.replaced_chunks[
            (source.user_id, source.source_type, source.source_id)
        ] = chunks

    def _mark_job_completed(self, job_id: str) -> None:
        self.completed_jobs.append(job_id)


def test_worker_processes_note_job() -> None:
    worker = RecordingWorker()

    worker._process_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_type": "note",
            "source_id": "note-1",
            "note_id": "note-1",
            "pdf_id": "pdf-1",
        }
    )

    chunks = worker.replaced_chunks[("user-1", "note", "note-1")]
    assert worker.completed_jobs == ["job-1"]
    assert chunks[0].source.source_type == "note"
    assert chunks[0].source.source_id == "note-1"
    assert chunks[0].content == "Note text"


def test_worker_processes_annotation_comment_job() -> None:
    worker = RecordingWorker()

    worker._process_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_type": "annotation_comment",
            "source_id": "annotation-1",
            "annotation_id": "annotation-1",
            "pdf_id": "pdf-1",
        }
    )

    chunks = worker.replaced_chunks[
        ("user-1", "annotation_comment", "annotation-1")
    ]
    assert worker.completed_jobs == ["job-1"]
    assert chunks[0].source.source_type == "annotation_comment"
    assert chunks[0].page_index == 2
    assert "Highlighted quote" in chunks[0].content
    assert "User comment" in chunks[0].content


def test_note_reindex_replaces_chunks_without_duplication() -> None:
    worker = RecordingWorker()
    job = {
        "id": "job-1",
        "user_id": "user-1",
        "source_type": "note",
        "source_id": "note-1",
        "note_id": "note-1",
        "pdf_id": "pdf-1",
    }

    worker._process_job(job)
    worker._process_job({**job, "id": "job-2"})

    chunks = worker.replaced_chunks[("user-1", "note", "note-1")]
    assert len(chunks) == 1
    assert worker.completed_jobs == ["job-1", "job-2"]
