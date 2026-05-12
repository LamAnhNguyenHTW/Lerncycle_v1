"""Tests for worker source branches."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import Any

from rag_pipeline.models import RagChunk, SourceRef
from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.sparse_embeddings import SparseVectorData
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
        self.failed_jobs: list[tuple[str, str]] = []

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

    def _mark_job_failed(self, job_id: str, error_message: str) -> None:
        self.failed_jobs.append((job_id, error_message))


class FakeGraphExtractor:
    def __init__(self, fail_all: bool = False, fail_first: bool = False) -> None:
        self.fail_all = fail_all
        self.fail_first = fail_first
        self.calls = []

    def extract_from_chunk(self, chunk):
        self.calls.append(chunk)
        if self.fail_all or (self.fail_first and len(self.calls) == 1):
            raise RuntimeError("extract failed")
        return SimpleNamespace(nodes=[object()], edges=[object()])


class FakeGraphStore:
    def __init__(self) -> None:
        self.deleted = []
        self.upserts = []

    def delete_by_source(self, user_id, source_type, source_id):
        self.deleted.append((user_id, source_type, source_id))

    def upsert_extraction(self, user_id, chunk, extraction):
        self.upserts.append((user_id, chunk, extraction))
        return {"nodes_upserted": len(extraction.nodes), "relationships_upserted": len(extraction.edges)}


class GraphWorker(RecordingWorker):
    def __init__(self, enabled: bool = True, extractor=None, store=None) -> None:
        super().__init__()
        self._config.graph_extraction_enabled = enabled
        self._graph_extractor = extractor or FakeGraphExtractor()
        self._graph_store = store or FakeGraphStore()
        self.graph_completed: list[tuple[str, dict[str, Any] | None]] = []
        self.graph_failed: list[tuple[str, str]] = []
        self.chunks = [
            {"id": "chunk-1", "chunk_id": "chunk-1", "source_type": "pdf", "source_id": "pdf-1", "text": "A"},
            {"id": "chunk-2", "chunk_id": "chunk-2", "source_type": "pdf", "source_id": "pdf-1", "text": "B"},
        ]

    def _load_chunks_for_graph(self, user_id: str, source_type: str, source_id: str):
        self.loaded = (user_id, source_type, source_id)
        return self.chunks

    def _mark_job_completed(self, job_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.graph_completed.append((job_id, metadata))

    def _mark_job_failed(self, job_id: str, error_message: str) -> None:
        self.graph_failed.append((job_id, error_message))


class DeletedSourceWorker(RecordingWorker):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_sources: list[tuple[str, str, str]] = []

    def _fetch_note_row(self, note_id: str, user_id: str) -> dict[str, Any]:
        raise RuntimeError("missing note")

    def _fetch_annotation_row(
        self,
        annotation_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        raise RuntimeError("missing annotation")

    def _delete_source_chunks(self, source) -> None:
        self.deleted_sources.append(
            (source.user_id, source.source_type, source.source_id)
        )


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.dimension = 2
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("embedding failed")
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeSparseEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[SparseVectorData]:
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("sparse failed")
        return [
            SparseVectorData(indices=[index], values=[float(index + 1)])
            for index, _ in enumerate(texts)
        ]


class FakeQdrantStore:
    def __init__(self, fail_upsert: bool = False) -> None:
        self.fail_upsert = fail_upsert
        self.ensure_calls: list[tuple[int, bool]] = []
        self.upsert_calls: list[tuple[list[dict[str, Any]], bool]] = []
        self.delete_calls: list[tuple[str, str, str]] = []

    def ensure_collection(self, dim: int, sparse_enabled: bool = False) -> None:
        self.ensure_calls.append((dim, sparse_enabled))

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        sparse_enabled: bool = False,
    ) -> None:
        if self.fail_upsert:
            raise RuntimeError("qdrant failed")
        self.upsert_calls.append((chunks, sparse_enabled))

    def delete_points_by_source(
        self,
        user_id: str,
        source_type: str,
        source_id: str,
    ) -> None:
        self.delete_calls.append((user_id, source_type, source_id))


class FakeResponse:
    def __init__(self, data: Any = None) -> None:
        self.data = data


class FakeTable:
    def __init__(self, supabase: "FakeSupabase", name: str) -> None:
        self.supabase = supabase
        self.name = name
        self.pending_update: dict[str, Any] | None = None
        self.filters: dict[str, Any] = {}

    def upsert(self, rows, **kwargs):
        self.supabase.operations.append((self.name, "upsert", rows, kwargs))
        if self.name == "rag_chunks":
            row_list = rows if isinstance(rows, list) else [rows]
            self.supabase.chunk_rows = [
                {
                    **row,
                    "id": f"chunk-{index + 1}",
                    "created_at": "created",
                    "updated_at": row.get("updated_at", "updated"),
                }
                for index, row in enumerate(row_list)
            ]
        return self

    def select(self, *_args):
        return self

    def update(self, values):
        self.pending_update = values
        return self

    def delete(self):
        self.supabase.operations.append((self.name, "delete_start", None, {}))
        return self

    def eq(self, key, value):
        self.filters[key] = value
        if self.pending_update is not None and key == "id":
            self.supabase.chunk_updates.append((value, self.pending_update))
        return self

    def execute(self):
        self.supabase.operations.append(
            (self.name, "execute", self.pending_update, dict(self.filters))
        )
        if self.name == "rag_chunks" and self.pending_update is None:
            return FakeResponse(self.supabase.chunk_rows)
        return FakeResponse([])


class FakeSupabase:
    def __init__(self) -> None:
        self.chunk_rows: list[dict[str, Any]] = []
        self.chunk_updates: list[tuple[str, dict[str, Any]]] = []
        self.operations: list[tuple[str, str, Any, dict[str, Any]]] = []

    def table(self, name: str) -> FakeTable:
        return FakeTable(self, name)


class ChunkWorker(RagWorker):
    def __init__(
        self,
        embedder: FakeEmbedder | None = None,
        qdrant_store: FakeQdrantStore | None = None,
        sparse_embedder: FakeSparseEmbedder | None = None,
        sparse_enabled: bool = False,
    ) -> None:
        self._config = SimpleNamespace(
            chunking_strategy="test_strategy",
            chunking_version="v1",
            qdrant_collection="learncycle_chunks",
            embedding_model="text-embedding-3-small",
            sparse_enabled=sparse_enabled,
            sparse_model="Qdrant/bm25",
        )
        self._supabase = FakeSupabase()
        self._embedder = embedder or FakeEmbedder()
        self._sparse_embedder = sparse_embedder or FakeSparseEmbedder()
        self._qdrant_store = qdrant_store or FakeQdrantStore()


def _chunk(content: str = "Note text") -> RagChunk:
    return RagChunk(
        source=SourceRef(
            user_id="user-1",
            source_type="note",
            source_id="note-1",
            note_id="note-1",
        ),
        content=content,
        content_hash=f"hash-{content}",
        metadata={"kind": "test"},
    )


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


def test_worker_embeds_and_upserts_chunks_to_qdrant() -> None:
    worker = ChunkWorker()

    worker._upsert_chunks("document-1", [_chunk()])

    assert worker._embedder.calls == [["Note text"]]
    assert worker._qdrant_store.ensure_calls == [(2, False)]
    upserted = worker._qdrant_store.upsert_calls[0][0][0]
    assert upserted["id"] == "chunk-1"
    assert upserted["chunk_id"] == "chunk-1"
    assert upserted["text"] == "Note text"
    assert upserted["source_type"] == "note"


def test_note_reindex_deletes_old_qdrant_points_before_upsert() -> None:
    worker = ChunkWorker()
    source = _chunk().source

    worker._replace_source_chunks("document-1", source, [_chunk()])

    assert worker._qdrant_store.delete_calls == [("user-1", "note", "note-1")]
    assert worker._qdrant_store.upsert_calls


def test_supabase_chunk_rows_marked_completed_after_qdrant_upsert() -> None:
    worker = ChunkWorker()

    worker._upsert_chunks("document-1", [_chunk()])

    update = worker._supabase.chunk_updates[-1][1]
    assert update["embedding_status"] == "completed"
    assert update["embedding_model"] == "text-embedding-3-small"
    assert update["qdrant_collection"] == "learncycle_chunks"
    assert update["qdrant_point_id"] == "chunk-1"
    assert update["embedding_error"] is None


def test_worker_marks_chunks_failed_when_embedding_fails() -> None:
    worker = ChunkWorker(embedder=FakeEmbedder(fail=True))

    with pytest.raises(RuntimeError, match="embedding failed"):
        worker._upsert_chunks("document-1", [_chunk()])

    update = worker._supabase.chunk_updates[-1][1]
    assert update["embedding_status"] == "failed"
    assert update["embedding_error"] == "embedding failed"


def test_worker_marks_chunks_failed_when_qdrant_upsert_fails() -> None:
    worker = ChunkWorker(qdrant_store=FakeQdrantStore(fail_upsert=True))

    with pytest.raises(RuntimeError, match="qdrant failed"):
        worker._upsert_chunks("document-1", [_chunk()])

    update = worker._supabase.chunk_updates[-1][1]
    assert update["embedding_status"] == "failed"
    assert update["embedding_error"] == "qdrant failed"


def test_deleted_note_triggers_qdrant_cleanup() -> None:
    worker = DeletedSourceWorker()

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

    assert worker.deleted_sources == [("user-1", "note", "note-1")]
    assert worker.documents[-1] == (
        "note",
        "completed",
        {"skipped": "source_deleted"},
    )
    assert worker.completed_jobs == ["job-1"]


def test_deleted_annotation_triggers_qdrant_cleanup() -> None:
    worker = DeletedSourceWorker()

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

    assert worker.deleted_sources == [
        ("user-1", "annotation_comment", "annotation-1")
    ]
    assert worker.documents[-1] == (
        "annotation_comment",
        "completed",
        {"skipped": "source_deleted"},
    )
    assert worker.completed_jobs == ["job-1"]


def test_worker_does_not_mark_job_completed_when_embedding_fails() -> None:
    worker = RecordingWorker()

    def fail_replace(_document_id, _source, _chunks) -> None:
        raise RuntimeError("embedding failed")

    worker._replace_source_chunks = fail_replace

    with pytest.raises(RuntimeError, match="embedding failed"):
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

    assert worker.completed_jobs == []


def test_worker_generates_sparse_embeddings_for_chunks() -> None:
    worker = ChunkWorker(sparse_enabled=True)

    worker._upsert_chunks("document-1", [_chunk()])

    assert worker._sparse_embedder.calls == [["Note text"]]


def test_worker_upserts_dense_and_sparse_to_qdrant() -> None:
    worker = ChunkWorker(sparse_enabled=True)

    worker._upsert_chunks("document-1", [_chunk()])

    assert worker._qdrant_store.ensure_calls == [(2, True)]
    chunks, sparse_enabled = worker._qdrant_store.upsert_calls[0]
    assert sparse_enabled is True
    assert chunks[0]["embedding"] == [0.0, 1.0]
    assert chunks[0]["sparse_embedding"] == SparseVectorData(
        indices=[0],
        values=[1.0],
    )


def test_worker_marks_sparse_embedding_completed() -> None:
    worker = ChunkWorker(sparse_enabled=True)

    worker._upsert_chunks("document-1", [_chunk()])

    update = worker._supabase.chunk_updates[-1][1]
    assert update["sparse_embedding_status"] == "completed"
    assert update["sparse_embedding_model"] == "Qdrant/bm25"
    assert update["sparse_embedded_at"] is not None
    assert update["sparse_embedding_error"] is None


def test_worker_marks_sparse_embedding_failed_when_sparse_embedder_fails() -> None:
    worker = ChunkWorker(
        sparse_enabled=True,
        sparse_embedder=FakeSparseEmbedder(fail=True),
    )

    with pytest.raises(RuntimeError, match="sparse failed"):
        worker._upsert_chunks("document-1", [_chunk()])

    update = worker._supabase.chunk_updates[-1][1]
    assert update["sparse_embedding_status"] == "failed"
    assert update["sparse_embedding_error"] == "sparse failed"


def test_worker_does_not_mark_job_completed_when_sparse_fails() -> None:
    worker = RecordingWorker()

    def fail_replace(_document_id, _source, _chunks) -> None:
        raise RuntimeError("sparse failed")

    worker._replace_source_chunks = fail_replace

    with pytest.raises(RuntimeError, match="sparse failed"):
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

    assert worker.completed_jobs == []


def test_worker_dispatches_knowledge_graph_job() -> None:
    worker = GraphWorker()

    worker._process_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_type": "knowledge_graph",
            "source_id": "pdf-1",
            "metadata": {"original_source_type": "pdf", "original_source_id": "pdf-1"},
        }
    )

    assert worker.loaded == ("user-1", "pdf", "pdf-1")
    assert worker._graph_store.upserts
    assert worker.graph_completed[0][0] == "job-1"


def test_worker_deletes_old_graph_by_source_before_reextract() -> None:
    store = FakeGraphStore()
    worker = GraphWorker(store=store)

    worker._process_knowledge_graph_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_id": "pdf-1",
            "metadata": {"original_source_type": "pdf", "original_source_id": "pdf-1"},
        }
    )

    assert store.deleted[0] == ("user-1", "pdf", "pdf-1")


def test_worker_continues_on_single_chunk_extraction_failure() -> None:
    extractor = FakeGraphExtractor(fail_first=True)
    store = FakeGraphStore()
    worker = GraphWorker(extractor=extractor, store=store)

    worker._process_knowledge_graph_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_id": "pdf-1",
            "metadata": {"original_source_type": "pdf", "original_source_id": "pdf-1"},
        }
    )

    assert len(store.upserts) == 1
    assert worker.graph_completed[0][1]["chunks_failed"] == 1


def test_worker_marks_graph_job_failed_when_all_chunks_fail() -> None:
    worker = GraphWorker(extractor=FakeGraphExtractor(fail_all=True))

    worker._process_knowledge_graph_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_id": "pdf-1",
            "metadata": {"original_source_type": "pdf", "original_source_id": "pdf-1"},
        }
    )

    assert worker.graph_failed
    assert worker.graph_completed == []


def test_worker_skips_graph_job_when_extraction_disabled() -> None:
    worker = GraphWorker(enabled=False)

    worker._process_knowledge_graph_job({"id": "job-1", "user_id": "user-1", "source_id": "pdf-1"})

    assert worker.graph_completed[0][1] == {"skipped": "graph_extraction_disabled"}


def test_worker_marks_graph_job_completed_when_no_chunks() -> None:
    store = FakeGraphStore()
    worker = GraphWorker(store=store)
    worker.chunks = []

    worker._process_knowledge_graph_job(
        {
            "id": "job-1",
            "user_id": "user-1",
            "source_id": "pdf-1",
            "metadata": {"original_source_type": "pdf", "original_source_id": "pdf-1"},
        }
    )

    assert store.deleted == [("user-1", "pdf", "pdf-1")]
    assert worker.graph_completed[0][1]["skipped"] == "no_chunks"
