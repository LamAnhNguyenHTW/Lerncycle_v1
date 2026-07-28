"""Tests for the delete_source cleanup job (track rag_data_lifecycle_20260613)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from rag_pipeline.worker import RagWorker


class FakeDeleteQuery:
    """Records chained .eq filters for a supabase table delete."""

    def __init__(self, log: list[dict[str, Any]], table: str) -> None:
        self._log = log
        self._entry: dict[str, Any] = {"table": table, "op": "delete", "filters": {}}

    def eq(self, column: str, value: Any) -> "FakeDeleteQuery":
        self._entry["filters"][column] = value
        return self

    def execute(self) -> SimpleNamespace:
        self._log.append(self._entry)
        return SimpleNamespace(data=[])


class FakeUpdateQuery:
    def __init__(self, log: list[dict[str, Any]], table: str, payload: dict[str, Any]) -> None:
        self._log = log
        self._entry: dict[str, Any] = {
            "table": table,
            "op": "update",
            "payload": payload,
            "filters": {},
        }

    def eq(self, column: str, value: Any) -> "FakeUpdateQuery":
        self._entry["filters"][column] = value
        return self

    def execute(self) -> SimpleNamespace:
        self._log.append(self._entry)
        return SimpleNamespace(data=[])


class FakeTable:
    def __init__(self, log: list[dict[str, Any]], table: str) -> None:
        self._log = log
        self._table = table

    def delete(self) -> FakeDeleteQuery:
        return FakeDeleteQuery(self._log, self._table)

    def update(self, payload: dict[str, Any]) -> FakeUpdateQuery:
        return FakeUpdateQuery(self._log, self._table, payload)


class FakeSupabase:
    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []

    def table(self, name: str) -> FakeTable:
        return FakeTable(self.log, name)


class FakeQdrantStore:
    def __init__(self) -> None:
        self.deleted_by_pdf: list[tuple[str, str]] = []
        self.deleted_by_source: list[tuple[str, str, str]] = []

    def delete_points_by_pdf_id(self, user_id: str, pdf_id: str) -> None:
        self.deleted_by_pdf.append((user_id, pdf_id))

    def delete_points_by_source(self, user_id: str, source_type: str, source_id: str) -> None:
        self.deleted_by_source.append((user_id, source_type, source_id))


class FakeGraphStore:
    def __init__(self) -> None:
        self.deleted_by_pdf: list[tuple[str, str]] = []
        self.deleted_by_source: list[tuple[str, str, str]] = []

    def delete_by_pdf_id(self, user_id: str, pdf_id: str) -> None:
        self.deleted_by_pdf.append((user_id, pdf_id))

    def delete_by_source(self, user_id: str, source_type: str, source_id: str) -> None:
        self.deleted_by_source.append((user_id, source_type, source_id))


class DeleteWorker(RagWorker):
    """Worker double wired with fakes; no real Supabase/Qdrant/Neo4j."""

    def __init__(self, graph_store: FakeGraphStore | None = None) -> None:
        self._config = SimpleNamespace(
            chunking_strategy="test_strategy",
            chunking_version="v1",
            qdrant_collection="test",
            worker_max_attempts=3,
            neo4j_uri=None,
            neo4j_user=None,
            neo4j_password=None,
        )
        self._supabase = FakeSupabase()
        self._qdrant_store = FakeQdrantStore()
        self._graph_store = graph_store
        self.completed: list[tuple[str, dict[str, Any] | None]] = []
        self.learning_deletes: list[tuple[str, str, str]] = []

    def _mark_job_completed(self, job_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.completed.append((job_id, metadata))

    def _delete_learning_graph_for_source(
        self,
        user_id: str,
        source_type: str,
        source_id: str,
    ) -> None:
        self.learning_deletes.append((user_id, source_type, source_id))


def _pdf_job(**overrides: Any) -> dict[str, Any]:
    job = {
        "id": "job-1",
        "user_id": "user-1",
        "job_kind": "delete_source",
        "source_type": "pdf",
        "source_id": "pdf-1",
        "pdf_id": None,
        "metadata": {"deleted_pdf_id": "pdf-1"},
    }
    job.update(overrides)
    return job


def test_delete_source_pdf_cleans_qdrant_chunks_documents_and_primers() -> None:
    worker = DeleteWorker(graph_store=FakeGraphStore())

    worker._process_job(_pdf_job())

    assert worker._qdrant_store.deleted_by_pdf == [("user-1", "pdf-1")]
    tables = [(entry["table"], entry["filters"]) for entry in worker._supabase.log]
    assert ("rag_chunks", {"user_id": "user-1", "pdf_id": "pdf-1"}) in tables
    assert ("rag_documents", {"user_id": "user-1", "pdf_id": "pdf-1"}) in tables
    assert (
        "rag_document_primers",
        {"user_id": "user-1", "source_type": "pdf", "source_id": "pdf-1"},
    ) in tables
    assert worker.completed == [
        ("job-1", {"deleted_pdf_id": "pdf-1", "cleanup": "completed"})
    ]


def test_delete_source_pdf_cleans_both_graph_layers() -> None:
    graph_store = FakeGraphStore()
    worker = DeleteWorker(graph_store=graph_store)

    worker._process_job(_pdf_job())

    assert graph_store.deleted_by_pdf == [("user-1", "pdf-1")]
    assert graph_store.deleted_by_source == [("user-1", "pdf", "pdf-1")]
    assert worker.learning_deletes == [("user-1", "pdf", "pdf-1")]


def test_delete_source_pdf_without_metadata_falls_back_to_source_id() -> None:
    worker = DeleteWorker()

    worker._process_job(_pdf_job(metadata={}))

    assert worker._qdrant_store.deleted_by_pdf == [("user-1", "pdf-1")]
    assert worker.completed[0][1] == {"cleanup": "completed"}


def test_delete_source_annotation_uses_source_scoped_deletes() -> None:
    graph_store = FakeGraphStore()
    worker = DeleteWorker(graph_store=graph_store)

    worker._process_job(
        _pdf_job(
            source_type="annotation_comment",
            source_id="ann-1",
            metadata={"deleted_pdf_id": "pdf-1"},
        )
    )

    assert worker._qdrant_store.deleted_by_pdf == []
    assert worker._qdrant_store.deleted_by_source == [
        ("user-1", "annotation_comment", "ann-1")
    ]
    assert graph_store.deleted_by_pdf == []
    assert graph_store.deleted_by_source == [("user-1", "annotation_comment", "ann-1")]
    chunk_deletes = [
        entry
        for entry in worker._supabase.log
        if entry["table"] == "rag_chunks"
    ]
    assert chunk_deletes[0]["filters"] == {
        "user_id": "user-1",
        "source_type": "annotation_comment",
        "source_id": "ann-1",
    }
    assert worker.learning_deletes == [("user-1", "annotation_comment", "ann-1")]


def test_delete_source_without_graph_store_skips_graph_layers() -> None:
    worker = DeleteWorker(graph_store=None)

    worker._process_job(_pdf_job())

    # Neo4j is unconfigured in the fake config: no store construction, no crash,
    # learning-layer delete guard also short-circuits via _neo4j_configured.
    assert worker.completed[0][0] == "job-1"


def test_delete_source_unsupported_source_type_completes_with_skip() -> None:
    worker = DeleteWorker()

    worker._process_job(_pdf_job(source_type="chat_memory", source_id="mem-1"))

    assert worker._qdrant_store.deleted_by_pdf == []
    assert worker._qdrant_store.deleted_by_source == []
    assert worker.completed == [
        ("job-1", {"skipped": "unsupported_delete_source_type"})
    ]


def test_delete_source_on_already_clean_source_is_noop_success() -> None:
    worker = DeleteWorker(graph_store=FakeGraphStore())

    worker._process_job(_pdf_job())
    worker._process_job(_pdf_job(id="job-2"))

    assert [job_id for job_id, _ in worker.completed] == ["job-1", "job-2"]
