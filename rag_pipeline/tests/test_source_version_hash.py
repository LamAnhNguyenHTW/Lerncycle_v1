from __future__ import annotations

from rag_pipeline.result_cache import build_source_version_hash


def test_source_version_hash_is_stable_and_order_independent() -> None:
    rows = [
        {"source_id": "note-1", "source_type": "note", "updated_at": "2026-01-01"},
        {"source_id": "pdf-1", "source_type": "pdf", "content_hash": "hash-1"},
    ]

    assert build_source_version_hash(source_rows=rows) == build_source_version_hash(source_rows=list(reversed(rows)))


def test_source_version_hash_changes_for_source_metadata_updates() -> None:
    base = [{"source_id": "note-1", "source_type": "note", "updated_at": "2026-01-01"}]
    changed = [{"source_id": "note-1", "source_type": "note", "updated_at": "2026-01-02"}]

    assert build_source_version_hash(source_rows=base) != build_source_version_hash(source_rows=changed)


def test_source_version_hash_uses_only_completed_index_source_jobs() -> None:
    base = [
        {
            "source_id": "pdf-1",
            "content_hash": "hash-1",
            "index_jobs": [
                {"job_kind": "extract_learning_graph", "status": "completed", "updated_at": "2026-01-03"},
                {"job_kind": "index_source", "status": "failed", "updated_at": "2026-01-04"},
            ],
        }
    ]
    changed = [
        {
            "source_id": "pdf-1",
            "content_hash": "hash-1",
            "index_jobs": [
                {"job_kind": "index_source", "status": "completed", "updated_at": "2026-01-05"},
            ],
        }
    ]

    assert build_source_version_hash(source_rows=base) != build_source_version_hash(source_rows=changed)
    assert build_source_version_hash(source_rows=base) == build_source_version_hash(
        source_rows=[
            {
                "source_id": "pdf-1",
                "content_hash": "hash-1",
                "index_jobs": [
                    {"job_kind": "extract_learning_graph", "status": "completed", "updated_at": "2026-02-03"},
                ],
            }
        ]
    )
