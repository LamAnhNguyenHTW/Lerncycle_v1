"""Tests for isolated evaluation Concept-Graph indexing."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rag_pipeline.eval.index_graph import index_graph_chunks, load_retry_chunk_ids


class _Extractor:
    def extract_from_chunk(self, chunk):
        if chunk["chunk_id"] == "bad":
            raise RuntimeError("secret raw failure")
        return SimpleNamespace(nodes=[1, 2], edges=[1])


class _Store:
    def __init__(self):
        self.calls = []

    def upsert_extraction(self, user_id, chunk, extraction):
        self.calls.append((user_id, chunk["chunk_id"]))
        return {"nodes_upserted": len(extraction.nodes), "relationships_upserted": len(extraction.edges)}


def test_index_graph_chunks_reports_sanitized_failures() -> None:
    report = index_graph_chunks(
        [
            {"chunk_id": "good", "text": "content"},
            {"chunk_id": "bad", "text": "content"},
        ],
        user_id="eval-user",
        extractor=_Extractor(),
        store=_Store(),
        concurrency=2,
    )

    assert report["chunks_requested"] == 2
    assert report["chunks_indexed"] == 1
    assert report["nodes_upserted"] == 2
    assert report["relationships_upserted"] == 1
    assert report["failures"] == [{"chunk_id": "bad", "error_type": "RuntimeError"}]
    assert "secret raw failure" not in str(report)


def test_load_retry_chunk_ids_reads_only_sanitized_failure_ids(tmp_path) -> None:
    report = tmp_path / "graph-index.json"
    report.write_text(
        json.dumps(
            {
                "stats": {
                    "failures": [
                        {"chunk_id": "b", "error_type": "GraphExtractionError"},
                        {"chunk_id": "a", "error_type": "GraphExtractionError"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_retry_chunk_ids(report) == {"a", "b"}
