from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_healthz_exposes_cache_metrics(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    import rag_pipeline.api as api

    importlib.reload(api)
    response = TestClient(api.app).get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert "query_embedding_cache_hits" in data
    assert "query_embedding_cache_misses" in data
    assert "retrieval_result_cache_hits" in data
    assert "retrieval_result_cache_misses" in data
    assert "reranker_cache_hits" in data
    assert "reranker_cache_misses" in data
