from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_healthz_returns_uptime_cache_metrics_and_upstream_flags(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.local")
    import rag_pipeline.api as api

    importlib.reload(api)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    class FakeQdrant:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(api, "OpenAILlmClient", FakeOpenAI)
    monkeypatch.setattr(api, "QdrantStore", FakeQdrant)

    response = TestClient(api.app).get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["uptime_s"] >= 0
    assert data["qdrant_ok"] is True
    assert data["openai_ok"] is True
    assert "query_embedding_cache_hits" in data
    assert "retrieval_result_cache_hits" in data
    assert "reranker_cache_hits" in data
