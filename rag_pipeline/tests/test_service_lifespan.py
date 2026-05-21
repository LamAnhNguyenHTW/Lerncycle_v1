from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_runtime_clients_are_constructed_once(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.local")
    import rag_pipeline.api as api

    importlib.reload(api)
    calls = {"openai": 0, "qdrant": 0}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["openai"] += 1

    class FakeQdrant:
        def __init__(self, **kwargs):
            calls["qdrant"] += 1

    monkeypatch.setattr(api, "OpenAILlmClient", FakeOpenAI)
    monkeypatch.setattr(api, "QdrantStore", FakeQdrant)

    with TestClient(api.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200

    assert calls == {"openai": 1, "qdrant": 1}
