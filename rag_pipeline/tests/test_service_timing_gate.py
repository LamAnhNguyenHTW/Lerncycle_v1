from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-secret"}


def _payload() -> dict:
    return {
        "query": "Was ist Process Mining?",
        "user_id": "user-1",
        "source_types": ["pdf"],
        "top_k": 8,
    }


def _load_api(monkeypatch, *, debug_enabled: bool):
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("RERANKING_ENABLED", "false")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("RAG_DEBUG_TIMING_ENABLED", "true" if debug_enabled else "false")

    import rag_pipeline.api as api

    importlib.reload(api)
    return api


def test_debug_timing_query_param_is_ignored_when_env_flag_is_false(monkeypatch) -> None:
    api = _load_api(monkeypatch, debug_enabled=False)
    calls = []

    def fake_answer_with_rag(**kwargs):
        calls.append(kwargs)
        return {"answer": "ok", "sources": [], "timing": {"should_not": "leak"}}

    monkeypatch.setattr(api, "answer_with_rag", fake_answer_with_rag)

    response = TestClient(api.app).post(
        "/rag/answer?debug_timing=1",
        headers=_headers(),
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json().get("timing") is None
    assert calls[0]["enable_timing"] is False


def test_debug_timing_query_param_populates_timing_when_env_flag_is_true(monkeypatch) -> None:
    api = _load_api(monkeypatch, debug_enabled=True)
    calls = []

    def fake_answer_with_rag(**kwargs):
        calls.append(kwargs)
        response = {"answer": "ok", "sources": []}
        if kwargs["enable_timing"]:
            response["timing"] = {"stages": [], "total_ms": 1}
        return response

    monkeypatch.setattr(api, "answer_with_rag", fake_answer_with_rag)

    response = TestClient(api.app).post(
        "/rag/answer?debug_timing=1",
        headers=_headers(),
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json()["timing"]["total_ms"] == 1
    assert calls[0]["enable_timing"] is True
