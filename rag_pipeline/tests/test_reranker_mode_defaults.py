from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    monkeypatch.setenv("RERANKING_ENABLED", "false")
    monkeypatch.setenv("RERANKING_PROVIDER", "fastembed")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "false")
    import rag_pipeline.api as api

    importlib.reload(api)
    return TestClient(api.app), api


def _payload(**overrides):
    payload = {
        "query": "Was ist Process Mining?",
        "user_id": "user-1",
        "source_types": ["pdf"],
    }
    payload.update(overrides)
    return payload


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-secret"}


def test_normal_mode_defaults_to_noop_with_candidate_cap(client, monkeypatch) -> None:
    test_client, api = client
    reranker_calls = []
    answer_calls = []
    monkeypatch.setenv("RERANKING_ENABLED", "true")
    monkeypatch.setenv("RERANKER_NORMAL_MODE_DEFAULT", "noop")
    monkeypatch.setattr(
        api,
        "create_reranker",
        lambda **kwargs: reranker_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        api,
        "answer_with_rag",
        lambda **kwargs: answer_calls.append(kwargs) or {"answer": "ok", "sources": []},
    )

    response = test_client.post("/rag/answer", headers=_headers(), json=_payload())

    assert response.status_code == 200
    assert reranker_calls[0]["provider"] == "noop"
    assert reranker_calls[0]["enabled"] is False
    assert answer_calls[0]["reranking_candidate_k"] <= 15
    assert answer_calls[0]["reranking_enabled"] is False


def test_normal_mode_can_opt_into_fast_reranker(client, monkeypatch) -> None:
    test_client, api = client
    reranker_calls = []
    answer_calls = []
    monkeypatch.setenv("RERANKING_ENABLED", "true")
    monkeypatch.setattr(
        api,
        "create_reranker",
        lambda **kwargs: reranker_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        api,
        "answer_with_rag",
        lambda **kwargs: answer_calls.append(kwargs) or {"answer": "ok", "sources": []},
    )

    response = test_client.post(
        "/rag/answer",
        headers=_headers(),
        json=_payload(use_fast_reranker=True),
    )

    assert response.status_code == 200
    assert reranker_calls[0]["provider"] == "fastembed"
    assert reranker_calls[0]["enabled"] is True
    assert answer_calls[0]["reranking_candidate_k"] <= 15


def test_deep_mode_uses_fast_reranker_with_candidate_cap(client, monkeypatch) -> None:
    test_client, api = client
    reranker_calls = []
    answer_calls = []
    monkeypatch.setattr(
        api,
        "create_reranker",
        lambda **kwargs: reranker_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        api,
        "answer_with_rag",
        lambda **kwargs: answer_calls.append(kwargs) or {"answer": "ok", "sources": []},
    )

    response = test_client.post(
        "/rag/answer",
        headers=_headers(),
        json=_payload(retrieval_mode="deep"),
    )

    assert response.status_code == 200
    assert reranker_calls[0]["provider"] == "fastembed"
    assert reranker_calls[0]["enabled"] is True
    assert answer_calls[0]["reranking_candidate_k"] <= 30


def test_deep_mode_can_opt_into_llm_reranker(client, monkeypatch) -> None:
    test_client, api = client
    reranker_calls = []
    monkeypatch.setattr(
        api,
        "create_reranker",
        lambda **kwargs: reranker_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        api,
        "answer_with_rag",
        lambda **kwargs: {"answer": "ok", "sources": []},
    )

    response = test_client.post(
        "/rag/answer",
        headers=_headers(),
        json=_payload(retrieval_mode="deep", use_llm_reranker=True),
    )

    assert response.status_code == 200
    assert reranker_calls[0]["provider"] == "llm"
