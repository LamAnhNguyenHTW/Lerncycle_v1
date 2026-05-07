from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    import rag_pipeline.api as api

    importlib.reload(api)
    return TestClient(api.app), api


def _payload(**overrides):
    payload = {
        "query": "Was ist Process Mining?",
        "user_id": "user-1",
        "source_types": ["pdf", "note", "annotation_comment"],
        "top_k": 8,
    }
    payload.update(overrides)
    return payload


def _headers(token: str = "test-secret") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_returns_ok(client) -> None:
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rag_answer_requires_auth(client) -> None:
    test_client, _ = client

    response = test_client.post("/rag/answer", json=_payload())

    assert response.status_code == 401


def test_rag_answer_rejects_invalid_token(client) -> None:
    test_client, _ = client

    response = test_client.post("/rag/answer", headers=_headers("wrong"), json=_payload())

    assert response.status_code == 401


def test_rag_answer_calls_answer_with_rag(client, monkeypatch) -> None:
    test_client, api = client
    calls = []

    def fake_answer_with_rag(**kwargs):
        calls.append(kwargs)
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(api, "answer_with_rag", fake_answer_with_rag)

    response = test_client.post("/rag/answer", headers=_headers(), json=_payload())

    assert response.status_code == 200
    assert calls


def test_rag_answer_passes_user_id_from_internal_request(client, monkeypatch) -> None:
    test_client, api = client
    calls = []
    monkeypatch.setattr(
        api,
        "answer_with_rag",
        lambda **kwargs: calls.append(kwargs) or {"answer": "ok", "sources": []},
    )

    test_client.post("/rag/answer", headers=_headers(), json=_payload(user_id="user-2"))

    assert calls[0]["user_id"] == "user-2"


def test_rag_answer_validates_source_types(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/rag/answer",
        headers=_headers(),
        json=_payload(source_types=["web"]),
    )

    assert response.status_code == 422


def test_rag_answer_validates_top_k_bounds(client) -> None:
    test_client, _ = client

    low = test_client.post("/rag/answer", headers=_headers(), json=_payload(top_k=0))
    high = test_client.post("/rag/answer", headers=_headers(), json=_payload(top_k=21))

    assert low.status_code == 422
    assert high.status_code == 422


def test_rag_answer_returns_answer_and_sources(client, monkeypatch) -> None:
    test_client, api = client
    monkeypatch.setattr(
        api,
        "answer_with_rag",
        lambda **_: {"answer": "Antwort", "sources": [{"chunk_id": "chunk-1"}]},
    )

    response = test_client.post("/rag/answer", headers=_headers(), json=_payload())

    assert response.status_code == 200
    assert response.json()["answer"] == "Antwort"
    assert response.json()["sources"] == [{"chunk_id": "chunk-1"}]


def test_rag_answer_handles_internal_error_cleanly(client, monkeypatch) -> None:
    test_client, api = client

    def boom(**_):
        raise RuntimeError("secret stack detail")

    monkeypatch.setattr(api, "answer_with_rag", boom)

    response = test_client.post("/rag/answer", headers=_headers(), json=_payload())

    assert response.status_code == 500
    assert "secret stack detail" not in response.text


def test_rag_answer_rejects_query_over_2000_chars(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/rag/answer",
        headers=_headers(),
        json=_payload(query="x" * 2001),
    )

    assert response.status_code == 422
