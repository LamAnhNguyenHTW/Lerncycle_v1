from __future__ import annotations

import importlib
import json

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


def _load_api(monkeypatch):
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("RERANKING_ENABLED", "false")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "false")

    import rag_pipeline.api as api

    importlib.reload(api)
    return api


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data_lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data_lines:
            events.append(json.loads("\n".join(data_lines)))
    return events


def test_stream_endpoint_returns_sse_events_in_order(monkeypatch) -> None:
    api = _load_api(monkeypatch)

    async def fake_stream_answer_with_rag(**_):
        yield {"event_type": "status", "status": "retrieval_started", "stage": "qdrant_retrieve"}
        yield {"event_type": "status", "status": "retrieval_completed", "stage": "qdrant_retrieve"}
        yield {"event_type": "sources", "sources": [{"chunk_id": "chunk-1"}]}
        yield {"event_type": "status", "status": "generation_started", "stage": "llm_first_token"}
        yield {"event_type": "token", "content": "Antwort"}
        yield {"event_type": "done"}

    monkeypatch.setattr(api, "stream_answer_with_rag", fake_stream_answer_with_rag)

    response = TestClient(api.app).post(
        "/rag/answer/stream",
        headers=_headers(),
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert [event["event_type"] for event in events] == [
        "status",
        "status",
        "sources",
        "status",
        "token",
        "done",
    ]
    assert events[0]["status"] == "retrieval_started"


def test_stream_endpoint_requires_internal_auth(monkeypatch) -> None:
    api = _load_api(monkeypatch)

    response = TestClient(api.app).post("/rag/answer/stream", json=_payload())

    assert response.status_code == 401
