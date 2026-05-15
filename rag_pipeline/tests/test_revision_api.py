from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


USER_ID = "00000000-0000-0000-0000-000000000001"
PDF_ID = "00000000-0000-0000-0000-0000000000aa"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("RAG_INTERNAL_API_KEY", "test-secret")
    monkeypatch.setenv("RERANKING_ENABLED", "false")
    monkeypatch.setenv("RERANKING_PROVIDER", "fastembed")
    monkeypatch.setenv("GRAPH_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("GRAPH_ENABLED", "false")
    monkeypatch.setenv("GRAPH_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    import rag_pipeline.api as api

    importlib.reload(api)
    return TestClient(api.app), api


def _headers(token: str = "test-secret") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _flashcards_payload(**overrides):
    payload = {"user_id": USER_ID, "pdf_ids": [PDF_ID], "count": 5, "language": "de"}
    payload.update(overrides)
    return payload


def test_flashcards_requires_auth(client) -> None:
    test_client, _ = client
    response = test_client.post("/revision/flashcards", json=_flashcards_payload())
    assert response.status_code == 401


def test_flashcards_rejects_invalid_token(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers("wrong"),
        json=_flashcards_payload(),
    )
    assert response.status_code == 401


def test_flashcards_validates_count_lower_bound(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers(),
        json=_flashcards_payload(count=0),
    )
    assert response.status_code == 422


def test_flashcards_validates_user_id_uuid(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers(),
        json=_flashcards_payload(user_id="not-a-uuid"),
    )
    assert response.status_code == 422


def test_flashcards_validates_pdf_ids_uuid(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers(),
        json=_flashcards_payload(pdf_ids=["not-a-uuid"]),
    )
    assert response.status_code == 422


def test_flashcards_requires_at_least_one_pdf(client) -> None:
    test_client, _ = client
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers(),
        json=_flashcards_payload(pdf_ids=[]),
    )
    assert response.status_code == 422


def test_flashcards_calls_generator(client, monkeypatch) -> None:
    test_client, api = client
    captured: dict = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        from rag_pipeline.revision.models import FlashcardBatch, GeneratedFlashcard
        return FlashcardBatch(
            cards=[GeneratedFlashcard(front="Q", back="A", source_chunk_ids=["c-1"])]
        )

    monkeypatch.setattr(api, "generate_flashcards", fake_generate)
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers(),
        json=_flashcards_payload(count=3),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cards"] == [
        {"front": "Q", "back": "A", "source_chunk_ids": ["c-1"]}
    ]
    assert captured["user_id"] == USER_ID
    assert captured["pdf_ids"] == [PDF_ID]
    assert captured["count"] == 3


def test_flashcards_rejects_count_above_config_max(client, monkeypatch) -> None:
    monkeypatch.setenv("REVISION_MAX_CARDS_PER_DECK", "10")
    import rag_pipeline.api as api
    importlib.reload(api)
    test_client = TestClient(api.app)
    response = test_client.post(
        "/revision/flashcards",
        headers=_headers(),
        json=_flashcards_payload(count=25),
    )
    assert response.status_code == 422


def test_mocktest_requires_auth(client) -> None:
    test_client, _ = client
    response = test_client.post("/revision/mocktest", json=_flashcards_payload())
    assert response.status_code == 401


def test_mocktest_calls_generator(client, monkeypatch) -> None:
    test_client, api = client

    def fake_generate(**_):
        from rag_pipeline.revision.models import GeneratedMockQuestion, MockTestBatch
        return MockTestBatch(
            questions=[
                GeneratedMockQuestion(
                    prompt="Q?",
                    choices=["a", "b", "c", "d"],
                    correct_index=1,
                    explanation="exp",
                    source_chunk_ids=["c-1"],
                )
            ]
        )

    monkeypatch.setattr(api, "generate_mock_test", fake_generate)
    response = test_client.post(
        "/revision/mocktest",
        headers=_headers(),
        json=_flashcards_payload(count=5, language="en"),
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["questions"]) == 1
    q = body["questions"][0]
    assert q["choices"] == ["a", "b", "c", "d"]
    assert q["correct_index"] == 1
