from __future__ import annotations

from rag_pipeline.rag_answer import answer_with_rag


class FakeLlmClient:
    def __init__(self, answer: str = "Antwort") -> None:
        self.answer = answer
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.answer


def _result() -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": "Process Mining verbindet Data Science und Process Science.",
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "page_index": 9,
        "title": "GPAA.pdf",
        "heading": "Definition",
        "metadata": {"filename": "GPAA.pdf"},
    }


def test_answer_with_rag_calls_hybrid_retrieval_with_user_id() -> None:
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag("Was ist Process Mining?", "user-1", llm_client=FakeLlmClient(), retrieval_fn=retrieval_fn)

    assert calls[0]["user_id"] == "user-1"
    assert calls[0]["query"] == "Was ist Process Mining?"


def test_answer_with_rag_builds_context() -> None:
    llm = FakeLlmClient()

    answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [_result()])

    assert "Type: PDF" in llm.calls[0]["user_prompt"]
    assert "Page: 10" in llm.calls[0]["user_prompt"]


def test_answer_with_rag_returns_answer_and_sources() -> None:
    response = answer_with_rag(
        "Frage",
        "user-1",
        llm_client=FakeLlmClient("Eine Antwort"),
        retrieval_fn=lambda **_: [_result()],
    )

    assert response["answer"] == "Eine Antwort"
    assert response["sources"][0]["chunk_id"] == "chunk-1"


def test_answer_with_rag_handles_no_results() -> None:
    llm = FakeLlmClient()

    response = answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [])

    assert response["sources"] == []
    assert "keine passenden Quellen" in response["answer"]
    assert llm.calls == []


def test_answer_with_rag_does_not_call_llm_when_no_context() -> None:
    llm = FakeLlmClient()

    answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [])

    assert len(llm.calls) == 0


def test_answer_with_rag_passes_source_types_to_retrieval() -> None:
    calls = []

    def retrieval_fn(**kwargs):
        calls.append(kwargs)
        return [_result()]

    answer_with_rag(
        "Frage",
        "user-1",
        source_types=["note"],
        llm_client=FakeLlmClient(),
        retrieval_fn=retrieval_fn,
    )

    assert calls[0]["source_types"] == ["note"]


def test_answer_with_rag_uses_injected_llm_client() -> None:
    llm = FakeLlmClient("Custom")

    response = answer_with_rag("Frage", "user-1", llm_client=llm, retrieval_fn=lambda **_: [_result()])

    assert response["answer"] == "Custom"
    assert len(llm.calls) == 1
