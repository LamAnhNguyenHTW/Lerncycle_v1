from __future__ import annotations

import time

import pytest

from rag_pipeline.rag_answer import answer_with_rag
from rag_pipeline.web_search import WebSearchOutcome


class Llm:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "Antwort"


class SlowGraphStore:
    def search_concepts(self, **kwargs):
        time.sleep(1)
        return []

    def get_neighborhood(self, **kwargs):
        return {"relationships": []}


def _result(source_type: str = "pdf") -> dict:
    return {
        "chunk_id": f"{source_type}-chunk",
        "text": f"{source_type} text",
        "score": 0.9,
        "source_type": source_type,
        "source_id": "source-1",
        "pdf_id": "pdf-1",
        "page_index": 0,
        "title": "Title",
        "heading": None,
        "metadata": {},
    }


def test_graph_branch_timeout_falls_back_to_empty_graph_context() -> None:
    started = time.perf_counter()

    response = answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=Llm(),
        retrieval_fn=lambda **_: [_result()],
        graph_store=SlowGraphStore(),
        graph_retrieval_enabled=True,
        graph_mode="on",
        graph_retrieve_timeout_s=0.05,
    )

    assert response["answer"] == "Antwort"
    assert all(source["source_type"] != "knowledge_graph" for source in response["sources"])
    assert time.perf_counter() - started < 0.5


def test_memory_branch_timeout_falls_back_to_empty_memory_results() -> None:
    def retrieval_fn(**kwargs):
        if kwargs.get("source_types") == ["chat_memory"]:
            time.sleep(1)
            return [_result("chat_memory")]
        return [_result()]

    response = answer_with_rag(
        "Was hatten wir besprochen?",
        "user-1",
        llm_client=Llm(),
        retrieval_fn=retrieval_fn,
        chat_memory_retrieval_enabled=True,
        memory_mode="on",
        session_id="session-1",
        memory_retrieve_timeout_s=0.05,
    )

    assert response["answer"] == "Antwort"
    assert all(source["source_type"] != "chat_memory" for source in response["sources"])


def test_web_branch_timeout_falls_back_to_empty_web_results() -> None:
    def web_search_fn(**kwargs):
        time.sleep(1)
        return WebSearchOutcome([_result("web")], "tavily", 1)

    response = answer_with_rag(
        "Was ist aktuell neu?",
        "user-1",
        llm_client=Llm(),
        retrieval_fn=lambda **_: [_result()],
        web_search_enabled=True,
        web_mode="on",
        web_search_fn=web_search_fn,
        web_retrieve_timeout_s=0.05,
    )

    assert response["answer"] == "Antwort"
    assert all(source["source_type"] != "web" for source in response["sources"])


def test_vector_branch_timeout_raises() -> None:
    def retrieval_fn(**kwargs):
        time.sleep(1)
        return [_result()]

    with pytest.raises(TimeoutError):
        answer_with_rag(
            "Was ist Process Mining?",
            "user-1",
            llm_client=Llm(),
            retrieval_fn=retrieval_fn,
            vector_retrieve_timeout_s=0.05,
        )
