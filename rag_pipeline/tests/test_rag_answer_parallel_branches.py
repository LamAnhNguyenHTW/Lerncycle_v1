from __future__ import annotations

import time

from rag_pipeline.rag_answer import answer_with_rag
from rag_pipeline.web_search import WebSearchOutcome


class Llm:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return "Antwort"


class GraphStore:
    def search_concepts(self, **kwargs):
        time.sleep(0.2)
        return [{"name": "Process Mining", "normalized_name": "process mining"}]

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


def test_answer_with_rag_runs_independent_retrieval_branches_concurrently() -> None:
    def retrieval_fn(**kwargs):
        time.sleep(0.2)
        if kwargs.get("source_types") == ["chat_memory"]:
            return [_result("chat_memory")]
        return [_result("pdf")]

    def web_search_fn(**kwargs):
        time.sleep(0.2)
        return WebSearchOutcome([_result("web")], "tavily", 1)

    started = time.perf_counter()

    answer_with_rag(
        "Wie hängt Process Mining mit Event Logs zusammen?",
        "user-1",
        llm_client=Llm(),
        retrieval_fn=retrieval_fn,
        graph_store=GraphStore(),
        graph_retrieval_enabled=True,
        graph_mode="on",
        chat_memory_retrieval_enabled=True,
        memory_mode="on",
        session_id="session-1",
        web_search_enabled=True,
        web_mode="on",
        web_search_fn=web_search_fn,
    )

    assert time.perf_counter() - started <= 0.4
