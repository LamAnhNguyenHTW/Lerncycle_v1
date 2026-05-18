from __future__ import annotations

import asyncio

from rag_pipeline.rag_answer import stream_answer_with_rag


class StreamingLlmClient:
    def __init__(self) -> None:
        self.complete_calls = []
        self.stream_calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.complete_calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return ""

    async def stream_answer(self, *, system_prompt: str, user_prompt: str):
        self.stream_calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        yield {"event_type": "token", "content": "Antwort"}
        yield {"event_type": "done"}


def _result() -> dict:
    return {
        "chunk_id": "chunk-1",
        "text": "Process Mining verbindet Data Science und Process Science.",
        "score": 0.9,
        "source_type": "pdf",
        "source_id": "pdf-1",
        "pdf_id": "pdf-1",
        "page_index": 0,
        "title": "GPAA.pdf",
        "heading": "Definition",
        "metadata": {"filename": "GPAA.pdf"},
    }


def _collect(async_iterable):
    async def run():
        return [event async for event in async_iterable]

    return asyncio.run(run())


def test_streaming_pipeline_emits_retrieval_started_before_retrieval_work() -> None:
    events_seen = []

    def retrieval_fn(**_):
        events_seen.append("retrieval_called")
        return [_result()]

    async def first_two_events():
        stream = stream_answer_with_rag(
            "Was ist Process Mining?",
            "user-1",
            llm_client=StreamingLlmClient(),
            retrieval_fn=retrieval_fn,
        )
        first = await anext(stream)
        return first, list(events_seen)

    first, seen_at_first_event = asyncio.run(first_two_events())

    assert first == {
        "event_type": "status",
        "status": "retrieval_started",
        "stage": "qdrant_retrieve",
    }
    assert seen_at_first_event == []


def test_streaming_pipeline_emits_status_sources_tokens_and_done_in_order() -> None:
    events = _collect(
        stream_answer_with_rag(
            "Was ist Process Mining?",
            "user-1",
            llm_client=StreamingLlmClient(),
            retrieval_fn=lambda **_: [_result()],
        )
    )

    assert [event["event_type"] for event in events] == [
        "status",
        "status",
        "sources",
        "status",
        "token",
        "done",
    ]
    assert events[0]["status"] == "retrieval_started"
    assert events[1]["status"] == "retrieval_completed"
    assert events[2]["sources"][0]["chunk_id"] == "chunk-1"
    assert events[3] == {
        "event_type": "status",
        "status": "generation_started",
        "stage": "llm_first_token",
    }


def test_streaming_pipeline_emits_reranking_started_when_reranker_configured() -> None:
    class Reranker:
        def rerank(self, query, results, top_k):
            return results[:top_k]

    events = _collect(
        stream_answer_with_rag(
            "Was ist Process Mining?",
            "user-1",
            llm_client=StreamingLlmClient(),
            retrieval_fn=lambda **_: [_result()],
            reranker=Reranker(),
            reranking_enabled=True,
        )
    )

    statuses = [event["status"] for event in events if event["event_type"] == "status"]
    assert statuses == [
        "retrieval_started",
        "retrieval_completed",
        "reranking_started",
        "generation_started",
    ]
