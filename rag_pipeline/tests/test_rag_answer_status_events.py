from __future__ import annotations

import asyncio

from rag_pipeline.rag_answer import stream_answer_with_rag


class StreamingLlmClient:
    def __init__(self, chunks: list[str] | None = None, complete_response: str = "") -> None:
        self.complete_calls = []
        self.stream_calls = []
        self.chunks = chunks or ["Antwort"]
        self.complete_response = complete_response

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.complete_calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.complete_response

    async def stream_answer(self, *, system_prompt: str, user_prompt: str):
        self.stream_calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        for chunk in self.chunks:
            yield {"event_type": "token", "content": chunk}
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
            llm_client=StreamingLlmClient(chunks=["Process Mining verbindet Data Science. [Source 1]"]),
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
            llm_client=StreamingLlmClient(chunks=["Process Mining verbindet Data Science. [Source 1]"]),
            retrieval_fn=lambda **_: [_result()],
        )
    )

    assert [event["event_type"] for event in events] == [
        "status",
        "status",
        "sources",
        "status",
        "token",
        "replace_answer",
        "sources",
        "done",
    ]
    assert events[0]["status"] == "retrieval_started"
    assert events[1]["status"] == "retrieval_completed"
    assert events[2]["sources"] == []
    assert events[3] == {
        "event_type": "status",
        "status": "generation_started",
        "stage": "llm_first_token",
    }
    assert events[4]["content"] == "Process Mining verbindet Data Science. [Source 1]"
    assert events[5]["content"] == "Process Mining verbindet Data Science."
    assert events[6]["sources"][0]["chunk_id"] == "chunk-1"


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


def test_streaming_prompt_capture_does_not_feed_empty_output_to_query_understanding(caplog) -> None:
    events = _collect(
        stream_answer_with_rag(
            "Was ist Process Mining?",
            "user-1",
            llm_client=StreamingLlmClient(
                chunks=["Process Mining analysiert Event Logs."],
                complete_response=(
                    '{"resolved_query":"Was ist Process Mining?",'
                    '"question_type":"document_grounded",'
                    '"route":"internal_retrieval",'
                    '"needs_pdf":true,'
                    '"needs_notes":false,'
                    '"needs_annotations":false,'
                    '"needs_chat_memory":false,'
                    '"needs_graph":false,'
                    '"needs_web":false,'
                    '"should_show_sources":true,'
                    '"confidence":0.9,'
                    '"reasoning_summary":"material question"}'
                ),
            ),
            retrieval_fn=lambda **_: [_result()],
            intent_classifier_enabled=True,
        )
    )

    assert events[-1]["event_type"] == "done"
    assert "Query understanding returned invalid JSON" not in caplog.text


def test_streaming_active_learning_strips_state_from_tokens_and_done_event() -> None:
    events = _collect(
        stream_answer_with_rag(
            "Welche Haupttypen gibt es?",
            "user-1",
            chat_mode="feynman",
            active_learning_state={"mode": "feynman", "learner_name": "Lam Anh"},
            llm_client=StreamingLlmClient(
                [
                    "Das ist fast richtig.\n\n",
                    '<AL_STATE>{"current_step":"ask_types","covered_concepts":["Definition"]}</AL_STATE>',
                ]
            ),
            retrieval_fn=lambda **_: [_result()],
        )
    )

    token_events = [event for event in events if event["event_type"] == "token"]
    done_events = [event for event in events if event["event_type"] == "done"]

    assert token_events == [{"event_type": "token", "content": "Das ist fast richtig."}]
    assert done_events[0]["updated_active_learning_state"]["current_step"] == "ask_types"
    assert done_events[0]["updated_active_learning_state"]["mode"] == "feynman"
