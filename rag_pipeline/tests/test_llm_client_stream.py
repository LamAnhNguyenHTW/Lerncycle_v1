from __future__ import annotations

import asyncio

import pytest

from rag_pipeline.llm_client import OpenAILlmClient


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _StreamingCompletions:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.chunks)


class _Chat:
    def __init__(self, completions: _StreamingCompletions) -> None:
        self.completions = completions


class _OpenAIClient:
    def __init__(self, chunks) -> None:
        self.chat = _Chat(_StreamingCompletions(chunks))


def _collect(async_iterable):
    async def run():
        return [event async for event in async_iterable]

    return asyncio.run(run())


def test_stream_answer_yields_token_events_for_each_openai_chunk() -> None:
    openai_client = _OpenAIClient([_Chunk("Hel"), _Chunk("lo"), _Chunk(None)])
    llm = OpenAILlmClient(api_key="test", openai_client=openai_client)

    events = _collect(llm.stream_answer(system_prompt="sys", user_prompt="user"))

    assert events[:2] == [
        {"event_type": "token", "content": "Hel"},
        {"event_type": "token", "content": "lo"},
    ]
    assert events[-1]["event_type"] == "done"
    assert openai_client.chat.completions.calls[0]["stream"] is True


def test_stream_answer_yields_done_event_with_usage_info_when_available() -> None:
    chunk = _Chunk("")
    chunk.usage = {"total_tokens": 12}
    llm = OpenAILlmClient(api_key="test", openai_client=_OpenAIClient([chunk]))

    events = _collect(llm.stream_answer(system_prompt="sys", user_prompt="user"))

    assert events[-1] == {"event_type": "done", "usage": {"total_tokens": 12}}


def test_stream_answer_yields_error_event_and_stops_on_mid_stream_error() -> None:
    class BrokenStream:
        def __iter__(self):
            yield _Chunk("partial")
            raise RuntimeError("provider failed")

    llm = OpenAILlmClient(api_key="test", openai_client=_OpenAIClient(BrokenStream()))

    events = _collect(llm.stream_answer(system_prompt="sys", user_prompt="user"))

    assert events == [
        {"event_type": "token", "content": "partial"},
        {"event_type": "error", "message": "provider failed"},
    ]
