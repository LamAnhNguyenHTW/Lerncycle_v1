"""Shared non-streaming LLM clients for the RAG pipeline."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

StreamEvent = dict[str, Any]


class OpenAILlmClient:
    """Minimal non-streaming OpenAI chat client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        openai_client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._openai_client = openai_client
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI LLM calls. "
                "Set OPENAI_API_KEY or inject an llm_client for offline tests."
            )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def stream_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[StreamEvent]:
        """Yield OpenAI chat completion chunks as normalized stream events."""
        usage = None
        try:
            stream = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = _jsonable_usage(chunk_usage)
                content = _chunk_content(chunk)
                if content:
                    yield {"event_type": "token", "content": content}
        except Exception as exc:
            yield {"event_type": "error", "message": str(exc)}
            return
        done: StreamEvent = {"event_type": "done"}
        if usage is not None:
            done["usage"] = usage
        yield done

    def _client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client
        from openai import OpenAI

        self._openai_client = OpenAI(api_key=self.api_key)
        return self._openai_client


def _chunk_content(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


def _jsonable_usage(usage: Any) -> Any:
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return usage
