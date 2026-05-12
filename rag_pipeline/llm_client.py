"""Shared non-streaming LLM clients for the RAG pipeline."""

from __future__ import annotations

import os


class OpenAILlmClient:
    """Minimal non-streaming OpenAI chat client."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI LLM calls. "
                "Set OPENAI_API_KEY or inject an llm_client for offline tests."
            )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
