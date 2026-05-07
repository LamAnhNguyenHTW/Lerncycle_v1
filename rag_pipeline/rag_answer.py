"""Single-turn RAG answer generation helper."""

from __future__ import annotations

import os
from typing import Any, Callable

from rag_pipeline.context_builder import build_rag_context
from rag_pipeline.retrieval import search_hybrid_chunks


SYSTEM_PROMPT = (
    "You are LearnCycle's learning assistant. Use only the provided context from the "
    "user's uploaded PDFs, notes, and annotations. If the context does not contain "
    "enough information, say so clearly. Do not invent facts or sources. Prefer German "
    "when the user asks in German. Give a helpful learning-oriented explanation."
)

FALLBACK_ANSWER = (
    "Ich habe in deinen Materialien keine passenden Quellen gefunden. "
    "Bitte formuliere die Frage etwas konkreter oder lade passende Unterlagen hoch."
)


class OpenAILlmClient:
    """Minimal non-streaming OpenAI chat client."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for RAG answer generation.")

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


def answer_with_rag(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 8,
    pdf_ids: list[str] | None = None,
    llm_client: Any = None,
    retrieval_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Retrieve user-scoped chunks, generate an answer, and return citations."""
    active_retrieval = retrieval_fn or search_hybrid_chunks
    results = active_retrieval(
        query=query,
        user_id=user_id,
        source_types=source_types,
        top_k=top_k,
        pdf_ids=pdf_ids,
    )
    if not results:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context = build_rag_context(results, max_chunks=top_k)
    if not context["context_text"]:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    active_llm = llm_client or OpenAILlmClient()
    user_prompt = (
        "Answer the user's question using only this context.\n\n"
        f"Context:\n{context['context_text']}\n\n"
        f"Question:\n{query}"
    )
    answer = active_llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    return {"answer": answer, "sources": context["sources"]}
