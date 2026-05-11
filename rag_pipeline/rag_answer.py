"""Single-turn RAG answer generation helper."""

from __future__ import annotations

import os
import logging
from typing import Any, Callable

from rag_pipeline.context_builder import build_rag_context
from rag_pipeline.retrieval import search_hybrid_chunks


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are LearnCycle's learning assistant. Use only the provided context from the "
    "user's uploaded PDFs, notes, and annotations. If the context does not contain "
    "enough information, say so clearly. Do not invent facts or sources. Do not begin "
    'your answer with "Im bereitgestellten Kontext". Prefer German when the user asks '
    "in German. Give a helpful learning-oriented explanation."
)
REWRITE_SYSTEM_PROMPT = (
    "Rewrite the current user question into a standalone retrieval query using the recent "
    "conversation only to resolve references. Output only the rewritten query, no "
    "explanation. Keep it concise, at most two short sentences or about 200 characters. "
    "Preserve the user's language. Do not answer the question."
)
CONVERSATION_SYSTEM_PROMPT = (
    "You are LearnCycle's learning assistant. Use the retrieved context as the factual "
    "source of truth. The recent conversation is provided only for continuity and "
    "resolving references. If the retrieved context does not contain enough information, "
    "say so clearly. Do not invent facts or sources. Do not begin your answer with "
    '"Im bereitgestellten Kontext". Do not mention raw source metadata. '
    'If the user asks to simplify ("einfacher", "kurz", "für Anfänger", "nochmal"): '
    "use short sentences, explain only the main idea first, add one concrete example "
    "if helpful, and do not enumerate unrelated topics from the context. Prefer German "
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


def _format_conversation_block(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role_label = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {message['content'][:2000]}")
    return "\n".join(lines)


def _is_vague_followup(query: str) -> bool:
    normalized = query.lower()
    cues = [
        "das",
        "davon",
        "dazu",
        "hier",
        "nochmal",
        "noch mal",
        "einfacher",
        "kurz",
        "anfänger",
        "anfaenger",
        "it",
        "this",
        "that",
        "again",
        "simpler",
        "shorter",
    ]
    return any(cue in normalized for cue in cues)


def rewrite_query_for_retrieval(
    query: str,
    recent_messages: list[dict] | None = None,
    llm_client: Any = None,
) -> str:
    """Rewrite vague follow-up questions into standalone retrieval queries."""
    if not recent_messages or not _is_vague_followup(query):
        return query

    try:
        active_llm = llm_client or OpenAILlmClient()
        user_prompt = (
            f"Recent conversation:\n{_format_conversation_block(recent_messages)}\n\n"
            f"Current query:\n{query}\n\n"
            "Standalone retrieval query:"
        )
        rewritten = active_llm.complete(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ).strip()
        return rewritten[:200] or query
    except Exception as exc:
        logger.warning("Query rewriting failed; using original query.", exc_info=exc)
        return query


def answer_with_rag(
    query: str,
    user_id: str,
    source_types: list[str] | None = None,
    top_k: int = 8,
    pdf_ids: list[str] | None = None,
    recent_messages: list[dict] | None = None,
    llm_client: Any = None,
    retrieval_fn: Callable[..., list[dict[str, Any]]] | None = None,
    reranker: Any = None,
    reranking_enabled: bool = False,
    reranking_candidate_k: int = 30,
    reranking_top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve user-scoped chunks, generate an answer, and return citations."""
    active_retrieval = retrieval_fn or search_hybrid_chunks
    retrieval_query = rewrite_query_for_retrieval(query, recent_messages, llm_client)
    retrieval_top_k = reranking_candidate_k if reranking_enabled and reranker is not None else top_k
    results = active_retrieval(
        query=retrieval_query,
        user_id=user_id,
        source_types=source_types,
        top_k=retrieval_top_k,
        pdf_ids=pdf_ids,
    )
    if not results:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context_results = results
    context_top_k = top_k
    if reranking_enabled and reranker is not None:
        context_top_k = reranking_top_k
        try:
            context_results = reranker.rerank(
                retrieval_query,
                results,
                top_k=reranking_top_k,
            )
        except Exception:
            logger.warning("Reranker failed; using original retrieval order.", exc_info=True)
            context_results = results[:reranking_top_k]

    context = build_rag_context(context_results, max_chunks=context_top_k)
    if not context["context_text"]:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    active_llm = llm_client or OpenAILlmClient()
    if recent_messages:
        user_prompt = (
            f"Recent conversation:\n{_format_conversation_block(recent_messages)}\n\n"
            f"Retrieved context:\n{context['context_text']}\n\n"
            f"Current question:\n{query}"
        )
        system_prompt = CONVERSATION_SYSTEM_PROMPT
    else:
        user_prompt = (
            "Answer the user's question using only this context.\n\n"
            f"Context:\n{context['context_text']}\n\n"
            f"Question:\n{query}"
        )
        system_prompt = SYSTEM_PROMPT
    answer = active_llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    return {"answer": answer, "sources": context["sources"]}
