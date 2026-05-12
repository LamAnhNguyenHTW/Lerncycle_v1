"""Single-turn RAG answer generation helper."""

from __future__ import annotations

import logging
from typing import Any, Callable

from rag_pipeline.context_builder import build_rag_context
from rag_pipeline.graph_retrieval import detect_graph_intent
from rag_pipeline.graph_retrieval import retrieve_graph_context
from rag_pipeline.llm_client import OpenAILlmClient
from rag_pipeline.memory_intent import detect_memory_intent
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.source_types import MATERIAL_SOURCE_TYPES


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
GRAPH_SYSTEM_PROMPT_ADDITION = (
    " Text chunks are the primary factual grounding. Knowledge graph context may "
    "explain relationships and structure backed by source chunks. If graph context "
    "conflicts with text chunks, prefer the text chunks. Do not treat graph edges as "
    "facts unless they are backed by cited chunks."
)

FALLBACK_ANSWER = (
    "Ich habe in deinen Materialien keine passenden Quellen gefunden. "
    "Bitte formuliere die Frage etwas konkreter oder lade passende Unterlagen hoch."
)


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
    session_id: str | None = None,
    memory_mode: str = "auto",
    chat_memory_retrieval_enabled: bool | None = None,
    chat_memory_top_k: int | None = None,
    memory_source_ids: list[str] | None = None,
    graph_retrieval_enabled: bool | None = None,
    graph_mode: str = "auto",
    graph_top_k: int | None = None,
    graph_store: Any = None,
) -> dict[str, Any]:
    """Retrieve user-scoped chunks, generate an answer, and return citations."""
    active_retrieval = retrieval_fn or search_hybrid_chunks
    retrieval_query = rewrite_query_for_retrieval(query, recent_messages, llm_client)
    retrieval_top_k = reranking_candidate_k if reranking_enabled and reranker is not None else top_k
    material_source_types = source_types or list(MATERIAL_SOURCE_TYPES)
    results = active_retrieval(
        query=retrieval_query,
        user_id=user_id,
        source_types=material_source_types,
        top_k=retrieval_top_k,
        pdf_ids=pdf_ids,
    )
    graph_context = {"context_text": "", "sources": []}
    if _should_retrieve_graph(query, graph_mode, graph_retrieval_enabled, graph_store):
        try:
            graph_context = retrieve_graph_context(
                query=query,
                user_id=user_id,
                source_types=material_source_types,
                source_ids=pdf_ids,
                top_k=graph_top_k or 8,
                graph_store=graph_store,
            )
        except Exception:
            logger.warning("Graph retrieval failed; continuing without graph context.", exc_info=True)
            graph_context = {"context_text": "", "sources": []}
    memory_results: list[dict[str, Any]] = []
    if _should_retrieve_memory(
        query,
        recent_messages,
        session_id,
        memory_mode,
        chat_memory_retrieval_enabled,
    ):
        try:
            memory_ids = _memory_source_ids(session_id, memory_source_ids)
            memory_results = active_retrieval(
                query=retrieval_query,
                user_id=user_id,
                source_types=["chat_memory"],
                source_ids=memory_ids,
                top_k=chat_memory_top_k or 2,
                pdf_ids=None,
            )
        except Exception:
            logger.warning("Chat memory retrieval failed; continuing without memory.", exc_info=True)
            memory_results = []
    results = results + memory_results
    if not results and not graph_context.get("context_text"):
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
    text_context = context["context_text"]
    graph_text = str(graph_context.get("context_text") or "").strip()
    combined_context = _combine_context(text_context, graph_text)
    if not combined_context:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    active_llm = llm_client or OpenAILlmClient()
    if recent_messages:
        user_prompt = (
            f"Recent conversation:\n{_format_conversation_block(recent_messages)}\n\n"
            f"Retrieved context:\n{combined_context}\n\n"
            f"Current question:\n{query}"
        )
        system_prompt = CONVERSATION_SYSTEM_PROMPT + (GRAPH_SYSTEM_PROMPT_ADDITION if graph_text else "")
    else:
        user_prompt = (
            "Answer the user's question using only this context.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{query}"
        )
        system_prompt = SYSTEM_PROMPT + (GRAPH_SYSTEM_PROMPT_ADDITION if graph_text else "")
    answer = active_llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    return {"answer": answer, "sources": context["sources"] + list(graph_context.get("sources") or [])}


def _should_retrieve_memory(
    query: str,
    recent_messages: list[dict] | None,
    session_id: str | None,
    memory_mode: str,
    enabled: bool | None,
) -> bool:
    if enabled is not True or not session_id:
        return False
    if memory_mode == "off":
        return False
    if memory_mode == "on":
        return True
    return detect_memory_intent(query, recent_messages)


def _memory_source_ids(
    session_id: str | None,
    related_source_ids: list[str] | None,
) -> list[str]:
    ordered = []
    for value in [session_id, *(related_source_ids or [])]:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _should_retrieve_graph(
    query: str,
    graph_mode: str,
    enabled: bool | None,
    graph_store: Any,
) -> bool:
    if enabled is not True or graph_store is None:
        return False
    if graph_mode == "off":
        return False
    if graph_mode == "on":
        return True
    return detect_graph_intent(query)


def _combine_context(text_context: str, graph_context: str) -> str:
    parts = []
    if text_context:
        parts.append("Text Chunk Context:\n" + text_context)
    if graph_context:
        parts.append("Knowledge Graph Context:\n" + graph_context)
    return "\n\n".join(parts)
