"""Single-turn RAG answer generation helper."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Callable

from rag_pipeline.context_builder import build_rag_context
from rag_pipeline.graph_retrieval import detect_graph_intent
from rag_pipeline.graph_retrieval import retrieve_graph_context
from rag_pipeline.intent_classifier import RetrievalIntent, classify_intent
from rag_pipeline.llm_client import OpenAILlmClient
from rag_pipeline.memory_intent import detect_memory_intent
from rag_pipeline.retrieval_plan import PlanExecutionOutcome
from rag_pipeline.retrieval_plan import RetrievalPlan
from rag_pipeline.retrieval_plan import build_retrieval_plan
from rag_pipeline.retrieval_plan import execute_retrieval_plan
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.source_types import MATERIAL_SOURCE_TYPES
from rag_pipeline.web_search import WebSearchOutcome, search_web


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
WEB_SYSTEM_PROMPT_ADDITION = (
    " External web sources may provide current information outside the user's "
    "uploaded LearnCycle material. Clearly distinguish web information from "
    "the user's internal materials when relevant."
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
    context_summary: str | None = None,
    web_mode: str = "off",
    web_search_enabled: bool = False,
    web_search_query: str | None = None,
    web_search_fn: Callable[..., Any] | None = None,
    web_search_top_k: int | None = None,
    web_search_provider: str = "tavily",
    web_search_api_key: str | None = None,
    web_search_timeout_seconds: int = 15,
    web_search_max_query_chars: int = 300,
    web_search_max_context_sources: int = 5,
    web_search_max_chars_per_source: int = 1000,
    web_search_max_total_context_chars: int = 4000,
    intent_classifier_enabled: bool = False,
    intent: RetrievalIntent | None = None,
    intent_classifier_fn: Callable[..., RetrievalIntent] | None = None,
    intent_classifier_config: Any = None,
    retrieval_planner_enabled: bool = False,
    retrieval_plan: RetrievalPlan | None = None,
    retrieval_planner_fn: Callable[..., RetrievalPlan] | None = None,
    retrieval_plan_executor_fn: Callable[..., PlanExecutionOutcome] | None = None,
    retrieval_planner_config: Any = None,
) -> dict[str, Any]:
    """Retrieve user-scoped chunks, generate an answer, and return citations."""
    active_retrieval = retrieval_fn or search_hybrid_chunks
    active_intent = intent
    classifier_used = False
    fallback_used = False
    if active_intent is None and intent_classifier_enabled:
        classifier_used = True
        try:
            active_classifier = intent_classifier_fn or classify_intent
            active_intent = active_classifier(
                query=query,
                recent_messages=recent_messages,
                config=intent_classifier_config,
            )
        except Exception:
            logger.warning("Intent classifier integration failed; continuing without intent.", exc_info=True)
            fallback_used = True
            active_intent = None
    elif active_intent is not None:
        classifier_used = True

    effective_web_mode = web_mode
    effective_memory_mode = memory_mode
    graph_requested = False
    if active_intent is not None:
        if active_intent.needs_web and web_search_enabled:
            effective_web_mode = "on"
        if active_intent.needs_chat_memory and session_id and chat_memory_retrieval_enabled:
            effective_memory_mode = "on"
        graph_requested = bool(active_intent.needs_graph)

    retrieval_query = rewrite_query_for_retrieval(query, recent_messages, llm_client)
    retrieval_top_k = reranking_candidate_k if reranking_enabled and reranker is not None else top_k
    material_source_types = source_types or list(MATERIAL_SOURCE_TYPES)
    web_outcome = _empty_web_outcome(web_search_provider, web_search_enabled, effective_web_mode)
    planner_metadata: dict[str, Any] | None = None
    planner_used = False
    results: list[dict[str, Any]]
    if retrieval_plan is not None or (retrieval_planner_enabled and active_intent is not None):
        try:
            planner_used = True
            planner_config = retrieval_planner_config or _planner_config_from_kwargs(
                web_search_enabled=web_search_enabled,
                web_search_provider=web_search_provider,
                web_search_top_k=web_search_top_k or 5,
                web_search_timeout_seconds=web_search_timeout_seconds,
                web_search_max_query_chars=web_search_max_query_chars,
                tavily_api_key=web_search_api_key,
                chat_memory_top_k=chat_memory_top_k or 2,
            )
            plan = retrieval_plan or (retrieval_planner_fn or build_retrieval_plan)(
                query=retrieval_query,
                intent=active_intent,
                config=planner_config,
                session_id=session_id,
                selected_pdf_ids=pdf_ids,
                allowed_source_types=material_source_types,
            )
            outcome = (retrieval_plan_executor_fn or execute_retrieval_plan)(
                plan=plan,
                query=retrieval_query,
                user_id=user_id,
                config=planner_config,
                retrieval_fns={
                    "search_hybrid_chunks": active_retrieval,
                    "search_web": web_search_fn or search_web,
                },
                session_id=session_id,
            )
            results = outcome.results
            planner_metadata = _planner_metadata(True, outcome.fallback_used, outcome.step_outcomes)
            planner_web_results = [result for result in results if result.get("source_type") == "web"]
            if planner_web_results:
                effective_web_mode = "on"
                web_outcome = WebSearchOutcome(planner_web_results, web_search_provider, len(planner_web_results), None)
        except Exception:
            logger.warning("Retrieval planner failed; falling back to existing retrieval.", exc_info=True)
            results = active_retrieval(
                query=retrieval_query,
                user_id=user_id,
                source_types=material_source_types,
                top_k=retrieval_top_k,
                pdf_ids=pdf_ids,
            )
            planner_metadata = _planner_metadata(False, True, [], error_type="planner_error")
    else:
        results = active_retrieval(
            query=retrieval_query,
            user_id=user_id,
            source_types=material_source_types,
            top_k=retrieval_top_k,
            pdf_ids=pdf_ids,
        )
    graph_context = {"context_text": "", "sources": []}
    if not planner_used and _should_retrieve_graph(query, graph_mode, graph_retrieval_enabled, graph_store):
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
    if not planner_used and _should_retrieve_memory(
        query,
        recent_messages,
        session_id,
        effective_memory_mode,
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
    if not planner_used and web_search_enabled and effective_web_mode == "on":
        try:
            active_web_search = web_search_fn or search_web
            outcome = active_web_search(
                query=web_search_query or retrieval_query,
                top_k=web_search_top_k or 5,
                provider=web_search_provider,
                api_key=web_search_api_key,
                timeout_seconds=web_search_timeout_seconds,
                max_query_chars=web_search_max_query_chars,
            )
            web_outcome = _coerce_web_outcome(outcome, web_search_provider)
        except Exception:
            logger.warning("Web search failed; continuing without web context.", exc_info=True)
            web_outcome = WebSearchOutcome([], web_search_provider, 0, "provider_error")
    results = results + memory_results
    if web_outcome.results:
        results = results + web_outcome.results
    if not results and not graph_context.get("context_text"):
        return {"answer": FALLBACK_ANSWER, "sources": [], "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode), "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled), "retrieval_plan": planner_metadata}

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

    context = build_rag_context(
        context_results,
        max_chunks=context_top_k,
        web_max_sources=web_search_max_context_sources,
        web_max_chars_per_source=web_search_max_chars_per_source,
        web_max_total_chars=web_search_max_total_context_chars,
    )
    text_context = context["context_text"]
    graph_text = str(graph_context.get("context_text") or "").strip()
    combined_context = _combine_context(text_context, graph_text)
    if not combined_context:
        return {"answer": FALLBACK_ANSWER, "sources": [], "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode), "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled), "retrieval_plan": planner_metadata}

    active_llm = llm_client or OpenAILlmClient()
    web_text = any(result.get("source_type") == "web" for result in context_results)
    prompt_addition = ""
    if graph_text:
        prompt_addition += GRAPH_SYSTEM_PROMPT_ADDITION
    if web_text:
        prompt_addition += WEB_SYSTEM_PROMPT_ADDITION
    if recent_messages:
        summary_text = (context_summary or "").strip()
        conversation_prefix = (
            "[Conversation summary - earlier in this session]\n"
            f"{summary_text}\n\n"
            if summary_text
            else ""
        )
        user_prompt = (
            f"{conversation_prefix}Recent conversation:\n{_format_conversation_block(recent_messages)}\n\n"
            f"Retrieved context:\n{combined_context}\n\n"
            f"Current question:\n{query}"
        )
        system_prompt = CONVERSATION_SYSTEM_PROMPT + prompt_addition
    else:
        user_prompt = (
            "Answer the user's question using only this context.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{query}"
        )
        system_prompt = SYSTEM_PROMPT + prompt_addition
    answer = active_llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    return {
        "answer": answer,
        "sources": context["sources"] + list(graph_context.get("sources") or []),
        "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode),
        "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled),
        "retrieval_plan": planner_metadata,
    }


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


def _empty_web_outcome(provider: str, enabled: bool, mode: str) -> WebSearchOutcome:
    error_type = None
    if not enabled:
        error_type = "disabled"
    elif mode != "on":
        error_type = "not_requested"
    return WebSearchOutcome([], provider, 0, error_type)


def _coerce_web_outcome(outcome: Any, provider: str) -> WebSearchOutcome:
    if isinstance(outcome, WebSearchOutcome):
        return outcome
    if isinstance(outcome, list):
        return WebSearchOutcome(outcome, provider, len(outcome), None if outcome else "empty_results")
    return WebSearchOutcome([], provider, 0, "provider_error")


def _web_metadata(outcome: WebSearchOutcome, enabled: bool, mode: str) -> dict[str, Any]:
    requested = enabled and mode == "on"
    return {
        "enabled": enabled,
        "requested": requested,
        "used": bool(outcome.results),
        "provider": outcome.provider,
        "result_count": outcome.result_count,
        "error_type": outcome.error_type,
    }


def _intent_metadata(
    intent: RetrievalIntent | None,
    classifier_used: bool,
    fallback_used: bool,
    graph_requested: bool,
    web_enabled: bool,
    session_id: str | None,
    memory_enabled: bool | None,
) -> dict[str, Any] | None:
    if intent is None and not classifier_used:
        return None
    metadata: dict[str, Any] = {
        "classifier_used": classifier_used,
        "fallback_used": fallback_used,
    }
    if intent is not None:
        metadata.update(
            {
                "question_type": intent.question_type.value,
                "needs_pdf": intent.needs_pdf,
                "needs_notes": intent.needs_notes,
                "needs_annotations": intent.needs_annotations,
                "needs_chat_memory": intent.needs_chat_memory,
                "needs_graph": intent.needs_graph,
                "needs_web": intent.needs_web,
                "confidence": intent.confidence,
                "reasoning_summary": intent.reasoning_summary,
                "graph_requested": graph_requested,
                "graph_available": False if graph_requested else None,
                "web_skipped_reason": "disabled" if intent.needs_web and not web_enabled else None,
                "memory_skipped_reason": (
                    "missing_session_id"
                    if intent.needs_chat_memory and not session_id
                    else "disabled"
                    if intent.needs_chat_memory and memory_enabled is not True
                    else None
                ),
            }
        )
    return {key: value for key, value in metadata.items() if value is not None}


def _planner_config_from_kwargs(**kwargs: Any) -> Any:
    return SimpleNamespace(
        retrieval_planner_enabled=True,
        retrieval_planner_default_top_k=6,
        retrieval_planner_pdf_top_k=6,
        retrieval_planner_notes_top_k=4,
        retrieval_planner_annotations_top_k=4,
        retrieval_planner_memory_top_k=kwargs.get("chat_memory_top_k", 2),
        retrieval_planner_web_top_k=kwargs.get("web_search_top_k", 5),
        retrieval_planner_max_steps=5,
        retrieval_planner_include_disabled_steps=True,
        web_search_enabled=kwargs.get("web_search_enabled", False),
        web_search_provider=kwargs.get("web_search_provider", "tavily"),
        web_search_timeout_seconds=kwargs.get("web_search_timeout_seconds", 15),
        web_search_max_query_chars=kwargs.get("web_search_max_query_chars", 300),
        tavily_api_key=kwargs.get("tavily_api_key"),
    )


def _planner_metadata(
    planner_used: bool,
    fallback_used: bool,
    steps: list[dict[str, Any]],
    error_type: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "planner_used": planner_used,
        "fallback_used": fallback_used,
        "steps": steps,
    }
    if error_type:
        metadata["error_type"] = error_type
    if any(step.get("tool") == "query_knowledge_graph" for step in steps):
        metadata["graph_available"] = False
    return metadata
