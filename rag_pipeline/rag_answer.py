"""Single-turn RAG answer generation helper."""

from __future__ import annotations

import logging
import json
import re
from types import SimpleNamespace
from typing import Any, Callable

from rag_pipeline.context_builder import build_rag_context
from rag_pipeline.agentic_retriever import AgenticRetrievalOutcome
from rag_pipeline.agentic_retriever import run_controlled_agentic_retrieval
from rag_pipeline.graph_retrieval import detect_graph_intent
from rag_pipeline.graph_retrieval import retrieve_graph_context
from rag_pipeline.intent_classifier import RetrievalIntent, classify_intent
from rag_pipeline.llm_client import OpenAILlmClient
from rag_pipeline.memory_intent import detect_memory_intent
from rag_pipeline.pedagogical_prompts import FEYNMAN_SYSTEM_PROMPT
from rag_pipeline.pedagogical_prompts import GUIDED_LEARNING_SYSTEM_PROMPT
from rag_pipeline.pedagogical_prompts import extract_al_state_update
from rag_pipeline.query_understanding import QueryRoute, QueryUnderstanding, understand_query
from rag_pipeline.retrieval_plan import PlanExecutionOutcome
from rag_pipeline.retrieval_plan import RetrievalPlan
from rag_pipeline.retrieval_plan import build_retrieval_plan
from rag_pipeline.retrieval_plan import execute_retrieval_plan
from rag_pipeline.retrieval import search_hybrid_chunks
from rag_pipeline.source_types import MATERIAL_SOURCE_TYPES
from rag_pipeline.web_search import WebSearchOutcome, search_web

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_pipeline.retrieval_tools import RetrievalToolRegistry


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are LearnCycle's learning assistant. Use only the provided context from the "
    "user's uploaded PDFs, notes, and annotations. "
    "Pay special attention to the domain and topic of the context. If the user asks for a definition or an acronym (like 'PM'), you MUST deduce the meaning from the context (e.g., if the context is about 'Process Mining', then 'PM' means 'Process Mining', not 'Projektmanagement' or something else). "
    "Do NOT rely on your pre-trained general knowledge for acronyms if the context implies a different domain. "
    "If the context does not contain enough information to answer the question, but you can answer it from your general knowledge, "
    "you MUST start your response with the EXACT phrase '[GENERAL_KNOWLEDGE]'. "
    'Do not begin your answer with "Im bereitgestellten Kontext". Prefer German when the user asks '
    "in German. Give a helpful learning-oriented explanation."
)
REWRITE_SYSTEM_PROMPT = (
    "Rewrite the current user question into a standalone retrieval query using the recent "
    "conversation only to resolve references. If the user just says 'search now', 'try again', or 'activated', output the full actual question from the previous turn. "
    "Ensure the core subject of the conversation is explicitly included in the rewritten query (e.g. 'Use Cases für PM Consultant' instead of just 'Use Cases'). "
    "Output only the rewritten query, no explanation. Keep it concise, at most two short sentences or about 200 characters. "
    "Preserve the user's language. Do not answer the question."
)
CONVERSATION_SYSTEM_PROMPT = (
    "You are LearnCycle's learning assistant. Use the retrieved context as the factual "
    "source of truth. The recent conversation is provided only for continuity and "
    "resolving references. "
    "Pay special attention to the domain and topic of the context. If the user asks for a definition or an acronym (like 'PM'), you MUST deduce the meaning from the context (e.g., if the context is about 'Process Mining', then 'PM' means 'Process Mining', not 'Projektmanagement' or something else). "
    "Do NOT rely on your pre-trained general knowledge for acronyms if the context implies a different domain. "
    "If the retrieved context does not contain enough information to answer the question, but you can answer it from your general knowledge, "
    "you MUST start your response with the EXACT phrase '[GENERAL_KNOWLEDGE]'. "
    'Do not invent facts or sources. Do not begin your answer with "Im bereitgestellten Kontext". Do not mention raw source metadata. '
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


def _last_user_message(recent_messages: list[dict] | None) -> str | None:
    for message in reversed(recent_messages or []):
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content:
                return content
    return None


def _normalize_clarification_phrase(query: str) -> str:
    normalized = " ".join(query.strip().split())
    lowered = normalized.lower()
    prefixes = [
        "ich meinte ",
        "ich meine ",
        "damit meine ich ",
        "damit meinte ich ",
        "pm bedeutet ",
        "pm heisst ",
        "pm heißt ",
        "pm = ",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return normalized[len(prefix):].strip(" .:;-")
    return normalized


def _rewrite_abbreviation_clarification(query: str, recent_messages: list[dict] | None) -> str | None:
    phrase = _normalize_clarification_phrase(query)
    words = [word for word in re.split(r"\s+", phrase.strip()) if word]
    if not 1 <= len(words) <= 4:
        return None
    initials = "".join(word[0] for word in words if word[:1].isalpha()).lower()
    if len(initials) < 2:
        return None
    previous = _last_user_message(recent_messages)
    if not previous:
        return None
    pattern = re.compile(rf"\b{re.escape(initials)}\b", re.IGNORECASE)
    if not pattern.search(previous):
        return None
    rewritten = pattern.sub(phrase, previous)
    return rewritten if rewritten != previous else None


def _is_conversation_only_followup(query: str, recent_messages: list[dict] | None) -> bool:
    if not recent_messages:
        return False
    normalized = query.lower().strip()
    if len(normalized) > 120:
        return False
    phrase_cues = [
        "wie viel ist das netto",
        "was ist das netto",
        "netto",
        "brutto",
        "monatlich",
        "jährlich",
        "jaehrlich",
        "davon",
        "daraus",
        "umgerechnet",
        "pro monat",
        "pro jahr",
    ]
    word_cues = ["davon", "daraus"]
    return any(cue in normalized for cue in phrase_cues) or any(
        re.search(rf"\b{re.escape(cue)}\b", normalized)
        for cue in word_cues
    )


def rewrite_query_for_retrieval(
    query: str,
    recent_messages: list[dict] | None = None,
    llm_client: Any = None,
) -> str:
    """Rewrite vague follow-up questions into standalone retrieval queries."""
    abbreviation_rewrite = _rewrite_abbreviation_clarification(query, recent_messages)
    if abbreviation_rewrite:
        return abbreviation_rewrite[:200]
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


def _is_active_learning_mode(chat_mode: str) -> bool:
    return chat_mode in {"guided_learning", "feynman"}


def _select_system_prompt(
    chat_mode: str,
    *,
    has_recent_messages: bool,
    has_graph: bool,
    has_web: bool,
    no_info_instruction: str,
) -> str:
    if chat_mode == "guided_learning":
        return GUIDED_LEARNING_SYSTEM_PROMPT + no_info_instruction
    if chat_mode == "feynman":
        return FEYNMAN_SYSTEM_PROMPT + no_info_instruction

    prompt = CONVERSATION_SYSTEM_PROMPT if has_recent_messages else SYSTEM_PROMPT
    if has_graph:
        prompt += GRAPH_SYSTEM_PROMPT_ADDITION
    if has_web:
        prompt += WEB_SYSTEM_PROMPT_ADDITION
    return prompt + no_info_instruction


def _append_active_learning_state(user_prompt: str, active_learning_state: dict[str, Any] | None) -> str:
    state = active_learning_state or {}
    if not state:
        return user_prompt
    try:
        state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))[:400]
    except TypeError:
        state_json = "{}"
    return f"{user_prompt}\n\n[Current learning state]: {state_json}"


def _language_instruction(chat_language: str | None) -> str:
    if chat_language == "de":
        return " Always respond in German."
    if chat_language == "en":
        return " Always respond in English."
    return ""


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
    tool_registry: "RetrievalToolRegistry | None" = None,
    agentic_retriever_enabled: bool = False,
    agentic_retriever_fn: Callable[..., AgenticRetrievalOutcome] | None = None,
    chat_mode: str = "normal",
    active_learning_state: dict[str, Any] | None = None,
    chat_language: str | None = None,
) -> dict[str, Any]:
    """Retrieve user-scoped chunks, generate an answer, and return citations."""
    active_retrieval = retrieval_fn or search_hybrid_chunks
    query_understanding: QueryUnderstanding | None = None
    conversation_only_followup = False
    active_intent = intent
    classifier_used = False
    fallback_used = False
    if active_intent is None and intent_classifier_enabled and intent_classifier_fn is None:
        classifier_used = True
        query_understanding = understand_query(
            query,
            recent_messages=recent_messages,
            llm_client=llm_client,
            config=intent_classifier_config,
        )
        active_intent = query_understanding.to_intent()
        conversation_only_followup = query_understanding.route == QueryRoute.CONVERSATION_ONLY
    elif active_intent is None and intent_classifier_enabled:
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
    if query_understanding is None:
        conversation_only_followup = _is_conversation_only_followup(query, recent_messages)

    effective_web_mode = web_mode
    effective_memory_mode = memory_mode
    graph_requested = False
    if active_intent is not None:
        if active_intent.needs_web and web_search_enabled and web_mode == "on":
            effective_web_mode = "on"
        if active_intent.needs_chat_memory and session_id and chat_memory_retrieval_enabled:
            effective_memory_mode = "on"
        graph_requested = bool(active_intent.needs_graph)

    if query_understanding is not None:
        retrieval_query = query_understanding.resolved_query
    else:
        retrieval_query = rewrite_query_for_retrieval(query, recent_messages, llm_client)
    retrieval_top_k = reranking_candidate_k if reranking_enabled and reranker is not None else top_k
    material_source_types = source_types or list(MATERIAL_SOURCE_TYPES)
    web_outcome = _empty_web_outcome(web_search_provider, web_search_enabled, effective_web_mode)
    planner_metadata: dict[str, Any] | None = None
    planner_used = False
    _registry_used = False
    _tool_outcomes: list[dict[str, Any]] = []
    agentic_metadata: dict[str, Any] | None = None
    results: list[dict[str, Any]]
    semantic_route = query_understanding.route if query_understanding is not None else None
    skip_internal_retrieval = semantic_route in {
        QueryRoute.WEB_SEARCH,
        QueryRoute.CONVERSATION_ONLY,
        QueryRoute.GENERAL_KNOWLEDGE,
        QueryRoute.CLARIFICATION,
    }
    if conversation_only_followup or skip_internal_retrieval:
        results = []
    elif retrieval_plan is not None or (retrieval_planner_enabled and active_intent is not None):
        try:
            planner_used = True
            planner_config = retrieval_planner_config or _planner_config_from_kwargs(
                web_search_enabled=web_search_enabled and effective_web_mode == "on",
                web_search_provider=web_search_provider,
                web_search_top_k=web_search_top_k or 5,
                web_search_timeout_seconds=web_search_timeout_seconds,
                web_search_max_query_chars=web_search_max_query_chars,
                tavily_api_key=web_search_api_key,
                chat_memory_top_k=chat_memory_top_k or 2,
            )
            planner_config = _planner_config_with_web_gate(
                planner_config,
                web_allowed=web_search_enabled and effective_web_mode == "on",
            )
            plan = retrieval_plan or (retrieval_planner_fn or build_retrieval_plan)(
                query=retrieval_query,
                intent=active_intent,
                config=planner_config,
                session_id=session_id,
                memory_source_ids=memory_source_ids,
                selected_pdf_ids=pdf_ids,
                allowed_source_types=material_source_types,
            )
            if agentic_retriever_enabled and active_intent is not None and tool_registry is not None:
                try:
                    agentic = (agentic_retriever_fn or run_controlled_agentic_retrieval)(
                        query=retrieval_query,
                        intent=active_intent,
                        plan=plan,
                        user_id=user_id,
                        config=planner_config,
                        tool_registry=tool_registry,
                        plan_executor_fn=retrieval_plan_executor_fn,
                        llm_client=llm_client,
                        session_id=session_id,
                        selected_pdf_ids=pdf_ids,
                        allowed_source_types=material_source_types,
                    )
                    results = agentic.results
                    _registry_used = True
                    _tool_outcomes = agentic.tool_outcomes
                    outcome = PlanExecutionOutcome(
                        results=agentic.results,
                        step_outcomes=agentic.tool_outcomes,
                        total_result_count=len(agentic.results),
                        fallback_used=agentic.fallback_used,
                        registry_used=True,
                        tool_outcomes=agentic.tool_outcomes,
                    )
                    agentic_metadata = _agentic_retriever_metadata(
                        enabled=True,
                        used=True,
                        quality_mode=getattr(planner_config, "agentic_retriever_quality_assessment_mode", "heuristic"),
                        refinement_mode=getattr(planner_config, "agentic_retriever_refinement_mode", "heuristic"),
                        outcome=agentic,
                    )
                except Exception:
                    logger.warning("Agentic retriever failed; falling back to planner execution.", exc_info=True)
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
                        tool_registry=tool_registry,
                    )
                    results = outcome.results
                    _registry_used = getattr(outcome, "registry_used", False)
                    _tool_outcomes = getattr(outcome, "tool_outcomes", [])
                    agentic_metadata = _agentic_retriever_metadata(
                        enabled=True,
                        used=False,
                        quality_mode=getattr(planner_config, "agentic_retriever_quality_assessment_mode", "heuristic"),
                        refinement_mode=getattr(planner_config, "agentic_retriever_refinement_mode", "heuristic"),
                        fallback_used=True,
                        error_type="agentic_error",
                    )
            else:
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
                    tool_registry=tool_registry,
                )
                results = outcome.results
                _registry_used = getattr(outcome, "registry_used", False)
                _tool_outcomes = getattr(outcome, "tool_outcomes", [])
                if agentic_retriever_enabled:
                    agentic_metadata = _agentic_retriever_metadata(
                        enabled=True,
                        used=False,
                        quality_mode=getattr(planner_config, "agentic_retriever_quality_assessment_mode", "heuristic"),
                        refinement_mode=getattr(planner_config, "agentic_retriever_refinement_mode", "heuristic"),
                        fallback_used=True,
                        error_type="registry_unavailable",
                    )
            planner_metadata = _planner_metadata(
                True, outcome.fallback_used, outcome.step_outcomes,
                registry_used=_registry_used,
                tool_outcomes=_tool_outcomes,
            )
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
    if not conversation_only_followup and not planner_used and _should_retrieve_graph(query, graph_mode, graph_retrieval_enabled, graph_store):
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
    if not conversation_only_followup and not planner_used and _should_retrieve_memory(
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
    if not conversation_only_followup and not planner_used and web_search_enabled and effective_web_mode == "on":
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
    web_allowed_bool = web_search_enabled and effective_web_mode == "on"
    no_info_instruction = (
        " If you cannot answer it at all, you MUST start your response with the EXACT phrase '[NO_INFO]', and then write a polite message in the user's language saying you cannot find the answer in the provided materials or web results. CRITICAL: DO NOT suggest activating web search, because it is already active! Do not invent facts or sources."
        if web_allowed_bool
        else " If you cannot answer it at all, you MUST start your response with the EXACT phrase '[NO_INFO]', and then write a polite message in the user's language saying you cannot find the answer in the provided materials and suggesting they activate Web Search. Do not invent facts or sources."
    )

    if _should_answer_without_retrieval(
        query_understanding,
        conversation_only_followup=conversation_only_followup,
        recent_messages=recent_messages,
        retrieval_query=retrieval_query,
        query=query,
        web_allowed=web_allowed_bool,
        has_results=bool(results or graph_context.get("context_text")),
    ):
        active_llm = llm_client or OpenAILlmClient()
        user_prompt = _no_retrieval_prompt(
            query=query,
            retrieval_query=retrieval_query,
            recent_messages=recent_messages,
            query_understanding=query_understanding,
            web_allowed=web_allowed_bool,
        )
        answer = active_llm.complete(
            system_prompt=CONVERSATION_SYSTEM_PROMPT + no_info_instruction + _language_instruction(chat_language),
            user_prompt=user_prompt,
        )
        
        synthetic_sources = []
        if answer.strip().startswith("[NO_INFO]"):
            answer = answer.replace("[NO_INFO]", "", 1).strip()
        else:
            if answer.strip().startswith("[GENERAL_KNOWLEDGE]"):
                answer = answer.replace("[GENERAL_KNOWLEDGE]", "", 1).strip()
            
            if query_understanding and getattr(query_understanding, "route", None) == QueryRoute.GENERAL_KNOWLEDGE and getattr(query_understanding, "should_show_sources", False):
                from rag_pipeline.query_understanding import synthetic_general_knowledge_source
                synthetic_sources.append(synthetic_general_knowledge_source())

        return {
            "answer": answer,
            "sources": synthetic_sources,
            "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode),
            "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled),
            "retrieval_plan": planner_metadata,
            "retrieval_tools": _retrieval_tools_metadata(_registry_used, _tool_outcomes),
            "agentic_retriever": agentic_metadata,
        }

    if not results and not graph_context.get("context_text"):
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode),
            "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled),
            "retrieval_plan": planner_metadata,
            "retrieval_tools": _retrieval_tools_metadata(_registry_used, _tool_outcomes),
            "agentic_retriever": agentic_metadata,
        }

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
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode),
            "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled),
            "retrieval_plan": planner_metadata,
            "retrieval_tools": _retrieval_tools_metadata(_registry_used, _tool_outcomes),
            "agentic_retriever": agentic_metadata,
        }

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
        resolved_question_block = (
            f"Resolved question for retrieval:\n{retrieval_query}\n\n"
            if retrieval_query != query
            else ""
        )
        user_prompt = (
            f"{conversation_prefix}Recent conversation:\n{_format_conversation_block(recent_messages)}\n\n"
            f"Retrieved context:\n{combined_context}\n\n"
            f"{resolved_question_block}"
            f"Current question:\n{query}"
        )
        system_prompt = _select_system_prompt(
            chat_mode,
            has_recent_messages=True,
            has_graph=bool(graph_text),
            has_web=web_text,
            no_info_instruction=no_info_instruction,
        ) + _language_instruction(chat_language)
    else:
        user_prompt = (
            "Answer the user's question using only this context.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{query}"
        )
        system_prompt = _select_system_prompt(
            chat_mode,
            has_recent_messages=False,
            has_graph=bool(graph_text),
            has_web=web_text,
            no_info_instruction=no_info_instruction,
        ) + _language_instruction(chat_language)
    if _is_active_learning_mode(chat_mode):
        user_prompt = _append_active_learning_state(user_prompt, active_learning_state)
    answer = active_llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    updated_active_learning_state = None
    if _is_active_learning_mode(chat_mode):
        answer, updated_active_learning_state = extract_al_state_update(
            answer,
            active_learning_state or {},
        )
    
    final_sources = context["sources"] + list(graph_context.get("sources") or [])
    if answer.strip().startswith("[NO_INFO]"):
        answer = answer.replace("[NO_INFO]", "", 1).strip()
        final_sources = []
    elif answer.strip().startswith("[GENERAL_KNOWLEDGE]"):
        answer = answer.replace("[GENERAL_KNOWLEDGE]", "", 1).strip()
        from rag_pipeline.query_understanding import synthetic_general_knowledge_source
        final_sources = [synthetic_general_knowledge_source()]

    response = {
        "answer": answer,
        "sources": final_sources,
        "web_search": _web_metadata(web_outcome, web_search_enabled, effective_web_mode),
        "intent": _intent_metadata(active_intent, classifier_used, fallback_used, graph_requested, web_search_enabled, session_id, chat_memory_retrieval_enabled),
        "retrieval_plan": planner_metadata,
        "retrieval_tools": _retrieval_tools_metadata(_registry_used, _tool_outcomes),
        "agentic_retriever": agentic_metadata,
    }
    if updated_active_learning_state is not None:
        response["updated_active_learning_state"] = updated_active_learning_state
    return response


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


def _should_answer_without_retrieval(
    understanding: QueryUnderstanding | None,
    *,
    conversation_only_followup: bool,
    recent_messages: list[dict] | None,
    retrieval_query: str,
    query: str,
    web_allowed: bool,
    has_results: bool,
) -> bool:
    if has_results:
        return False
    if understanding is not None:
        if understanding.route in {
            QueryRoute.CONVERSATION_ONLY,
            QueryRoute.GENERAL_KNOWLEDGE,
            QueryRoute.CLARIFICATION,
        }:
            return True
        if understanding.route == QueryRoute.WEB_SEARCH and not web_allowed:
            return True
    return bool(conversation_only_followup and recent_messages and retrieval_query == query)


def _no_retrieval_prompt(
    *,
    query: str,
    retrieval_query: str,
    recent_messages: list[dict] | None,
    query_understanding: QueryUnderstanding | None,
    web_allowed: bool,
) -> str:
    parts = [
        "Answer the current question without uploaded-material citations.",
        "Do not cite PDF, note, annotation, or chat-memory sources because no retrieval context is used.",
    ]
    if query_understanding is not None:
        parts.append(f"Route: {query_understanding.route.value}.")
        parts.append(f"Resolved query: {query_understanding.resolved_query}")
        if query_understanding.needs_web and not web_allowed:
            parts.append(
                "Web search is not enabled for this request. Give a cautious general answer or say that current sourced values require enabling web search."
            )
    elif retrieval_query != query:
        parts.append(f"Resolved query: {retrieval_query}")
    if recent_messages:
        parts.extend(["", f"Recent conversation:\n{_format_conversation_block(recent_messages)}"])
    parts.extend(["", f"Current question:\n{query}"])
    return "\n".join(parts)


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


def _planner_config_with_web_gate(config: Any, *, web_allowed: bool) -> Any:
    if web_allowed or not getattr(config, "web_search_enabled", False):
        return config
    if hasattr(config, "model_dump"):
        values = config.model_dump()
    else:
        values = vars(config).copy()
    values["web_search_enabled"] = False
    return SimpleNamespace(**values)


def _planner_metadata(
    planner_used: bool,
    fallback_used: bool,
    steps: list[dict[str, Any]],
    error_type: str | None = None,
    registry_used: bool = False,
    tool_outcomes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "planner_used": planner_used,
        "fallback_used": fallback_used,
        "steps": steps,
    }
    if error_type:
        metadata["error_type"] = error_type
    if registry_used:
        metadata["registry_used"] = True
    graph_steps = [step for step in steps if step.get("tool") == "query_knowledge_graph"]
    if graph_steps:
        # graph is available if at least one graph step is not disabled
        graph_available = any(s.get("status") not in ("disabled", "skipped") for s in graph_steps)
        metadata["graph_available"] = graph_available
    return metadata


def _retrieval_tools_metadata(
    registry_used: bool,
    tool_outcomes: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not registry_used:
        return None
    return {
        "registry_used": True,
        "tools": [
            {
                "tool": t.get("tool"),
                "status": t.get("status"),
                "result_count": t.get("result_count"),
                "latency_ms": t.get("latency_ms"),
                "error_type": t.get("error_type"),
            }
            for t in (tool_outcomes or [])
        ],
    }


def _agentic_retriever_metadata(
    *,
    enabled: bool,
    used: bool,
    quality_mode: str,
    refinement_mode: str,
    outcome: AgenticRetrievalOutcome | None = None,
    fallback_used: bool = False,
    error_type: str | None = None,
) -> dict[str, Any]:
    quality = outcome.quality if outcome else None
    return {
        "enabled": enabled,
        "used": used,
        "quality_mode": quality_mode,
        "refinement_mode": refinement_mode,
        "refinement_used": bool(outcome.refinement_used) if outcome else False,
        "refinement_rounds": outcome.refinement_rounds if outcome else 0,
        "tool_call_count": outcome.tool_call_count if outcome else 0,
        "quality": {
            "status": quality.status.value if quality else None,
            "missing_aspects": quality.missing_aspects if quality else [],
        },
        "fallback_used": bool(outcome.fallback_used) if outcome else fallback_used,
        "error_type": outcome.error_type if outcome else error_type,
    }
