"""LLM-first query understanding for LearnCycle retrieval routing.

This module does not execute retrieval tools. It converts the user's current
query into a structured routing decision for the retrieval planner.
"""

from __future__ import annotations

from enum import Enum
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag_pipeline.intent_classifier import (
    QuestionType,
    RetrievalIntent,
    classify_intent_heuristic,
)
from rag_pipeline.llm_client import OpenAILlmClient


logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

LEARNING_QUESTION_TYPE_VALUES = {
    "document_grounded",
    "note_grounded",
    "conversation_memory",
    "concept_explanation",
    "concept_relationship",
    "comparison",
    "learning_path",
    "summary",
    "quiz_or_flashcards",
}

INTERNAL_SOURCE_TYPES = {"pdf", "notes", "annotations", "chat_memory", "graph"}


class QueryRoute(str, Enum):
    INTERNAL_RETRIEVAL = "internal_retrieval"
    WEB_SEARCH = "web_search"
    CONVERSATION_ONLY = "conversation_only"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CLARIFICATION = "clarification"


class QueryUnderstanding(BaseModel):
    """Structured LLM result for query semantics before retrieval."""

    model_config = ConfigDict(extra="forbid")

    resolved_query: str = Field(min_length=1, max_length=500)
    question_type: QuestionType
    route: QueryRoute
    needs_pdf: bool
    needs_notes: bool
    needs_annotations: bool
    needs_chat_memory: bool
    needs_graph: bool
    needs_web: bool
    should_show_sources: bool
    confidence: float = Field(ge=0, le=1)
    reasoning_summary: str = Field(max_length=300)

    @field_validator("resolved_query", "reasoning_summary")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return " ".join(value.strip().split())

    def to_intent(self) -> RetrievalIntent:
        return RetrievalIntent(
            question_type=self.question_type,
            needs_pdf=self.needs_pdf,
            needs_notes=self.needs_notes,
            needs_annotations=self.needs_annotations,
            needs_chat_memory=self.needs_chat_memory,
            needs_graph=self.needs_graph,
            needs_web=self.needs_web,
            confidence=self.confidence,
            reasoning_summary=self.reasoning_summary,
        )


QUERY_UNDERSTANDING_SYSTEM_PROMPT = (
    "Understand the user's current query for LearnCycle retrieval routing. "
    "Return JSON only. Resolve follow-ups, corrections, abbreviations, and "
    "implicit references using recent conversation. If the user just says 'search now', 'try again', or 'I activated it', the resolved_query MUST be the full actual question from the previous turn, NOT the conversational filler. Decide one route: "
    "internal_retrieval, web_search, conversation_only, general_knowledge, "
    "or clarification. LearnCycle is an internal learning system: prefer "
    "internal_retrieval for learning-related questions, concept explanations, "
    "summaries, comparisons, quiz/flashcard requests, learning paths, and "
    "document-grounded questions whenever uploaded PDFs, notes, annotations, "
    "chat memory, a knowledge graph, or an active document/session scope may "
    "be relevant. Use web_search only when the user asks for current, external, "
    "salary, market, tax, legal, news, latest, live, or facts unlikely to be in "
    "uploaded learning materials. Use conversation_only for calculations, "
    "conversions, corrections, rewrites, formatting changes, or follow-ups "
    "answerable only from recent messages. Use general_knowledge only when no "
    "internal learning context is relevant or available, or when the user clearly "
    "asks for broad knowledge independent of their materials. For "
    "general_knowledge, set should_show_sources=true so the UI can show a "
    "transparent synthetic source such as 'Allgemeines Modellwissen'. Do not "
    "pretend general knowledge came from uploaded documents. Do not execute "
    "tools. Do not include chain of thought."
)


def build_query_understanding_prompt(
    query: str,
    recent_messages: list[dict[str, Any]] | None = None,
    *,
    max_recent_messages: int = 6,
    max_message_chars: int = 1200,
    has_internal_sources: bool | None = None,
    has_active_scope: bool | None = None,
    available_source_types: list[str] | None = None,
) -> str:
    """Build the LLM prompt for query understanding.

    has_internal_sources / has_active_scope should ideally be passed by the
    API layer per request. If unknown, the LLM is instructed to assume internal
    sources may be relevant for learning questions.
    """

    source_types = _clean_source_types(available_source_types)

    lines = [
        "Return this exact JSON shape:",
        "{",
        '  "resolved_query": "...",',
        '  "question_type": "document_grounded|note_grounded|conversation_memory|concept_explanation|concept_relationship|comparison|learning_path|summary|quiz_or_flashcards|current_external_info|general_chat",',
        '  "route": "internal_retrieval|web_search|conversation_only|general_knowledge|clarification",',
        '  "needs_pdf": true,',
        '  "needs_notes": true,',
        '  "needs_annotations": true,',
        '  "needs_chat_memory": false,',
        '  "needs_graph": false,',
        '  "needs_web": false,',
        '  "should_show_sources": true,',
        '  "confidence": 0.0,',
        '  "reasoning_summary": "short summary"',
        "}",
        "",
        "Current user query:",
        str(query or "").strip()[:max_message_chars],
        "",
        "LearnCycle context:",
        f"- Internal sources available: {_format_bool_unknown(has_internal_sources)}",
        f"- Active document/session scope: {_format_bool_unknown(has_active_scope)}",
        f"- Available source types: {', '.join(source_types) if source_types else 'unknown'}",
        "- If internal source availability is unknown, assume internal sources may be relevant for learning questions.",
        "",
        "Routing guidance:",
        "- internal_retrieval: DEFAULT for LearnCycle learning questions when uploaded PDFs, notes, annotations, chat memory, graph context, or an active document/session scope may be relevant.",
        "- Use internal_retrieval for concept explanations, summaries, comparisons, quiz/flashcard generation, learning paths, and document-grounded questions unless the user clearly asks for general knowledge only.",
        "- web_search: current/external salary, market, tax, legal, news, latest, live, or facts unlikely to be in uploaded learning materials.",
        "- conversation_only: answer can be derived from recent conversation only, such as rewriting, shortening, correcting, translating, formatting, calculating, or continuing the immediately previous answer.",
        "- general_knowledge: only when no internal learning context is relevant or available, or the user explicitly asks for a general answer independent of their materials.",
        "- clarification: the user needs to clarify before answering.",
        "",
        "Source guidance:",
        "- should_show_sources=true for internal_retrieval, web_search, and general_knowledge.",
        "- For general_knowledge, sources should later show a synthetic source card such as 'Allgemeines Modellwissen'; never present it as a PDF/note/annotation citation.",
        "- should_show_sources=false only for pure conversation_only rewrites, corrections, calculations, greetings, or clarification requests.",
        "- Web may later be blocked by server policy; still set needs_web according to meaning.",
        "- Never request chat memory for arbitrary browser-provided sessions; use it only for persistent LearnCycle memory-related questions.",
    ]

    recent = (recent_messages or [])[-max_recent_messages:]
    if recent:
        lines.extend(["", "Recent conversation:"])
        for message in recent:
            role = message.get("role")
            content = " ".join(str(message.get("content") or "").split())[:max_message_chars]
            if role in {"user", "assistant"} and content:
                lines.append(f"{role}: {content}")

    return "\n".join(lines)


def understand_query(
    query: str,
    *,
    recent_messages: list[dict[str, Any]] | None = None,
    llm_client: Any = None,
    config: Any = None,
    has_internal_sources: bool | None = None,
    has_active_scope: bool | None = None,
    available_source_types: list[str] | None = None,
) -> QueryUnderstanding:
    """Run LLM-first query understanding with deterministic fallback.

    The returned object is normalized after LLM/heuristic classification so that
    LearnCycle prefers internal retrieval for learning questions and exposes
    transparent source behavior for general-knowledge answers.
    """

    try:
        active_llm = llm_client
        if active_llm is None:
            api_key = getattr(config, "openai_api_key", None)
            if not api_key:
                raise RuntimeError("missing_api_key")
            active_llm = OpenAILlmClient(
                api_key=api_key,
                model=getattr(config, "intent_classifier_model", "gpt-4.1-mini"),
            )

        raw = active_llm.complete(
            system_prompt=QUERY_UNDERSTANDING_SYSTEM_PROMPT,
            user_prompt=build_query_understanding_prompt(
                query,
                recent_messages=recent_messages,
                max_recent_messages=getattr(config, "intent_classifier_max_recent_messages", 6),
                max_message_chars=getattr(config, "intent_classifier_max_message_chars", 1200),
                has_internal_sources=has_internal_sources,
                has_active_scope=has_active_scope,
                available_source_types=available_source_types,
            ),
        )
        understanding = QueryUnderstanding.model_validate(_parse_llm_json(str(raw)))
        return normalize_query_understanding(
            understanding,
            query=query,
            has_internal_sources=has_internal_sources,
            has_active_scope=has_active_scope,
            available_source_types=available_source_types,
        )
    except Exception as exc:
        logger.warning(
            "Query understanding failed; using heuristic fallback.",
            extra={
                "error_type": "query_understanding_failure",
                "exception_type": type(exc).__name__,
            },
            exc_info=True,
        )
        fallback = fallback_query_understanding(query, recent_messages)
        return normalize_query_understanding(
            fallback,
            query=query,
            has_internal_sources=has_internal_sources,
            has_active_scope=has_active_scope,
            available_source_types=available_source_types,
        )


def fallback_query_understanding(
    query: str,
    recent_messages: list[dict[str, Any]] | None = None,
) -> QueryUnderstanding:
    """Fallback to deterministic heuristic classification.

    This is intentionally conservative: web intent remains web, pure chat
    remains conversation-only, and all other learning-style questions default to
    internal retrieval so the vector database is not silently bypassed.
    """

    intent = classify_intent_heuristic(query, recent_messages)
    route = QueryRoute.INTERNAL_RETRIEVAL
    should_show_sources = True

    if intent.needs_web:
        route = QueryRoute.WEB_SEARCH
        should_show_sources = True
    elif intent.needs_chat_memory:
        route = QueryRoute.INTERNAL_RETRIEVAL
        should_show_sources = True
    elif intent.question_type == QuestionType.GENERAL_CHAT:
        if _looks_like_conversation_only(query):
            route = QueryRoute.CONVERSATION_ONLY
            should_show_sources = False
        else:
            route = QueryRoute.GENERAL_KNOWLEDGE
            should_show_sources = True

    return QueryUnderstanding(
        resolved_query=query.strip() or "general question",
        question_type=intent.question_type,
        route=route,
        needs_pdf=intent.needs_pdf,
        needs_notes=intent.needs_notes,
        needs_annotations=intent.needs_annotations,
        needs_chat_memory=intent.needs_chat_memory,
        needs_graph=intent.needs_graph,
        needs_web=intent.needs_web,
        should_show_sources=should_show_sources,
        confidence=intent.confidence,
        reasoning_summary=intent.reasoning_summary,
    )


def normalize_query_understanding(
    understanding: QueryUnderstanding,
    *,
    query: str,
    has_internal_sources: bool | None = None,
    has_active_scope: bool | None = None,
    available_source_types: list[str] | None = None,
) -> QueryUnderstanding:
    """Apply deterministic LearnCycle routing guardrails after classification."""

    data = understanding.model_dump()
    source_types = _clean_source_types(available_source_types)
    has_relevant_internal_context = bool(has_internal_sources or has_active_scope)
    is_learning_question = _is_learning_question_type(understanding.question_type)

    # Web route: web is the source. Do not mix it with internal source flags here;
    # later retrieval planning may still perform policy checks.
    if understanding.route == QueryRoute.WEB_SEARCH:
        data.update(
            needs_web=True,
            should_show_sources=True,
        )
        return QueryUnderstanding.model_validate(data)

    # Pure conversation route: no retrieval and no source cards.
    if understanding.route in {QueryRoute.CONVERSATION_ONLY, QueryRoute.CLARIFICATION}:
        data.update(
            needs_pdf=False,
            needs_notes=False,
            needs_annotations=False,
            needs_chat_memory=False,
            needs_graph=False,
            needs_web=False,
            should_show_sources=False,
        )
        return QueryUnderstanding.model_validate(data)

    # Internal-first guardrail: REMOVED as per user request. 
    # If the LLM classifies as GENERAL_KNOWLEDGE, let it be GENERAL_KNOWLEDGE.
    # if (
    #     understanding.route == QueryRoute.GENERAL_KNOWLEDGE
    #     and has_relevant_internal_context
    #     and is_learning_question
    # ):
    #     data.update(
    #         route=QueryRoute.INTERNAL_RETRIEVAL,
    #         should_show_sources=True,
    #     )
    #     _enable_internal_sources(data, source_types, has_active_scope=bool(has_active_scope))
    #     return QueryUnderstanding.model_validate(data)

    # General knowledge is allowed, but it should still be transparent in the UI
    # through a synthetic source card such as "Allgemeines Modellwissen".
    if understanding.route == QueryRoute.GENERAL_KNOWLEDGE:
        data.update(
            needs_pdf=False,
            needs_notes=False,
            needs_annotations=False,
            needs_chat_memory=False,
            needs_graph=False,
            needs_web=False,
            should_show_sources=True,
        )
        return QueryUnderstanding.model_validate(data)

    # Internal retrieval should always show sources. If no specific source flag
    # was selected, default to the available internal source types. This prevents
    # route=internal_retrieval with an empty retrieval plan.
    if understanding.route == QueryRoute.INTERNAL_RETRIEVAL:
        data.update(
            needs_web=False,
            should_show_sources=True,
        )
        if not _has_any_internal_source_flag(data):
            _enable_internal_sources(data, source_types, has_active_scope=bool(has_active_scope))
        return QueryUnderstanding.model_validate(data)

    return QueryUnderstanding.model_validate(data)


def synthetic_general_knowledge_source() -> dict[str, Any]:
    """Create a transparent pseudo-source for general-knowledge answers.

    Use this downstream when route == GENERAL_KNOWLEDGE and
    should_show_sources == True. Adapt field names to your SourceCard model if
    needed.
    """

    return {
        "chunk_id": "synthetic_general_knowledge",
        "source_type": "general_knowledge",
        "source_id": "synthetic_general_knowledge",
        "title": "Allgemeines Modellwissen",
        "heading": None,
        "page": None,
        "score": 1.0,
        "snippet": "Diese Antwort basiert nicht auf deinen hochgeladenen LearnCycle-Quellen.",
        "metadata": {
            "is_synthetic": True,
            "citation_policy": "not_a_document_source",
        },
    }


def _parse_llm_json(text: str) -> dict[str, Any]:
    stripped = _strip_json_fence(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = json.loads(_extract_json_object(stripped))

    if not isinstance(payload, dict):
        raise ValueError("query_understanding_response_must_be_json_object")
    return payload


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = _JSON_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return text[start : end + 1]


def _format_bool_unknown(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"


def _clean_source_types(source_types: list[str] | None) -> list[str]:
    if not source_types:
        return []
    cleaned: list[str] = []
    for source_type in source_types:
        value = str(source_type).strip().lower()
        if value in INTERNAL_SOURCE_TYPES and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _question_type_value(question_type: QuestionType) -> str:
    return str(getattr(question_type, "value", question_type))


def _is_learning_question_type(question_type: QuestionType) -> bool:
    return _question_type_value(question_type) in LEARNING_QUESTION_TYPE_VALUES


def _has_any_internal_source_flag(data: dict[str, Any]) -> bool:
    return bool(
        data.get("needs_pdf")
        or data.get("needs_notes")
        or data.get("needs_annotations")
        or data.get("needs_chat_memory")
        or data.get("needs_graph")
    )


def _enable_internal_sources(
    data: dict[str, Any],
    source_types: list[str],
    *,
    has_active_scope: bool,
) -> None:
    """Enable source flags for internal retrieval.

    If the API passes available_source_types, respect that exact set. Without it,
    default to PDFs for active document scope and to PDFs/notes/annotations for
    general LearnCycle internal retrieval. Chat memory and graph are only enabled
    when explicitly present in available_source_types.
    """

    data.update(
        needs_pdf=False,
        needs_notes=False,
        needs_annotations=False,
        needs_chat_memory=False,
        needs_graph=False,
    )

    if source_types:
        data["needs_pdf"] = "pdf" in source_types
        data["needs_notes"] = "notes" in source_types
        data["needs_annotations"] = "annotations" in source_types
        data["needs_chat_memory"] = "chat_memory" in source_types
        data["needs_graph"] = "graph" in source_types
        return

    if has_active_scope:
        data["needs_pdf"] = True
        return

    data["needs_pdf"] = True
    data["needs_notes"] = True
    data["needs_annotations"] = True


def _looks_like_conversation_only(query: str) -> bool:
    normalized = " ".join(str(query or "").strip().lower().split())
    if not normalized:
        return True

    exact_phrases = {
        "hi",
        "hallo",
        "hey",
        "ok",
        "okay",
        "danke",
        "thanks",
        "thank you",
        "ja",
        "nein",
        "mach kürzer",
        "mach es kürzer",
        "mach länger",
        "schreib kürzer",
        "schreib es kürzer",
        "korrigiere das",
        "übersetze das",
    }
    if normalized in exact_phrases:
        return True

    conversation_prefixes = (
        "formuliere das",
        "schreib das",
        "schreibe das",
        "kürze das",
        "kürz das",
        "mach das kürzer",
        "mach es kürzer",
        "mach daraus",
        "korrigiere den text",
        "verbessere den text",
        "übersetze den text",
        "translate this",
        "make it shorter",
        "make this shorter",
        "rewrite this",
        "fix this",
    )
    return normalized.startswith(conversation_prefixes)
