"""LLM-backed retrieval intent classification with deterministic fallback."""

from __future__ import annotations

from enum import Enum
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag_pipeline.llm_client import OpenAILlmClient


logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    DOCUMENT_GROUNDED = "document_grounded"
    NOTE_GROUNDED = "note_grounded"
    CONVERSATION_MEMORY = "conversation_memory"
    CONCEPT_EXPLANATION = "concept_explanation"
    CONCEPT_RELATIONSHIP = "concept_relationship"
    COMPARISON = "comparison"
    LEARNING_PATH = "learning_path"
    SUMMARY = "summary"
    QUIZ_OR_FLASHCARDS = "quiz_or_flashcards"
    CURRENT_EXTERNAL_INFO = "current_external_info"
    GENERAL_CHAT = "general_chat"


class RetrievalIntent(BaseModel):
    """Validated routing flags for retrieval aids."""

    model_config = ConfigDict(extra="forbid")

    question_type: QuestionType
    needs_pdf: bool
    needs_notes: bool
    needs_annotations: bool
    needs_chat_memory: bool
    needs_graph: bool
    needs_web: bool
    confidence: float = Field(ge=0, le=1)
    reasoning_summary: str = Field(max_length=300)

    @field_validator("reasoning_summary")
    @classmethod
    def strip_reasoning_summary(cls, value: str) -> str:
        return value.strip()[:300]


INTENT_CLASSIFIER_SYSTEM_PROMPT = (
    "Classify the user's retrieval intent for LearnCycle. Return JSON only with "
    "question_type, needs_pdf, needs_notes, needs_annotations, needs_chat_memory, "
    "needs_graph, needs_web, confidence, reasoning_summary. Do not include chain "
    "of thought. Use web only for current, external, latest, or news-like "
    "questions. Use chat memory only for previous conversation or learning "
    "progress questions. Use graph for relationships, dependencies, prerequisites, "
    "comparisons, and learning paths. Graph is only a planning flag for now."
)


def default_intent() -> RetrievalIntent:
    """Return safe existing-RAG default routing."""
    return RetrievalIntent(
        question_type=QuestionType.DOCUMENT_GROUNDED,
        needs_pdf=True,
        needs_notes=True,
        needs_annotations=True,
        needs_chat_memory=False,
        needs_graph=False,
        needs_web=False,
        confidence=0.5,
        reasoning_summary="Default internal material retrieval.",
    )


def general_chat_intent() -> RetrievalIntent:
    """Return a safe intent for empty or conversational input."""
    return RetrievalIntent(
        question_type=QuestionType.GENERAL_CHAT,
        needs_pdf=True,
        needs_notes=True,
        needs_annotations=True,
        needs_chat_memory=False,
        needs_graph=False,
        needs_web=False,
        confidence=0.5,
        reasoning_summary="General chat fallback.",
    )


def build_intent_classifier_prompt(
    query: str,
    recent_messages: list[dict[str, Any]] | None = None,
    selected_source_context: dict[str, Any] | None = None,
    max_recent_messages: int = 4,
    max_message_chars: int = 1000,
) -> str:
    """Build a compact prompt containing only query and truncated recent messages."""
    lines = [
        "Return JSON only. No markdown.",
        "Current user query:",
        str(query or "").strip()[:max_message_chars],
        "",
        "Source rules:",
        "- PDF: uploaded course material questions.",
        "- Notes/annotations: user-created study content questions.",
        "- Chat memory: previous conversation, learning progress, or what was discussed.",
        "- Graph: relationships, dependencies, prerequisites, comparisons, learning paths.",
        "- Web: current/external/latest information not expected in uploaded materials.",
        "- Normal factual questions about uploaded materials should not use web.",
        "",
        "Examples:",
        'Was steht in meiner PDF über Process Mining? -> {"question_type":"document_grounded","needs_pdf":true}',
        'Was hatten wir gestern zu BPMN besprochen? -> {"question_type":"conversation_memory","needs_chat_memory":true}',
        'Was ist der Zusammenhang zwischen BPMN und Process Mining? -> {"question_type":"concept_relationship","needs_graph":true}',
        'Was ist aktuell neu bei OpenAI Agents SDK? -> {"question_type":"current_external_info","needs_web":true}',
    ]
    if selected_source_context:
        lines.extend(["", f"Source scope hint: {json.dumps(selected_source_context, sort_keys=True)}"])
    recent = (recent_messages or [])[-max_recent_messages:]
    if recent:
        lines.extend(["", "Recent conversation excerpt:"])
        for message in recent:
            role = message.get("role")
            content = " ".join(str(message.get("content") or "").split())[:max_message_chars]
            if role in {"user", "assistant"} and content:
                lines.append(f"{role}: {content}")
    return "\n".join(lines)


def classify_intent(
    query: str,
    llm_client: Any = None,
    recent_messages: list[dict[str, Any]] | None = None,
    config: Any = None,
) -> RetrievalIntent:
    """Classify retrieval intent via LLM with heuristic fallback on all failures."""
    if not query or not query.strip():
        return general_chat_intent()
    fallback_enabled = bool(getattr(config, "intent_classifier_fallback_enabled", True))
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
        prompt = build_intent_classifier_prompt(
            query,
            recent_messages=recent_messages,
            max_recent_messages=getattr(config, "intent_classifier_max_recent_messages", 4),
            max_message_chars=getattr(config, "intent_classifier_max_message_chars", 1000),
        )
        raw = active_llm.complete(
            system_prompt=INTENT_CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        payload = json.loads(_strip_json_fence(str(raw)))
        return RetrievalIntent.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, Exception):
        logger.warning("Intent classification failed; using fallback.", extra={"error_type": "classifier_failure"})
        return classify_intent_heuristic(query, recent_messages) if fallback_enabled else default_intent()


def classify_intent_heuristic(
    query: str,
    recent_messages: list[dict[str, Any]] | None = None,
) -> RetrievalIntent:
    """Deterministic keyword fallback for retrieval intent routing."""
    normalized = query.lower().strip()
    if not normalized:
        return general_chat_intent()
    if _has(normalized, ["was hatten wir", "was haben wir besprochen", "vorhin", "eben", "gestern besprochen", "what did we discuss", "as before", "like earlier"]):
        return _intent(QuestionType.CONVERSATION_MEMORY, needs_chat_memory=True, confidence=0.75, summary="Previous conversation or learning progress.")
    if _has(normalized, ["aktuell", "neu", "heute", "2026", "latest", "current", "news", "recent"]):
        return _intent(QuestionType.CURRENT_EXTERNAL_INFO, needs_web=True, confidence=0.75, summary="Current or external information requested.")
    if _has(normalized, ["karteikarten", "flashcards", "quiz", "fragen erstellen"]):
        return _intent(QuestionType.QUIZ_OR_FLASHCARDS, confidence=0.7, summary="Quiz or flashcard generation request.")
    if _has(normalized, ["fasse zusammen", "summary", "summarize"]):
        return _intent(QuestionType.SUMMARY, needs_notes="notiz" in normalized or "note" in normalized, confidence=0.7, summary="Summary request.")
    if _has(normalized, ["meine notizen", "notiz", "my notes"]):
        return _intent(QuestionType.NOTE_GROUNDED, needs_pdf=False, needs_notes=True, needs_annotations=False, confidence=0.75, summary="User notes requested.")
    if _has(normalized, ["markierung", "kommentar", "annotation", "highlight"]):
        return _intent(QuestionType.NOTE_GROUNDED, needs_pdf=False, needs_notes=False, needs_annotations=True, confidence=0.75, summary="Annotation content requested.")
    if _has(normalized, ["lernpfad", "learning path"]):
        return _intent(QuestionType.LEARNING_PATH, needs_graph=True, confidence=0.75, summary="Learning path relationship planning.")
    if _has(normalized, ["zusammenhang", "zusammenhängen", "hängt", "abhängig", "voraussetzung", "beziehung", "verbunden", "relationship", "depends", "prerequisite"]):
        return _intent(QuestionType.CONCEPT_RELATIONSHIP, needs_graph=True, confidence=0.75, summary="Concept relationship requested.")
    if _has(normalized, ["unterschied", "vergleich", "compare", "difference"]):
        return _intent(QuestionType.COMPARISON, needs_graph=True, confidence=0.75, summary="Comparison requested.")
    return default_intent()


def _intent(
    question_type: QuestionType,
    needs_pdf: bool = True,
    needs_notes: bool = True,
    needs_annotations: bool = True,
    needs_chat_memory: bool = False,
    needs_graph: bool = False,
    needs_web: bool = False,
    confidence: float = 0.65,
    summary: str = "Heuristic routing.",
) -> RetrievalIntent:
    return RetrievalIntent(
        question_type=question_type,
        needs_pdf=needs_pdf,
        needs_notes=needs_notes,
        needs_annotations=needs_annotations,
        needs_chat_memory=needs_chat_memory,
        needs_graph=needs_graph,
        needs_web=needs_web,
        confidence=confidence,
        reasoning_summary=summary,
    )


def _has(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped
