"""LLM-backed flashcard and mock-test generation grounded in user chunks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from rag_pipeline.retrieval_tools import (
    RetrievalToolName,
    RetrievalToolRegistry,
    RetrievalToolRequest,
    RetrievalToolResult,
    RetrievalToolStatus,
)
from rag_pipeline.revision.models import (
    FlashcardBatch,
    GeneratedFlashcard,
    GeneratedMockQuestion,
    MockTestBatch,
)


logger = logging.getLogger(__name__)


_FLASHCARD_SYSTEM_PROMPT_DE = (
    "Du erstellst Lernkarten für eine Lernanwendung. Gib NUR gültiges JSON zurück "
    "mit der Struktur {\"cards\": [{\"front\": str, \"back\": str, "
    "\"source_chunk_ids\": [str, ...]}]}. Keine Erklärungen, kein Markdown, "
    "kein Code-Fence. Jede Karte muss aus dem bereitgestellten Material kommen. "
    "Vorderseite = kurze Frage oder Begriff. Rückseite = präzise Antwort in 1-3 Sätzen. "
    "source_chunk_ids verweist auf die chunk_id-Werte aus dem Kontext, die die Antwort stützen."
)
_FLASHCARD_SYSTEM_PROMPT_EN = (
    "You create flashcards for a learning app. Return ONLY valid JSON in the form "
    "{\"cards\": [{\"front\": str, \"back\": str, \"source_chunk_ids\": [str, ...]}]}. "
    "No prose, no markdown, no code fence. Every card must come from the provided "
    "material. Front = short question or term. Back = precise 1-3 sentence answer. "
    "source_chunk_ids must reference chunk_id values from the context."
)

_MOCKTEST_SYSTEM_PROMPT_DE = (
    "Du erstellst Multiple-Choice-Fragen (genau 4 Optionen, eine korrekte) für eine "
    "Lernanwendung. Gib NUR gültiges JSON zurück mit der Struktur {\"questions\": "
    "[{\"prompt\": str, \"choices\": [str, str, str, str], \"correct_index\": int (0-3), "
    "\"explanation\": str, \"source_chunk_ids\": [str, ...]}]}. Keine Erklärungen "
    "außerhalb des JSON, kein Markdown, kein Code-Fence. Jede Frage muss aus dem "
    "bereitgestellten Material kommen. Die korrekte Antwort muss eindeutig sein."
)
_MOCKTEST_SYSTEM_PROMPT_EN = (
    "You create multiple-choice questions (exactly 4 options, one correct) for a "
    "learning app. Return ONLY valid JSON in the form {\"questions\": [{\"prompt\": str, "
    "\"choices\": [str, str, str, str], \"correct_index\": int (0-3), \"explanation\": str, "
    "\"source_chunk_ids\": [str, ...]}]}. No prose outside the JSON, no markdown, no code "
    "fence. Every question must come from the provided material. The correct answer must "
    "be unambiguous."
)


def generate_flashcards(
    user_id: str,
    pdf_ids: list[str],
    count: int,
    language: str,
    registry: RetrievalToolRegistry,
    llm_client: Any,
    retrieval_top_k: int = 8,
    max_cards: int = 30,
) -> FlashcardBatch:
    """Generate up to `count` flashcards grounded in the user's PDF/note/annotation chunks."""
    if not user_id or not pdf_ids:
        return FlashcardBatch(cards=[])
    safe_count = max(1, min(int(count or 0), int(max_cards)))
    chunks = _collect_chunks(user_id, pdf_ids, registry, retrieval_top_k, language)
    if not chunks:
        logger.info("Flashcard generation: no chunks retrieved.", extra={"user_id": user_id})
        return FlashcardBatch(cards=[])

    system_prompt = _FLASHCARD_SYSTEM_PROMPT_DE if language == "de" else _FLASHCARD_SYSTEM_PROMPT_EN
    user_prompt = _build_user_prompt(
        chunks=chunks,
        count=safe_count,
        task="flashcards",
        language=language,
    )

    try:
        raw = llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = json.loads(_strip_json_fence(str(raw or "")))
        batch = FlashcardBatch.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, Exception):
        logger.warning("Flashcard generation failed; returning empty batch.")
        return FlashcardBatch(cards=[])

    cleaned = _filter_cards(batch.cards, allowed_chunk_ids={c.chunk_id for c in chunks})
    return FlashcardBatch(cards=cleaned[:safe_count])


def generate_mock_test(
    user_id: str,
    pdf_ids: list[str],
    count: int,
    language: str,
    registry: RetrievalToolRegistry,
    llm_client: Any,
    retrieval_top_k: int = 8,
    max_questions: int = 20,
) -> MockTestBatch:
    """Generate up to `count` MCQ questions grounded in the user's PDF/note/annotation chunks."""
    if not user_id or not pdf_ids:
        return MockTestBatch(questions=[])
    safe_count = max(1, min(int(count or 0), int(max_questions)))
    chunks = _collect_chunks(user_id, pdf_ids, registry, retrieval_top_k, language)
    if not chunks:
        logger.info("Mock test generation: no chunks retrieved.", extra={"user_id": user_id})
        return MockTestBatch(questions=[])

    system_prompt = _MOCKTEST_SYSTEM_PROMPT_DE if language == "de" else _MOCKTEST_SYSTEM_PROMPT_EN
    user_prompt = _build_user_prompt(
        chunks=chunks,
        count=safe_count,
        task="mocktest",
        language=language,
    )

    try:
        raw = llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = json.loads(_strip_json_fence(str(raw or "")))
        batch = MockTestBatch.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, Exception):
        logger.warning("Mock test generation failed; returning empty batch.")
        return MockTestBatch(questions=[])

    cleaned = _filter_questions(batch.questions, allowed_chunk_ids={c.chunk_id for c in chunks})
    return MockTestBatch(questions=cleaned[:safe_count])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_chunks(
    user_id: str,
    pdf_ids: list[str],
    registry: RetrievalToolRegistry,
    top_k: int,
    language: str,
) -> list[RetrievalToolResult]:
    """Collect PDF + note + annotation chunks scoped to the given PDFs.

    Uses a broad placeholder query ("overview / Überblick") so retrieval surfaces
    representative chunks instead of biasing toward a single sub-topic.
    """
    seen: set[str] = set()
    collected: list[RetrievalToolResult] = []
    placeholder_query = "Überblick wichtige Konzepte" if language == "de" else "overview key concepts"

    plan = [
        (RetrievalToolName.SEARCH_PDF_CHUNKS, {"pdf_ids": pdf_ids}, top_k),
        (RetrievalToolName.SEARCH_NOTES, {"pdf_ids": pdf_ids}, max(2, top_k // 2)),
        (RetrievalToolName.SEARCH_ANNOTATIONS, {"pdf_ids": pdf_ids}, max(2, top_k // 2)),
    ]
    for tool, filters, k in plan:
        try:
            outcome = registry.execute(
                RetrievalToolRequest(
                    tool=tool,
                    query=placeholder_query,
                    top_k=k,
                    user_id=user_id,
                    filters=filters,
                )
            )
        except Exception:
            logger.warning("Retrieval failed for tool %s during revision generation.", tool.value)
            continue
        if outcome.status not in {RetrievalToolStatus.SUCCESS, RetrievalToolStatus.EMPTY}:
            continue
        for result in outcome.results:
            if result.chunk_id and result.chunk_id not in seen:
                seen.add(result.chunk_id)
                collected.append(result)
    return collected


def _build_user_prompt(
    *,
    chunks: list[RetrievalToolResult],
    count: int,
    task: str,
    language: str,
) -> str:
    if task == "flashcards":
        instruction_de = f"Erstelle {count} Lernkarten aus dem folgenden Material."
        instruction_en = f"Create {count} flashcards from the following material."
    else:
        instruction_de = (
            f"Erstelle {count} Multiple-Choice-Fragen mit je 4 Antwortmöglichkeiten "
            "und genau einer richtigen Antwort."
        )
        instruction_en = (
            f"Create {count} multiple-choice questions, each with 4 options and exactly one "
            "correct answer."
        )

    lines: list[str] = []
    lines.append(instruction_de if language == "de" else instruction_en)
    lines.append("")
    lines.append("Verfügbares Material:" if language == "de" else "Available material:")
    for chunk in chunks:
        snippet = _truncate(chunk.text, 700)
        title = chunk.title or chunk.source_type or "chunk"
        lines.append(f"---")
        lines.append(f"chunk_id: {chunk.chunk_id}")
        lines.append(f"title: {title}")
        lines.append(f"source_type: {chunk.source_type}")
        lines.append(f"text: {snippet}")
    lines.append("---")
    lines.append("")
    if language == "de":
        lines.append(
            "Gib NUR das JSON-Objekt zurück. Verwende ausschließlich die oben angegebenen "
            "chunk_id-Werte in source_chunk_ids."
        )
    else:
        lines.append(
            "Return ONLY the JSON object. Use only the chunk_id values listed above for "
            "source_chunk_ids."
        )
    return "\n".join(lines)


def _filter_cards(
    cards: list[GeneratedFlashcard],
    allowed_chunk_ids: set[str],
) -> list[GeneratedFlashcard]:
    cleaned: list[GeneratedFlashcard] = []
    for card in cards:
        valid_ids = [cid for cid in card.source_chunk_ids if cid in allowed_chunk_ids]
        cleaned.append(
            GeneratedFlashcard(
                front=card.front,
                back=card.back,
                source_chunk_ids=valid_ids,
            )
        )
    return cleaned


def _filter_questions(
    questions: list[GeneratedMockQuestion],
    allowed_chunk_ids: set[str],
) -> list[GeneratedMockQuestion]:
    cleaned: list[GeneratedMockQuestion] = []
    for question in questions:
        valid_ids = [cid for cid in question.source_chunk_ids if cid in allowed_chunk_ids]
        cleaned.append(
            GeneratedMockQuestion(
                prompt=question.prompt,
                choices=question.choices,
                correct_index=question.correct_index,
                explanation=question.explanation,
                source_chunk_ids=valid_ids,
            )
        )
    return cleaned


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


def _truncate(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 1] + "…"
