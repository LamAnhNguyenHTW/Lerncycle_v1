"""Validation rules for learning-structure extraction output."""

from __future__ import annotations

import re
from typing import Any

from rag_pipeline.learning_structure.extractor import LearningExtraction
from rag_pipeline.learning_structure.models import (
    ChunkGroup,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
)


GENERIC_TITLE_BLOCKLIST = {
    "einleitung",
    "zusammenfassung",
    "kapitel",
    "definition",
    "beispiel",
    "aufgabe",
    "abbildung",
    "tabelle",
    "lernziele",
    "introduction",
    "summary",
    "chapter",
    "example",
    "exercise",
    "figure",
    "table",
    "objectives",
}


def validate_extraction(
    extraction: LearningExtraction,
    group: ChunkGroup,
    config: Any,
) -> tuple[LearningExtraction, list[dict[str, Any]]]:
    """Return accepted extraction items and bounded rejection metadata."""
    valid_chunk_ids = {chunk.chunk_id for chunk in group.chunks}
    min_confidence = float(getattr(config, "learning_graph_min_confidence", 0.5))
    rejected: list[dict[str, Any]] = []
    accepted_topics = []
    accepted_concepts = []
    accepted_objectives = []

    for topic in extraction.topics:
        reason = _topic_rejection_reason(topic, valid_chunk_ids, min_confidence, group)
        if reason:
            rejected.append(_rejected("topic", topic.title, reason))
        else:
            accepted_topics.append(topic)

    for concept in extraction.concepts:
        reason = _item_rejection_reason(concept, valid_chunk_ids, min_confidence)
        if reason:
            rejected.append(_rejected("concept", concept.name, reason))
        else:
            accepted_concepts.append(concept)

    for objective in extraction.objectives:
        reason = _item_rejection_reason(objective, valid_chunk_ids, min_confidence)
        if reason:
            rejected.append(_rejected("objective", objective.objective, reason))
        else:
            accepted_objectives.append(objective)

    return (
        LearningExtraction(
            topics=accepted_topics,
            concepts=accepted_concepts,
            objectives=accepted_objectives,
        ),
        rejected,
    )


def _topic_rejection_reason(
    topic: ExtractedTopic,
    valid_chunk_ids: set[str],
    min_confidence: float,
    group: ChunkGroup,
) -> str | None:
    reason = _item_rejection_reason(topic, valid_chunk_ids, min_confidence)
    if reason:
        return reason
    if _is_generic_title(topic.title):
        return "generic_title"
    if not _valid_page_range(topic, group):
        return "invalid_page_range"
    return None


def _item_rejection_reason(
    item: ExtractedTopic | ExtractedConcept | ExtractedLearningObjective,
    valid_chunk_ids: set[str],
    min_confidence: float,
) -> str | None:
    chunk_ids = list(getattr(item, "chunk_ids", []) or [])
    if not chunk_ids:
        return "empty_chunk_ids"
    if any(chunk_id not in valid_chunk_ids for chunk_id in chunk_ids):
        return "unknown_chunk_id"
    if float(getattr(item, "confidence", 0)) < min_confidence:
        return "low_confidence"
    return None


def _is_generic_title(title: str) -> bool:
    normalized = _normalize_title(title)
    if normalized not in GENERIC_TITLE_BLOCKLIST:
        return False
    tokens = [token for token in normalized.split() if len(token) > 2]
    return len(tokens) <= 2


def _valid_page_range(topic: ExtractedTopic, group: ChunkGroup) -> bool:
    if topic.page_start is None or topic.page_end is None:
        return True
    if topic.page_start > topic.page_end:
        return False
    pages = [chunk.page_index for chunk in group.chunks if chunk.page_index is not None]
    if not pages:
        return True
    return min(pages) <= topic.page_start and topic.page_end <= max(pages)


def _normalize_title(title: str) -> str:
    value = title.strip().lower()
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _rejected(item_type: str, label: str, reason: str) -> dict[str, Any]:
    return {"type": item_type, "label": label, "reason": reason}
