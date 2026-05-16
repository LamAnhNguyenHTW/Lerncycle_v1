"""Deterministic pre-write filtering for extracted learning topics."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from rag_pipeline.learning_structure.models import ExtractedConcept, ExtractedLearningObjective, ExtractedTopic
from rag_pipeline.learning_structure.normalizer import normalize_title


RejectionReason = Literal["artifact", "low_content", "generic_summary"]


class TopicRejection(TypedDict, total=False):
    topic_id: str
    title: str
    reason: RejectionReason
    evidence_chars: int
    summary_chars: int
    generic_summary: bool
    confidence: float


_HARD_ARTIFACT_TITLES = {
    "agenda",
    "themen",
    "gliederung",
    "inhaltsverzeichnis",
    "outline",
    "recap",
    "vielen dank",
    "danke",
    "fragen",
    "q a",
    "kontakt",
    "ueber mich",
    "uber mich",
    "über mich",
    "about",
    "disclaimer",
}
_REFERENCE_TITLES = {"literatur", "references", "quellen", "bibliographie"}
_GENERIC_PATTERNS = (
    r"^\s*dieses thema behandelt\b",
    r"^\s*in diesem abschnitt geht es um\b",
    r"^\s*this topic discusses\b",
    r"^\s*this section explains\b",
    r"^\s*es wird erl[aä]utert\s*,?\s*dass\b",
)


def is_slide_artifact(
    topic: ExtractedTopic,
    document_page_count: int | None,
    chunk_text_by_id: dict[str, str] | None = None,
) -> bool:
    """Return true when a topic is slide meta rather than subject matter."""
    normalized = normalize_title(topic.title)
    if normalized in _HARD_ARTIFACT_TITLES:
        return True
    if normalized not in _REFERENCE_TITLES:
        return False

    if not document_page_count or topic.page_start is None:
        return False

    min_chars = 200
    evidence_chars = _evidence_chars(topic, chunk_text_by_id or {})
    is_late = topic.page_start >= max(1, int(document_page_count * 0.85))
    weak_summary = len(topic.summary.strip()) < 80 or is_generic_summary(topic.summary, topic.title)
    return is_late and evidence_chars < min_chars and weak_summary


def is_low_content_topic(
    topic: ExtractedTopic,
    chunk_text_by_id: dict[str, str],
    *,
    min_topic_chars: int = 200,
) -> bool:
    """Return true when the topic lacks enough summary or cited evidence."""
    if len(topic.summary.strip()) < 40:
        return True
    evidence_chars = _evidence_chars(topic, chunk_text_by_id)
    if evidence_chars < 80:
        return True
    return evidence_chars < min_topic_chars and is_generic_summary(topic.summary, topic.title)


def is_generic_summary(summary: str, title: str) -> bool:
    """Detect generic or title-restating summaries used as quality signals."""
    summary_text = summary.strip()
    lowered = summary_text.lower()
    if any(re.search(pattern, lowered) for pattern in _GENERIC_PATTERNS):
        return True

    normalized_title = normalize_title(title)
    normalized_summary = normalize_title(summary_text)
    if normalized_summary == normalized_title:
        return True
    return normalized_summary.startswith(f"{normalized_title} {normalized_title} ")


def filter_candidates(
    topics: list[ExtractedTopic],
    concepts: list[ExtractedConcept],
    objectives: list[ExtractedLearningObjective],
    document_meta: Any,
    chunk_text_by_id: dict[str, str],
) -> tuple[list[ExtractedTopic], list[ExtractedConcept], list[ExtractedLearningObjective], list[TopicRejection]]:
    """Filter rejected topics and any concept/objective attached to them."""
    document_page_count = _read_meta(document_meta, "document_page_count")
    min_topic_chars = int(_read_meta(document_meta, "learning_graph_min_topic_chars") or 200)

    kept_topics: list[ExtractedTopic] = []
    kept_titles: set[str] = set()
    rejections: list[TopicRejection] = []

    for topic in topics:
        reason: RejectionReason | None = None
        evidence_chars = _evidence_chars(topic, chunk_text_by_id)
        summary_chars = len(topic.summary.strip())
        generic_summary = is_generic_summary(topic.summary, topic.title)
        if is_slide_artifact(topic, document_page_count, chunk_text_by_id):
            reason = "artifact"
        elif summary_chars >= 40 and generic_summary and evidence_chars < min_topic_chars:
            reason = "generic_summary"
        elif is_low_content_topic(topic, chunk_text_by_id, min_topic_chars=min_topic_chars):
            reason = "low_content"

        if reason is None:
            kept_topics.append(topic)
            kept_titles.add(normalize_title(topic.title))
        else:
            rejections.append(
                {
                    "topic_id": topic.topic_id,
                    "title": topic.title,
                    "reason": reason,
                    "evidence_chars": evidence_chars,
                    "summary_chars": summary_chars,
                    "generic_summary": generic_summary,
                    "confidence": topic.confidence,
                }
            )

    kept_concepts = [concept for concept in concepts if normalize_title(concept.topic_title) in kept_titles]
    kept_objectives = [objective for objective in objectives if normalize_title(objective.topic_title) in kept_titles]
    return kept_topics, kept_concepts, kept_objectives, rejections


def _evidence_chars(topic: ExtractedTopic, chunk_text_by_id: dict[str, str]) -> int:
    return sum(len(chunk_text_by_id.get(chunk_id, "")) for chunk_id in topic.chunk_ids)


def _read_meta(document_meta: Any, name: str) -> Any:
    if isinstance(document_meta, dict):
        return document_meta.get(name)
    return getattr(document_meta, name, None)
