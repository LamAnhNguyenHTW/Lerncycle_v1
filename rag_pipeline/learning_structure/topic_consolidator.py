"""Document-level topic hierarchy consolidation."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from rag_pipeline.learning_structure.models import (
    ConsolidatedHierarchy,
    ConsolidatedMainTopic,
    ConsolidatedSubtopic,
    ExtractedTopic,
)
from rag_pipeline.llm_client import OpenAILlmClient


CONSOLIDATION_SYSTEM_PROMPT = (
    "Consolidate candidate learning topics into a clear document hierarchy. Return "
    "strict JSON with main_topics. Target 4-8 main topics and 2-5 subtopics per "
    "main topic when the document is large enough. Every emitted main topic or "
    "subtopic must carry source_topic_ids copied verbatim from the provided input. "
    "Use source_topic_ids only; never source_titles. Every input topic_id must "
    "appear exactly once across all main-topic and subtopic source_topic_ids."
)


class TopicConsolidator:
    """Consolidate deduplicated extracted topics into a document-level hierarchy."""

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client
        self.last_diagnostics: dict[str, Any] = {}

    def consolidate(self, topics: list[ExtractedTopic], *, llm_client: Any = None) -> ConsolidatedHierarchy | None:
        """Return a validated hierarchy or None when the LLM response is unusable."""
        self.last_diagnostics = {}
        if not topics:
            return ConsolidatedHierarchy(main_topics=[])

        client = llm_client or self.llm_client or OpenAILlmClient()
        try:
            raw = client.complete(
                system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
                user_prompt=_build_prompt(topics),
            )
            hierarchy = ConsolidatedHierarchy.model_validate(json.loads(_strip_json_fence(raw)))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self.last_diagnostics = {"reason": "consolidator_fallback", "error": str(exc)[:500]}
            return _fallback_hierarchy(topics)

        repaired, diagnostics = _repair_source_mapping(hierarchy, topics)
        self.last_diagnostics = diagnostics
        return repaired


def _build_prompt(topics: list[ExtractedTopic]) -> str:
    items = [
        {
            "topic_id": topic.topic_id,
            "title": topic.title,
            "summary": topic.summary,
            "page_range": [topic.page_start, topic.page_end],
        }
        for topic in topics
    ]
    return json.dumps({"candidate_topics": items}, ensure_ascii=False)


def _repair_source_mapping(
    hierarchy: ConsolidatedHierarchy,
    topics: list[ExtractedTopic],
) -> tuple[ConsolidatedHierarchy, dict[str, Any]]:
    input_ids = {topic.topic_id for topic in topics}
    used: set[str] = set()
    duplicate_ids: list[str] = []
    hallucinated_ids: list[str] = []
    repaired_mains: list[ConsolidatedMainTopic] = []

    for main_topic in hierarchy.main_topics:
        main_ids, main_duplicates, main_hallucinations = _clean_ids(main_topic.source_topic_ids, input_ids, used)
        duplicate_ids.extend(main_duplicates)
        hallucinated_ids.extend(main_hallucinations)
        repaired_subtopics: list[ConsolidatedSubtopic] = []
        for subtopic in main_topic.subtopics:
            sub_ids, sub_duplicates, sub_hallucinations = _clean_ids(subtopic.source_topic_ids, input_ids, used)
            duplicate_ids.extend(sub_duplicates)
            hallucinated_ids.extend(sub_hallucinations)
            if sub_ids:
                repaired_subtopics.append(subtopic.model_copy(update={"source_topic_ids": sub_ids}))
        if main_ids or repaired_subtopics:
            repaired_mains.append(
                main_topic.model_copy(
                    update={
                        "source_topic_ids": main_ids,
                        "subtopics": repaired_subtopics,
                    }
                )
            )

    missing_ids = [topic.topic_id for topic in sorted(topics, key=_topic_order) if topic.topic_id not in used]
    if not repaired_mains:
        repaired = _fallback_hierarchy(topics)
    else:
        repaired = _append_missing_as_subtopics(ConsolidatedHierarchy(main_topics=repaired_mains), missing_ids, topics)

    diagnostics: dict[str, Any] = {
        "reason": "consolidator_repaired",
        "missing_topic_ids": missing_ids,
        "duplicate_topic_ids": duplicate_ids,
        "hallucinated_topic_ids": hallucinated_ids,
    }
    if not missing_ids and not duplicate_ids and not hallucinated_ids:
        diagnostics = {"reason": "consolidator_ok"}
    return repaired, diagnostics


def _clean_ids(
    source_topic_ids: list[str],
    input_ids: set[str],
    used: set[str],
) -> tuple[list[str], list[str], list[str]]:
    kept: list[str] = []
    duplicates: list[str] = []
    hallucinations: list[str] = []
    for topic_id in source_topic_ids:
        if topic_id not in input_ids:
            hallucinations.append(topic_id)
            continue
        if topic_id in used:
            duplicates.append(topic_id)
            continue
        used.add(topic_id)
        kept.append(topic_id)
    return kept, duplicates, hallucinations


def _append_missing_as_subtopics(
    hierarchy: ConsolidatedHierarchy,
    missing_ids: list[str],
    topics: list[ExtractedTopic],
) -> ConsolidatedHierarchy:
    if not missing_ids:
        return hierarchy
    by_id = {topic.topic_id: topic for topic in topics}
    main_topics = list(hierarchy.main_topics)
    if not main_topics:
        return _fallback_hierarchy(topics)
    for index, topic_id in enumerate(missing_ids):
        topic = by_id[topic_id]
        target_index = index % len(main_topics)
        target = main_topics[target_index]
        target_subtopics = [
            *target.subtopics,
            ConsolidatedSubtopic(title=topic.title, summary=topic.summary, source_topic_ids=[topic.topic_id]),
        ]
        main_topics[target_index] = target.model_copy(update={"subtopics": target_subtopics})
    return ConsolidatedHierarchy(main_topics=main_topics)


def _fallback_hierarchy(topics: list[ExtractedTopic]) -> ConsolidatedHierarchy:
    ordered = sorted(topics, key=_topic_order)
    if not ordered:
        return ConsolidatedHierarchy(main_topics=[])
    target_main_count = min(8, max(1, (len(ordered) + 3) // 4))
    buckets: list[list[ExtractedTopic]] = [[] for _ in range(target_main_count)]
    for index, topic in enumerate(ordered):
        buckets[index % target_main_count].append(topic)

    main_topics: list[ConsolidatedMainTopic] = []
    for bucket in buckets:
        if not bucket:
            continue
        main = bucket[0]
        subtopics = [
            ConsolidatedSubtopic(title=topic.title, summary=topic.summary, source_topic_ids=[topic.topic_id])
            for topic in bucket[1:]
        ]
        main_topics.append(
            ConsolidatedMainTopic(
                title=main.title,
                summary=main.summary,
                source_topic_ids=[main.topic_id],
                subtopics=subtopics,
            )
        )
    return ConsolidatedHierarchy(main_topics=main_topics)


def _topic_order(topic: ExtractedTopic) -> tuple[int, str]:
    page = topic.page_start if topic.page_start is not None else 10**9
    return (page, topic.title)


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text
