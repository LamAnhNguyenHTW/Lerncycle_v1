"""Normalize, deduplicate, and reattach extracted learning topics."""

from __future__ import annotations

from typing import TypeVar

from rag_pipeline.learning_structure.models import ExtractedConcept, ExtractedLearningObjective, ExtractedTopic
from rag_pipeline.learning_structure.normalizer import normalize_title


_SUMMARY_CAP = 1200
_Attachable = TypeVar("_Attachable", ExtractedConcept, ExtractedLearningObjective)


def dedupe_topics(topics: list[ExtractedTopic]) -> tuple[list[ExtractedTopic], dict[str, str]]:
    """Merge topics with the same normalized title, preserving deterministic IDs."""
    groups: dict[str, list[ExtractedTopic]] = {}
    for topic in topics:
        groups.setdefault(normalize_title(topic.title), []).append(topic)

    deduped: list[ExtractedTopic] = []
    merged_map: dict[str, str] = {}
    for normalized_title, group in groups.items():
        kept = group[0]
        merged_from = [topic.topic_id for topic in group[1:]]
        for merged_id in merged_from:
            merged_map[merged_id] = kept.topic_id

        if merged_from:
            kept = kept.model_copy(
                update={
                    "chunk_ids": _union_chunks(topic.chunk_ids for topic in group),
                    "page_start": _min_page(topic.page_start for topic in group),
                    "page_end": _max_page(topic.page_end for topic in group),
                    "confidence": max(topic.confidence for topic in group),
                    "summary": _merge_summaries(topic.summary for topic in group),
                    "merged_from": [*kept.merged_from, *merged_from],
                }
            )
        deduped.append(kept)

    return sorted(deduped, key=lambda topic: (_page_sort(topic.page_start), normalize_title(topic.title))), merged_map


def reattach_orphans(
    concepts: list[ExtractedConcept],
    objectives: list[ExtractedLearningObjective],
    surviving_topic_ids: set[str],
    merged_map: dict[str, str],
) -> tuple[list[ExtractedConcept], list[ExtractedLearningObjective]]:
    """Rewrite merged topic references and drop attachments to filtered topics."""
    return (
        _reattach_items(concepts, surviving_topic_ids, merged_map),
        _reattach_items(objectives, surviving_topic_ids, merged_map),
    )


def _reattach_items(
    items: list[_Attachable],
    surviving_topic_ids: set[str],
    merged_map: dict[str, str],
) -> list[_Attachable]:
    reattached: list[_Attachable] = []
    for item in items:
        topic_id = item.topic_id
        if not topic_id:
            continue
        target_id = merged_map.get(topic_id, topic_id)
        if target_id not in surviving_topic_ids:
            continue
        reattached.append(item.model_copy(update={"topic_id": target_id}))
    return reattached


def _document_order(topic: ExtractedTopic) -> tuple[int, str, str]:
    return (_page_sort(topic.page_start), normalize_title(topic.title), topic.topic_id)


def _page_sort(page: int | None) -> int:
    return page if page is not None else 10**9


def _union_chunks(chunks: object) -> list[str]:
    seen: set[str] = set()
    union: list[str] = []
    for chunk_ids in chunks:
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                union.append(chunk_id)
    return union


def _min_page(pages: object) -> int | None:
    values = [page for page in pages if page is not None]
    return min(values) if values else None


def _max_page(pages: object) -> int | None:
    values = [page for page in pages if page is not None]
    return max(values) if values else None


def _merge_summaries(summaries: object) -> str:
    merged = " ".join(summary.strip() for summary in summaries if summary.strip())
    return merged[:_SUMMARY_CAP]
