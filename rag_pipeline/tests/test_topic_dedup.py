from __future__ import annotations

from rag_pipeline.learning_structure.ids import make_extracted_topic_id
from rag_pipeline.learning_structure.models import ExtractedConcept, ExtractedLearningObjective, ExtractedTopic
from rag_pipeline.learning_structure.topic_dedup import dedupe_topics, reattach_orphans


def _topic(title: str, index: int, **overrides) -> ExtractedTopic:
    data = {
        "topic_id": make_extracted_topic_id("source", "group", index),
        "title": title,
        "summary": f"{title} teaches a substantive part of the material and explains evidence in enough detail.",
        "level": 1,
        "parent_title": None,
        "chunk_ids": [f"c{index}"],
        "page_start": index,
        "page_end": index,
        "confidence": 0.6,
        "group_id": "group",
        "heading_path": [title],
        "order_hint": index,
    }
    data.update(overrides)
    if "page_start" in overrides and "page_end" not in overrides:
        data["page_end"] = overrides["page_start"]
    return ExtractedTopic(**data)


def _concept(topic_id: str, topic_title: str = "Process Mining") -> ExtractedConcept:
    return ExtractedConcept(
        topic_id=topic_id,
        name="Event Log",
        definition="Recorded events",
        explanation="Evidence for process analysis",
        topic_title=topic_title,
        chunk_ids=["c1"],
        difficulty="medium",
        confidence=0.8,
    )


def _objective(topic_id: str, topic_title: str = "Process Mining") -> ExtractedLearningObjective:
    return ExtractedLearningObjective(
        topic_id=topic_id,
        objective="Explain event logs",
        topic_title=topic_title,
        bloom_level="understand",
        chunk_ids=["c1"],
        confidence=0.8,
    )


def test_dedupe_topics_merges_normalized_titles_and_tracks_sources() -> None:
    first = _topic("Process-Mining", 1, chunk_ids=["c1", "c2"], page_start=2, page_end=3, confidence=0.6)
    second = _topic(" Process Mining ", 2, chunk_ids=["c2", "c3"], page_start=1, page_end=5, confidence=0.9)

    deduped, merged_map = dedupe_topics([first, second])

    assert len(deduped) == 1
    assert deduped[0].topic_id == first.topic_id
    assert deduped[0].merged_from == [second.topic_id]
    assert deduped[0].chunk_ids == ["c1", "c2", "c3"]
    assert deduped[0].page_start == 1
    assert deduped[0].page_end == 5
    assert deduped[0].confidence == 0.9
    assert merged_map == {second.topic_id: first.topic_id}


def test_dedupe_topics_keeps_distinct_titles_in_document_order() -> None:
    later = _topic("Zeta", 1, page_start=10)
    earlier = _topic("Alpha", 2, page_start=1)

    deduped, merged_map = dedupe_topics([later, earlier])

    assert [topic.title for topic in deduped] == ["Alpha", "Zeta"]
    assert merged_map == {}


def test_reattach_orphans_rewrites_merged_ids_and_drops_filtered_ids() -> None:
    kept = _topic("Process Mining", 1)
    merged = _topic("Process-Mining", 2)
    filtered = _topic("Agenda", 3)

    concepts, objectives = reattach_orphans(
        [_concept(kept.topic_id), _concept(merged.topic_id), _concept(filtered.topic_id)],
        [_objective(merged.topic_id), _objective(filtered.topic_id)],
        {kept.topic_id},
        {merged.topic_id: kept.topic_id},
    )

    assert [concept.topic_id for concept in concepts] == [kept.topic_id, kept.topic_id]
    assert [objective.topic_id for objective in objectives] == [kept.topic_id]
