from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_pipeline.learning_structure.ids import (
    make_concept_id,
    make_objective_id,
    make_extracted_topic_id,
    make_topic_id,
)
from rag_pipeline.learning_structure.models import (
    ChunkForExtraction,
    ChunkGroup,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
    ExtractionReport,
    ConsolidatedHierarchy,
    ConsolidatedMainTopic,
    ConsolidatedSubtopic,
    LearningTreeNode,
)


def test_learning_structure_ids_are_deterministic_and_sensitive() -> None:
    first = make_topic_id("user", "source", "process mining", 1, 3)
    assert first == make_topic_id("user", "source", "process mining", 1, 3)
    assert first != make_topic_id("user", "source", "process mining", 2, 3)
    assert len(first) == 40

    concept = make_concept_id("user", "source", first, "event log")
    objective = make_objective_id("user", "source", first, "explain event logs")
    assert concept == make_concept_id("user", "source", first, "event log")
    assert concept != objective


def test_extracted_topic_id_is_deterministic_and_sensitive() -> None:
    first = make_extracted_topic_id("source", "group-1", 0)

    assert first == make_extracted_topic_id("source", "group-1", 0)
    assert first != make_extracted_topic_id("source", "group-1", 1)
    assert first != make_extracted_topic_id("source", "group-2", 0)
    assert len(first) == 40


def test_extracted_topic_requires_topic_id() -> None:
    with pytest.raises(ValidationError):
        ExtractedTopic(
            title="Process Mining",
            summary="A field for analyzing event data.",
            level=1,
            parent_title=None,
            chunk_ids=["chunk-1"],
            page_start=2,
            page_end=2,
            confidence=0.9,
            group_id="group-1",
            heading_path=["Process Mining"],
            order_hint=1,
        )


def test_extracted_topic_requires_non_trivial_summary() -> None:
    base = {
        "topic_id": make_extracted_topic_id("source", "group-1", 0),
        "title": "Process Mining",
        "level": 1,
        "parent_title": None,
        "chunk_ids": ["chunk-1"],
        "page_start": 2,
        "page_end": 2,
        "confidence": 0.9,
        "group_id": "group-1",
        "heading_path": ["Process Mining"],
        "order_hint": 1,
    }

    with pytest.raises(ValidationError):
        ExtractedTopic(summary="", **base)

    with pytest.raises(ValidationError):
        ExtractedTopic(summary="Too short.", **base)


def test_learning_structure_models_accept_valid_payloads() -> None:
    chunk = ChunkForExtraction(
        chunk_id="chunk-1",
        text="Process mining discovers process models from event logs.",
        page_index=2,
        heading_path=["Process Mining"],
        content_hash="hash",
    )
    group = ChunkGroup(
        group_id="group-1",
        chunks=[chunk],
        heading_path=["Process Mining"],
        page_start=2,
        page_end=2,
        order_hint=1,
    )
    topic = ExtractedTopic(
        topic_id=make_extracted_topic_id("source", group.group_id, 0),
        title="Process Mining",
        summary="Process mining teaches how event data can be analyzed to discover and improve processes.",
        level=1,
        parent_title=None,
        chunk_ids=["chunk-1"],
        page_start=2,
        page_end=2,
        confidence=0.9,
        group_id=group.group_id,
        heading_path=group.heading_path,
        order_hint=group.order_hint,
    )
    concept = ExtractedConcept(
        name="Event Log",
        definition="A record of process events.",
        explanation="Event logs are the evidence base for process mining.",
        topic_title="Process Mining",
        chunk_ids=["chunk-1"],
        difficulty="medium",
        confidence=0.8,
    )
    objective = ExtractedLearningObjective(
        objective="Explain how event logs support process discovery.",
        topic_title="Process Mining",
        bloom_level="understand",
        chunk_ids=["chunk-1"],
        confidence=0.85,
    )
    tree = LearningTreeNode(
        id="topic-id",
        label=topic.title,
        type="topic",
        summary=topic.summary,
        page_start=topic.page_start,
        page_end=topic.page_end,
        confidence=topic.confidence,
        chunk_ids=topic.chunk_ids,
        children=[],
    )

    assert group.chunks[0].chunk_id == "chunk-1"
    assert concept.difficulty == "medium"
    assert objective.bloom_level == "understand"
    assert tree.children == []


def test_models_validate_confidence_and_enum_values() -> None:
    with pytest.raises(ValidationError):
        ExtractedTopic(
            topic_id=make_extracted_topic_id("source", "group", 0),
            title="X",
            summary="Too short.",
            level=1,
            parent_title=None,
            chunk_ids=["chunk-1"],
            page_start=1,
            page_end=1,
            confidence=1.2,
            group_id="group",
            heading_path=[],
            order_hint=None,
        )


def test_consolidated_hierarchy_models_are_id_based() -> None:
    subtopic = ConsolidatedSubtopic(
        title="Event Logs",
        summary="Event logs teach how process executions are captured as evidence for later analysis.",
        source_topic_ids=["topic-2"],
    )
    main = ConsolidatedMainTopic(
        title="Process Mining Foundations",
        summary="Process mining foundations teach the core inputs and goals needed to analyze process behavior.",
        source_topic_ids=["topic-1"],
        subtopics=[subtopic],
    )
    hierarchy = ConsolidatedHierarchy(main_topics=[main])

    assert hierarchy.main_topics[0].subtopics[0].source_topic_ids == ["topic-2"]
    with pytest.raises(ValidationError):
        ConsolidatedSubtopic(
            title="Invalid",
            summary="This summary is long enough but lacks any source topic identifier.",
            source_topic_ids=[],
        )
    with pytest.raises(ValidationError):
        ConsolidatedMainTopic(
            title="Invalid",
            summary="Too short.",
            source_topic_ids=["topic-1"],
            subtopics=[],
        )
    with pytest.raises(ValidationError):
        ConsolidatedSubtopic(
            title="Invalid",
            summary="This summary is long enough for validation and should reject source titles.",
            source_topic_ids=["topic-1"],
            source_titles=["Process Mining"],
        )
    with pytest.raises(ValidationError):
        ExtractedConcept(
            name="X",
            definition="Y",
            explanation="Z",
            topic_title="Topic",
            chunk_ids=["chunk-1"],
            difficulty="expert",
            confidence=0.7,
        )


def test_extraction_report_caps_rejected_samples() -> None:
    report = ExtractionReport(
        total_groups=30,
        successful_groups=10,
        failed_groups=20,
        accepted_topics=1,
        accepted_concepts=2,
        accepted_objectives=3,
        rejected_count=30,
        rejected_samples=[
            {"item": f"item-{index}", "error": "x" * 1000}
            for index in range(30)
        ],
        avg_confidence=0.8,
        chunk_coverage_ratio=0.5,
        page_coverage_ratio=0.4,
        quality_flag="ok",
    )

    assert len(report.rejected_samples) == 20
    assert all(len(str(sample)) <= 500 for sample in report.rejected_samples)
