from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.learning_structure.ids import make_extracted_topic_id
from rag_pipeline.learning_structure.models import (
    ChunkForExtraction,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
)
from rag_pipeline.learning_structure.normalizer import merge_duplicates, normalize_title
from rag_pipeline.learning_structure.coverage import compute_coverage


@dataclass(frozen=True)
class CoverageConfig:
    learning_graph_min_chunk_coverage: float = 0.5


def _topic(**overrides) -> ExtractedTopic:
    data = {
        "topic_id": make_extracted_topic_id("source", "group", 0),
        "title": "Process Mining",
        "summary": "Process mining teaches how event data is analyzed to understand and improve processes.",
        "level": 1,
        "parent_title": None,
        "chunk_ids": ["c1"],
        "page_start": 1,
        "page_end": 1,
        "confidence": 0.7,
        "group_id": "group",
        "heading_path": ["Process Mining"],
        "order_hint": 1,
    }
    data.update(overrides)
    return ExtractedTopic(**data)


def _concept(**overrides) -> ExtractedConcept:
    data = {
        "name": "Event Log",
        "definition": "Recorded events",
        "explanation": "Evidence base",
        "topic_title": "Process Mining",
        "chunk_ids": ["c1"],
        "difficulty": "medium",
        "confidence": 0.8,
    }
    data.update(overrides)
    return ExtractedConcept(**data)


def _objective(**overrides) -> ExtractedLearningObjective:
    data = {
        "objective": "Explain event logs",
        "topic_title": "Process Mining",
        "bloom_level": "understand",
        "chunk_ids": ["c2"],
        "confidence": 0.6,
    }
    data.update(overrides)
    return ExtractedLearningObjective(**data)


def test_normalize_title_collapses_punctuation_case_and_space() -> None:
    assert normalize_title("Process-Mining") == normalize_title(" Process Mining ")
    assert normalize_title("Änderungs-Management!") == "änderungs management"


def test_merge_duplicates_unions_chunk_ids_page_ranges_and_keeps_highest_confidence() -> None:
    merged = merge_duplicates(
        [
            _topic(title="Process-Mining", chunk_ids=["c1"], page_start=1, page_end=2, confidence=0.6),
            _topic(title=" Process Mining ", chunk_ids=["c2"], page_start=3, page_end=4, confidence=0.9),
        ]
    )

    assert len(merged) == 1
    assert merged[0].chunk_ids == ["c1", "c2"]
    assert merged[0].page_start == 1
    assert merged[0].page_end == 4
    assert merged[0].confidence == 0.9


def test_compute_coverage_reports_ratios_confidence_and_quality_flag() -> None:
    chunks = [
        ChunkForExtraction(chunk_id="c1", text="one", page_index=1),
        ChunkForExtraction(chunk_id="c2", text="two", page_index=2),
        ChunkForExtraction(chunk_id="c3", text="three", page_index=3),
    ]

    coverage = compute_coverage(
        chunks,
        accepted_topics=[_topic(chunk_ids=["c1"], page_start=1, page_end=1, confidence=0.7)],
        accepted_concepts=[_concept(chunk_ids=["c1"], confidence=0.8)],
        accepted_objectives=[_objective(chunk_ids=["c2"], confidence=0.6)],
        config=CoverageConfig(learning_graph_min_chunk_coverage=0.5),
    )

    assert coverage["chunk_coverage_ratio"] == 2 / 3
    assert coverage["page_coverage_ratio"] == 2 / 3
    assert coverage["avg_confidence"] == 0.7
    assert coverage["quality_flag"] == "ok"


def test_compute_coverage_marks_low_quality_below_threshold() -> None:
    chunks = [
        ChunkForExtraction(chunk_id="c1", text="one", page_index=1),
        ChunkForExtraction(chunk_id="c2", text="two", page_index=2),
        ChunkForExtraction(chunk_id="c3", text="three", page_index=3),
    ]

    coverage = compute_coverage(
        chunks,
        accepted_topics=[_topic(chunk_ids=["c1"])],
        accepted_concepts=[],
        accepted_objectives=[],
        config=CoverageConfig(learning_graph_min_chunk_coverage=0.5),
    )

    assert coverage["chunk_coverage_ratio"] == 1 / 3
    assert coverage["quality_flag"] == "low_quality"
