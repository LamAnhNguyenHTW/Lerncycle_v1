from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.learning_structure.extractor import LearningExtraction
from rag_pipeline.learning_structure.ids import make_extracted_topic_id
from rag_pipeline.learning_structure.models import (
    ChunkForExtraction,
    ChunkGroup,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
)
from rag_pipeline.learning_structure.validator import validate_extraction


@dataclass(frozen=True)
class ValidatorConfig:
    learning_graph_min_confidence: float = 0.5


def _group() -> ChunkGroup:
    return ChunkGroup(
        group_id="group",
        chunks=[
            ChunkForExtraction(chunk_id="c1", text="One", page_index=1),
            ChunkForExtraction(chunk_id="c2", text="Two", page_index=2),
        ],
        heading_path=["Process Mining"],
        page_start=1,
        page_end=2,
        order_hint=1,
    )


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
        "confidence": 0.8,
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
        "chunk_ids": ["c1"],
        "confidence": 0.8,
    }
    data.update(overrides)
    return ExtractedLearningObjective(**data)


def test_validate_extraction_accepts_valid_items() -> None:
    accepted, rejected = validate_extraction(
        LearningExtraction(topics=[_topic()], concepts=[_concept()], objectives=[_objective()]),
        _group(),
        ValidatorConfig(),
    )

    assert len(accepted.topics) == 1
    assert len(accepted.concepts) == 1
    assert len(accepted.objectives) == 1
    assert rejected == []


def test_validate_extraction_rejects_empty_chunk_ids() -> None:
    topic = _topic()
    object.__setattr__(topic, "chunk_ids", [])

    accepted, rejected = validate_extraction(
        LearningExtraction(topics=[topic]),
        _group(),
        ValidatorConfig(),
    )

    assert accepted.topics == []
    assert rejected[0]["reason"] == "empty_chunk_ids"


def test_validate_extraction_rejects_unknown_chunk_ids() -> None:
    accepted, rejected = validate_extraction(
        LearningExtraction(topics=[_topic(chunk_ids=["missing"])]),
        _group(),
        ValidatorConfig(),
    )

    assert accepted.topics == []
    assert rejected[0]["reason"] == "unknown_chunk_id"


def test_validate_extraction_rejects_low_confidence() -> None:
    accepted, rejected = validate_extraction(
        LearningExtraction(
            topics=[_topic(confidence=0.4)],
            concepts=[_concept(confidence=0.4)],
            objectives=[_objective(confidence=0.4)],
        ),
        _group(),
        ValidatorConfig(learning_graph_min_confidence=0.5),
    )

    assert accepted.topics == []
    assert accepted.concepts == []
    assert accepted.objectives == []
    assert {item["reason"] for item in rejected} == {"low_confidence"}


def test_validate_extraction_rejects_generic_topic_titles() -> None:
    accepted, rejected = validate_extraction(
        LearningExtraction(topics=[_topic(title="Einleitung")]),
        _group(),
        ValidatorConfig(),
    )

    assert accepted.topics == []
    assert rejected[0]["reason"] == "generic_title"


def test_validate_extraction_allows_multi_token_generic_title_exception() -> None:
    accepted, rejected = validate_extraction(
        LearningExtraction(topics=[_topic(title="Lernziele des Process Mining Moduls")]),
        _group(),
        ValidatorConfig(),
    )

    assert accepted.topics[0].title == "Lernziele des Process Mining Moduls"
    assert rejected == []


def test_validate_extraction_rejects_invalid_page_ranges() -> None:
    accepted, rejected = validate_extraction(
        LearningExtraction(topics=[_topic(page_start=3, page_end=4)]),
        _group(),
        ValidatorConfig(),
    )

    assert accepted.topics == []
    assert rejected[0]["reason"] == "invalid_page_range"
