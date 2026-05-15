from __future__ import annotations

import pytest

from rag_pipeline.learning_structure.extractor import LearningExtractionError, LearningExtractor
from rag_pipeline.learning_structure.models import ChunkForExtraction, ChunkGroup


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def _group(chunks: list[ChunkForExtraction] | None = None) -> ChunkGroup:
    return ChunkGroup(
        group_id="group-1",
        chunks=chunks
        if chunks is not None
        else [
            ChunkForExtraction(
                chunk_id="chunk-1",
                text="Process Mining discovers process models from event logs.",
                page_index=1,
                heading_path=["Process Mining"],
                content_hash="hash-1",
            )
        ],
        heading_path=["Process Mining"],
        page_start=1,
        page_end=1,
        order_hint=7,
    )


def _payload(chunk_id: str = "chunk-1") -> str:
    return (
        '{"topics":[{"title":"Process Mining","summary":"Analyzes process data",'
        '"level":1,"parent_title":null,"chunk_ids":["'
        + chunk_id
        + '"],"page_start":1,"page_end":1,"confidence":0.9}],'
        '"concepts":[{"name":"Event Log","definition":"Recorded events",'
        '"explanation":"Used as evidence","topic_title":"Process Mining",'
        '"chunk_ids":["'
        + chunk_id
        + '"],"difficulty":"medium","confidence":0.8}],'
        '"objectives":[{"objective":"Explain event logs",'
        '"topic_title":"Process Mining","bloom_level":"understand",'
        '"chunk_ids":["'
        + chunk_id
        + '"],"confidence":0.85}]}'
    )


def test_learning_extractor_extracts_and_enriches_topics() -> None:
    llm = FakeLlm(_payload())
    extraction = LearningExtractor().extract_from_group(_group(), llm_client=llm)

    assert extraction.topics[0].title == "Process Mining"
    assert extraction.topics[0].group_id == "group-1"
    assert extraction.topics[0].heading_path == ["Process Mining"]
    assert extraction.topics[0].order_hint == 7
    assert extraction.concepts[0].name == "Event Log"
    assert extraction.objectives[0].objective == "Explain event logs"
    assert "chunk-1" in llm.calls[0]["user_prompt"]
    assert "no outside knowledge" in llm.calls[0]["system_prompt"].lower()


def test_learning_extractor_strips_json_fences() -> None:
    extraction = LearningExtractor().extract_from_group(
        _group(),
        llm_client=FakeLlm(f"```json\n{_payload()}\n```"),
    )

    assert len(extraction.topics) == 1


def test_learning_extractor_rejects_malformed_json() -> None:
    with pytest.raises(LearningExtractionError):
        LearningExtractor().extract_from_group(_group(), llm_client=FakeLlm("not-json"))


def test_learning_extractor_allows_unknown_chunk_id_passthrough() -> None:
    extraction = LearningExtractor().extract_from_group(
        _group(),
        llm_client=FakeLlm(_payload("unknown-chunk")),
    )

    assert extraction.topics[0].chunk_ids == ["unknown-chunk"]


def test_learning_extractor_coerces_common_llm_schema_variants() -> None:
    response = (
        '{"topics":["Process Mining"],'
        '"concepts":[{"name":"Event Log"}],'
        '"objectives":[{"text":"Explain event logs"}]}'
    )

    extraction = LearningExtractor().extract_from_group(
        _group(),
        llm_client=FakeLlm(response),
    )

    assert extraction.topics[0].title == "Process Mining"
    assert extraction.topics[0].chunk_ids == ["chunk-1"]
    assert extraction.concepts[0].topic_title == "Process Mining"
    assert extraction.concepts[0].chunk_ids == ["chunk-1"]
    assert extraction.objectives[0].objective == "Explain event logs"
    assert extraction.objectives[0].chunk_ids == ["chunk-1"]


def test_learning_extractor_empty_group_skips_llm() -> None:
    llm = FakeLlm(_payload())
    extraction = LearningExtractor().extract_from_group(_group(chunks=[]), llm_client=llm)

    assert extraction.topics == []
    assert extraction.concepts == []
    assert extraction.objectives == []
    assert llm.calls == []
