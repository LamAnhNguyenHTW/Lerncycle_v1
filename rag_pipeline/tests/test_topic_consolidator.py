from __future__ import annotations

import json

from rag_pipeline.learning_structure.ids import make_extracted_topic_id
from rag_pipeline.learning_structure.models import ExtractedTopic
from rag_pipeline.learning_structure.topic_consolidator import TopicConsolidator


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def _topic(index: int) -> ExtractedTopic:
    return ExtractedTopic(
        topic_id=make_extracted_topic_id("source", "group", index),
        title=f"Topic {index}",
        summary=f"Topic {index} teaches a substantive document concept with enough evidence for consolidation.",
        level=1,
        parent_title=None,
        chunk_ids=[f"c{index}"],
        page_start=index,
        page_end=index,
        confidence=0.8,
        group_id="group",
        heading_path=[f"Topic {index}"],
        order_hint=index,
    )


def _response(ids: list[str]) -> str:
    return json.dumps(
        {
            "main_topics": [
                {
                    "title": "Main A",
                    "summary": "Main A teaches a coherent area built from the first candidate topics.",
                    "source_topic_ids": [ids[0]],
                    "subtopics": [
                        {
                            "title": "Sub A",
                            "summary": "Sub A teaches a focused supporting idea backed by candidate evidence.",
                            "source_topic_ids": [ids[1]],
                        }
                    ],
                },
                {
                    "title": "Main B",
                    "summary": "Main B teaches the remaining coherent area built from candidate topics.",
                    "source_topic_ids": [ids[2]],
                    "subtopics": [],
                },
            ]
        }
    )


def test_topic_consolidator_happy_path_maps_every_input_once() -> None:
    topics = [_topic(1), _topic(2), _topic(3)]
    llm = FakeLlm(_response([topic.topic_id for topic in topics]))

    hierarchy = TopicConsolidator(llm_client=llm).consolidate(topics)

    assert hierarchy is not None
    assert [topic.title for topic in hierarchy.main_topics] == ["Main A", "Main B"]
    assert topics[0].topic_id in llm.calls[0]["user_prompt"]


def test_topic_consolidator_falls_back_for_malformed_json() -> None:
    consolidator = TopicConsolidator(llm_client=FakeLlm("not-json"))
    hierarchy = consolidator.consolidate([_topic(1), _topic(2)])

    assert hierarchy is not None
    assert hierarchy.main_topics
    assert consolidator.last_diagnostics["reason"] == "consolidator_fallback"


def test_topic_consolidator_repairs_partial_mapping() -> None:
    topics = [_topic(1), _topic(2)]
    response = _response([topics[0].topic_id, topics[0].topic_id, topics[0].topic_id])

    consolidator = TopicConsolidator(llm_client=FakeLlm(response))
    hierarchy = consolidator.consolidate(topics)

    assert hierarchy is not None
    seen = [
        topic_id
        for main in hierarchy.main_topics
        for topic_id in [*main.source_topic_ids, *(sid for sub in main.subtopics for sid in sub.source_topic_ids)]
    ]
    assert set(seen) == {topic.topic_id for topic in topics}
    assert consolidator.last_diagnostics["duplicate_topic_ids"]


def test_topic_consolidator_repairs_hallucinated_ids() -> None:
    topics = [_topic(1), _topic(2), _topic(3)]
    response = _response([topics[0].topic_id, "missing-id", topics[2].topic_id])

    consolidator = TopicConsolidator(llm_client=FakeLlm(response))
    hierarchy = consolidator.consolidate(topics)

    assert hierarchy is not None
    assert consolidator.last_diagnostics["hallucinated_topic_ids"] == ["missing-id"]
    assert consolidator.last_diagnostics["missing_topic_ids"] == [topics[1].topic_id]


def test_topic_consolidator_empty_input_skips_llm() -> None:
    llm = FakeLlm("{}")

    hierarchy = TopicConsolidator(llm_client=llm).consolidate([])

    assert hierarchy is not None
    assert hierarchy.main_topics == []
    assert llm.calls == []
