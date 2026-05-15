from __future__ import annotations

import json
from typing import Any

import pytest

from rag_pipeline.retrieval_tools import (
    RetrievalToolName,
    RetrievalToolOutcome,
    RetrievalToolResult,
    RetrievalToolStatus,
)
from rag_pipeline.revision.generator import (
    generate_flashcards,
    generate_mock_test,
)
from rag_pipeline.revision.models import GeneratedMockQuestion


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


class FakeRegistry:
    """Registry double that returns fixed chunks for each tool."""

    def __init__(self, chunks_by_tool: dict[RetrievalToolName, list[dict[str, Any]]]) -> None:
        self.chunks_by_tool = chunks_by_tool
        self.calls: list[Any] = []

    def execute(self, request: Any) -> RetrievalToolOutcome:
        self.calls.append(request)
        raw = self.chunks_by_tool.get(request.tool, [])
        results = [
            RetrievalToolResult(
                chunk_id=c["chunk_id"],
                source_type=c.get("source_type", "pdf"),
                source_id=c.get("source_id"),
                text=c.get("text", ""),
                score=c.get("score"),
                title=c.get("title"),
            )
            for c in raw
        ]
        return RetrievalToolOutcome(
            tool=request.tool,
            status=RetrievalToolStatus.SUCCESS if results else RetrievalToolStatus.EMPTY,
            results=results,
            result_count=len(results),
        )


USER_ID = "00000000-0000-0000-0000-000000000001"
PDF_ID = "00000000-0000-0000-0000-0000000000aa"


def _pdf_chunks() -> list[dict[str, Any]]:
    return [
        {"chunk_id": "c-1", "source_type": "pdf", "source_id": PDF_ID, "title": "Intro", "text": "Process Mining analyses event logs."},
        {"chunk_id": "c-2", "source_type": "pdf", "source_id": PDF_ID, "title": "Methods", "text": "Alpha algorithm discovers Petri nets."},
    ]


def test_generate_flashcards_returns_parsed_cards() -> None:
    llm_response = json.dumps({
        "cards": [
            {"front": "What does Process Mining analyse?", "back": "Event logs.", "source_chunk_ids": ["c-1"]},
            {"front": "What does the Alpha algorithm discover?", "back": "Petri nets.", "source_chunk_ids": ["c-2"]},
        ]
    })
    llm = FakeLlm(llm_response)
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_flashcards(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=5,
        language="en",
        registry=registry,
        llm_client=llm,
    )

    assert len(batch.cards) == 2
    assert batch.cards[0].front == "What does Process Mining analyse?"
    assert batch.cards[0].source_chunk_ids == ["c-1"]
    # registry must have been called with the user_id and pdf scope
    first = registry.calls[0]
    assert first.user_id == USER_ID
    assert first.tool == RetrievalToolName.SEARCH_PDF_CHUNKS
    assert first.filters.get("pdf_ids") == [PDF_ID]


def test_generate_flashcards_caps_to_count() -> None:
    cards = [
        {"front": f"Q{i}", "back": f"A{i}", "source_chunk_ids": ["c-1"]} for i in range(10)
    ]
    llm = FakeLlm(json.dumps({"cards": cards}))
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_flashcards(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=3,
        language="en",
        registry=registry,
        llm_client=llm,
    )
    assert len(batch.cards) == 3


def test_generate_flashcards_returns_empty_on_malformed_json() -> None:
    llm = FakeLlm("this is not json at all")
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_flashcards(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=5,
        language="en",
        registry=registry,
        llm_client=llm,
    )
    assert batch.cards == []


def test_generate_flashcards_filters_unknown_chunk_ids() -> None:
    llm_response = json.dumps({
        "cards": [
            {"front": "Q", "back": "A", "source_chunk_ids": ["c-1", "hallucinated-id"]},
        ]
    })
    llm = FakeLlm(llm_response)
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_flashcards(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=5,
        language="en",
        registry=registry,
        llm_client=llm,
    )
    assert batch.cards[0].source_chunk_ids == ["c-1"]


def test_generate_flashcards_strips_code_fence() -> None:
    fenced = "```json\n" + json.dumps({"cards": [{"front": "Q", "back": "A", "source_chunk_ids": ["c-1"]}]}) + "\n```"
    llm = FakeLlm(fenced)
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_flashcards(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=5,
        language="de",
        registry=registry,
        llm_client=llm,
    )
    assert len(batch.cards) == 1


def test_generate_flashcards_returns_empty_when_no_pdfs() -> None:
    llm = FakeLlm("{}")
    registry = FakeRegistry({})

    batch = generate_flashcards(
        user_id=USER_ID,
        pdf_ids=[],
        count=5,
        language="en",
        registry=registry,
        llm_client=llm,
    )
    assert batch.cards == []


def test_generate_mock_test_returns_parsed_questions() -> None:
    llm_response = json.dumps({
        "questions": [
            {
                "prompt": "What does PM analyse?",
                "choices": ["Event logs", "Images", "Audio", "Videos"],
                "correct_index": 0,
                "explanation": "Per chunk c-1.",
                "source_chunk_ids": ["c-1"],
            }
        ]
    })
    llm = FakeLlm(llm_response)
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_mock_test(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=5,
        language="en",
        registry=registry,
        llm_client=llm,
    )

    assert len(batch.questions) == 1
    assert len(batch.questions[0].choices) == 4
    assert 0 <= batch.questions[0].correct_index <= 3


def test_generate_mock_test_rejects_wrong_choice_count() -> None:
    # Pydantic should reject a question that has only 3 choices
    llm_response = json.dumps({
        "questions": [
            {"prompt": "Q?", "choices": ["a", "b", "c"], "correct_index": 0, "explanation": ""}
        ]
    })
    llm = FakeLlm(llm_response)
    registry = FakeRegistry({RetrievalToolName.SEARCH_PDF_CHUNKS: _pdf_chunks()})

    batch = generate_mock_test(
        user_id=USER_ID,
        pdf_ids=[PDF_ID],
        count=5,
        language="en",
        registry=registry,
        llm_client=llm,
    )
    # ValidationError → empty batch fallback
    assert batch.questions == []


def test_mock_question_model_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError):
        GeneratedMockQuestion(
            prompt="Q?",
            choices=["a", "b", "c", "d"],
            correct_index=4,
            explanation="",
        )
