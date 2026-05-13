from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_pipeline.intent_classifier import (
    QuestionType,
    RetrievalIntent,
    build_intent_classifier_prompt,
    classify_intent,
    classify_intent_heuristic,
    default_intent,
)


class FakeLlm:
    def __init__(self, response: str, raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.raises:
            raise RuntimeError("secret")
        return self.response


def test_retrieval_intent_schema_valid() -> None:
    intent = RetrievalIntent(
        question_type="document_grounded",
        needs_pdf=True,
        needs_notes=True,
        needs_annotations=True,
        needs_chat_memory=False,
        needs_graph=False,
        needs_web=False,
        confidence=0.8,
        reasoning_summary="ok",
    )

    assert intent.question_type == QuestionType.DOCUMENT_GROUNDED


def test_retrieval_intent_rejects_invalid_question_type() -> None:
    with pytest.raises(ValueError):
        RetrievalIntent(
            question_type="bad",
            needs_pdf=True,
            needs_notes=True,
            needs_annotations=True,
            needs_chat_memory=False,
            needs_graph=False,
            needs_web=False,
            confidence=0.8,
            reasoning_summary="ok",
        )


def test_retrieval_intent_rejects_confidence_bounds_and_extra_fields() -> None:
    base = {
        "question_type": "document_grounded",
        "needs_pdf": True,
        "needs_notes": True,
        "needs_annotations": True,
        "needs_chat_memory": False,
        "needs_graph": False,
        "needs_web": False,
        "reasoning_summary": "ok",
    }
    with pytest.raises(ValueError):
        RetrievalIntent(**base, confidence=-0.1)
    with pytest.raises(ValueError):
        RetrievalIntent(**base, confidence=1.1)
    with pytest.raises(ValueError):
        RetrievalIntent(**base, confidence=0.5, raw="no")


def test_default_intent_safe_for_existing_rag() -> None:
    intent = default_intent()

    assert intent.needs_pdf is True
    assert intent.needs_notes is True
    assert intent.needs_annotations is True
    assert intent.needs_web is False


def test_prompt_includes_query_and_truncates_recent_messages() -> None:
    prompt = build_intent_classifier_prompt(
        "Was ist aktuell neu?",
        recent_messages=[
            {"role": "user", "content": "a" * 20},
            {"role": "assistant", "content": "b" * 20},
        ],
        max_recent_messages=1,
        max_message_chars=5,
    )

    assert "Was i" in prompt
    assert "b" * 5 in prompt
    assert "a" * 10 not in prompt
    assert "JSON only" in prompt
    assert "Web:" in prompt
    assert "Graph:" in prompt
    assert "Chat memory:" in prompt


def test_classify_intent_returns_valid_model_from_fake_llm() -> None:
    llm = FakeLlm(
        '{"question_type":"current_external_info","needs_pdf":false,"needs_notes":false,'
        '"needs_annotations":false,"needs_chat_memory":false,"needs_graph":false,'
        '"needs_web":true,"confidence":0.9,"reasoning_summary":"Current info."}'
    )

    intent = classify_intent("Was ist aktuell neu?", llm_client=llm)

    assert intent.needs_web is True
    assert intent.question_type == QuestionType.CURRENT_EXTERNAL_INFO


def test_classify_intent_invalid_json_and_llm_error_fall_back() -> None:
    assert classify_intent("Was ist aktuell neu?", llm_client=FakeLlm("not json")).needs_web is True
    assert classify_intent("Was ist aktuell neu?", llm_client=FakeLlm("", raises=True)).needs_web is True


def test_classify_intent_missing_api_key_falls_back() -> None:
    config = SimpleNamespace(openai_api_key=None, intent_classifier_fallback_enabled=True)

    assert classify_intent("Was ist aktuell neu?", config=config).needs_web is True


def test_classify_intent_empty_query_returns_general_chat() -> None:
    assert classify_intent(" ").question_type == QuestionType.GENERAL_CHAT


def test_heuristic_detects_memory_web_graph_notes_annotations_and_flashcards() -> None:
    assert classify_intent_heuristic("Was hatten wir zu BPMN besprochen?").needs_chat_memory is True
    assert classify_intent_heuristic("What did we discuss earlier?").needs_chat_memory is True
    assert classify_intent_heuristic("Was ist aktuell neu?").needs_web is True
    assert classify_intent_heuristic("latest OpenAI news").needs_web is True
    assert classify_intent_heuristic("Was ist der Zusammenhang zwischen A und B?").needs_graph is True
    assert classify_intent_heuristic("Was ist der Unterschied zwischen A und B?").question_type == QuestionType.COMPARISON
    assert classify_intent_heuristic("Erstelle einen Lernpfad").question_type == QuestionType.LEARNING_PATH
    assert classify_intent_heuristic("Fasse meine Notizen zusammen").needs_notes is True
    assert classify_intent_heuristic("Was steht in meinen Annotationen?").needs_annotations is True
    assert classify_intent_heuristic("Erstelle Karteikarten").question_type == QuestionType.QUIZ_OR_FLASHCARDS
    assert classify_intent_heuristic("Was steht in meiner PDF?").needs_web is False
