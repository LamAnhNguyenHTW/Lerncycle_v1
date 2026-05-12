from __future__ import annotations

from rag_pipeline.memory_intent import detect_memory_intent


def test_detect_memory_intent_german_previous_discussion() -> None:
    assert detect_memory_intent("Was hatten wir zu Process Mining besprochen?")


def test_detect_memory_intent_german_learning_progress() -> None:
    assert detect_memory_intent("Was habe ich nicht verstanden?")


def test_detect_memory_intent_english_previous_discussion() -> None:
    assert detect_memory_intent("What did we discuss about event logs?")


def test_detect_memory_intent_normal_factual_question_false() -> None:
    assert not detect_memory_intent("Was ist Conformance Checking?")
