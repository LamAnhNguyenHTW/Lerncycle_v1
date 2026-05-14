from __future__ import annotations

import json
from types import SimpleNamespace

from rag_pipeline.query_understanding import (
    QueryRoute,
    build_query_understanding_prompt,
    fallback_query_understanding,
    understand_query,
)


class FakeLlm:
    def __init__(self, response: str, raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.raises:
            raise RuntimeError("llm failed")
        return self.response


def _payload(**overrides) -> str:
    payload = {
        "resolved_query": "wie viel verdient ein Process Mining Consultant in Deutschland",
        "question_type": "current_external_info",
        "route": "web_search",
        "needs_pdf": False,
        "needs_notes": False,
        "needs_annotations": False,
        "needs_chat_memory": False,
        "needs_graph": False,
        "needs_web": True,
        "should_show_sources": True,
        "confidence": 0.9,
        "reasoning_summary": "External salary question.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_understand_query_returns_structured_llm_result() -> None:
    llm = FakeLlm(_payload())

    result = understand_query(
        "wie viel verdient pm consultant?",
        recent_messages=[{"role": "user", "content": "pm bedeutet process mining"}],
        llm_client=llm,
        config=SimpleNamespace(intent_classifier_max_recent_messages=4, intent_classifier_max_message_chars=1000),
    )

    assert result.route == QueryRoute.WEB_SEARCH
    assert result.needs_web is True
    assert result.needs_pdf is False
    assert "Recent conversation:" in llm.calls[0]["user_prompt"]


def test_understand_query_falls_back_when_llm_fails() -> None:
    result = understand_query(
        "Was hatten wir besprochen?",
        llm_client=FakeLlm("{}", raises=True),
        config=SimpleNamespace(intent_classifier_max_recent_messages=4, intent_classifier_max_message_chars=1000),
    )

    assert result.needs_chat_memory is True
    assert result.route == QueryRoute.INTERNAL_RETRIEVAL


def test_understand_query_rejects_invalid_json_and_falls_back() -> None:
    result = understand_query(
        "Was ist aktuell neu?",
        llm_client=FakeLlm("not json"),
        config=SimpleNamespace(intent_classifier_max_recent_messages=4, intent_classifier_max_message_chars=1000),
    )

    assert result.route == QueryRoute.WEB_SEARCH
    assert result.needs_web is True


def test_fallback_query_understanding_uses_existing_heuristics() -> None:
    result = fallback_query_understanding("Was ist der Zusammenhang zwischen BPMN und Process Mining?")

    assert result.needs_graph is True
    assert result.route == QueryRoute.INTERNAL_RETRIEVAL


def test_query_understanding_prompt_includes_routing_guidance() -> None:
    prompt = build_query_understanding_prompt("wie viel ist das netto?")

    assert "conversation_only" in prompt
    assert "web_search" in prompt
