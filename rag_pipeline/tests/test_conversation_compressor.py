from __future__ import annotations

import pytest

from rag_pipeline.conversation_compressor import compress_conversation


class FakeLlm:
    def __init__(self, response: str = " summary ") -> None:
        self.response = response
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


class RaisingLlm:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("llm failed")


def test_compress_conversation_returns_summary() -> None:
    llm = FakeLlm(" Lernstand ")

    summary = compress_conversation([{"role": "user", "content": "Was ist PM?"}], llm)

    assert summary == "Lernstand"
    assert "User: Was ist PM?" in llm.calls[0]["user_prompt"]


def test_compress_conversation_uses_existing_summary() -> None:
    llm = FakeLlm()

    compress_conversation(
        [{"role": "assistant", "content": "PM nutzt Event Logs."}],
        llm,
        existing_summary="Bisher ging es um PM.",
    )

    assert "Existing memory summary" in llm.calls[0]["user_prompt"]
    assert "Bisher ging es um PM." in llm.calls[0]["user_prompt"]


def test_compress_conversation_empty_messages_returns_empty_without_llm_call() -> None:
    llm = FakeLlm()

    assert compress_conversation([], llm) == ""
    assert llm.calls == []


def test_compress_conversation_caps_summary_chars() -> None:
    summary = compress_conversation(
        [{"role": "user", "content": "x"}],
        FakeLlm("abcdef"),
        max_chars=3,
    )

    assert summary == "abc"


def test_compress_conversation_propagates_llm_error() -> None:
    with pytest.raises(RuntimeError, match="llm failed"):
        compress_conversation([{"role": "user", "content": "x"}], RaisingLlm())
