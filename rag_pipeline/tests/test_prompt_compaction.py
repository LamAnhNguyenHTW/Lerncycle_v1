from __future__ import annotations

from rag_pipeline.prompt_compaction import compress_conversation_summary


class FakeLlm:
    def __init__(self, response: str = "summary", raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.raises:
            raise RuntimeError("llm failed")
        return self.response


def test_compress_conversation_summary_returns_empty_without_messages_or_summary() -> None:
    assert compress_conversation_summary([], FakeLlm(), max_chars=300) == ""


def test_compress_conversation_summary_returns_existing_summary_without_messages() -> None:
    assert compress_conversation_summary([], FakeLlm(), max_chars=7, existing_summary="existing summary") == "existin"


def test_compress_conversation_summary_caps_llm_summary() -> None:
    summary = compress_conversation_summary(
        [{"role": "user", "content": "Hallo"}],
        FakeLlm("x" * 500),
        max_chars=300,
    )

    assert len(summary) == 300


def test_compress_conversation_summary_falls_back_on_llm_error() -> None:
    summary = compress_conversation_summary(
        [{"role": "user", "content": "Neue wichtige Entscheidung"}],
        FakeLlm(raises=True),
        max_chars=300,
        existing_summary="Alter Kontext",
    )

    assert "Alter Kontext" in summary
    assert "Neue wichtige Entscheidung" in summary
