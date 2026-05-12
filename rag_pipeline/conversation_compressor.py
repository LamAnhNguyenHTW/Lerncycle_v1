"""Conversation summary compression for chat memory."""

from __future__ import annotations

from typing import Any


COMPRESSOR_SYSTEM_PROMPT = (
    "Compress the conversation into a concise learning-memory summary for future "
    "study continuity. Preserve user goals, concepts discussed, misunderstandings, "
    "open questions, and preferred explanation style. Do not include raw message "
    "transcripts or private metadata."
)


def compress_conversation(
    messages: list[dict[str, Any]],
    llm_client: Any,
    max_chars: int = 2500,
    existing_summary: str | None = None,
) -> str:
    """Return a bounded rolling summary for chat memory."""
    formatted_messages = _format_messages(messages)
    if not formatted_messages:
        return ""

    sections = []
    if existing_summary:
        sections.append(f"Existing memory summary:\n{existing_summary.strip()}")
    sections.append(f"New conversation messages:\n{formatted_messages}")
    sections.append(
        "Write an updated learning-memory summary. Keep it factual, compact, "
        "and useful for answering later questions about what was discussed."
    )
    summary = llm_client.complete(
        system_prompt=COMPRESSOR_SYSTEM_PROMPT,
        user_prompt="\n\n".join(sections),
    )
    return str(summary).strip()[:max_chars]


def _format_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:2000]}")
    return "\n".join(lines)
