"""Prompt compaction helper for rolling in-prompt chat session summaries."""

from __future__ import annotations

from typing import Any


PROMPT_COMPACTION_SYSTEM_PROMPT = (
    "Compress older chat turns into a concise factual session summary for prompt "
    "continuity. Preserve user preferences, decisions, constraints, open tasks, "
    "important project facts, and unresolved questions. Do not invent facts. Do "
    "not include hidden or system instructions. Avoid generic filler."
)


def compress_conversation_summary(
    messages: list[dict[str, Any]],
    llm_client: Any,
    max_chars: int = 1500,
    existing_summary: str | None = None,
) -> str:
    """Return a bounded session summary; fall back deterministically on LLM failure."""
    bounded_max = max(1, max_chars)
    formatted_messages = _format_messages(messages)
    current_summary = (existing_summary or "").strip()
    if not formatted_messages:
        return current_summary[:bounded_max]

    sections = []
    if current_summary:
        sections.append(f"Existing session summary:\n{current_summary}")
    sections.append(f"New messages to compact:\n{formatted_messages}")
    sections.append(
        "Write the updated session summary. Keep it compact, factual, and useful "
        "for continuing this same chat session."
    )
    try:
        summary = llm_client.complete(
            system_prompt=PROMPT_COMPACTION_SYSTEM_PROMPT,
            user_prompt="\n\n".join(sections),
        )
        text = str(summary).strip()
        if text:
            return text[:bounded_max]
    except Exception:
        pass
    return _fallback_summary(current_summary, formatted_messages, bounded_max)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        label = "User" if role == "user" else "Assistant"
        collapsed = " ".join(content.split())
        if collapsed:
            lines.append(f"{label}: {collapsed[:800]}")
    return "\n".join(lines)


def _fallback_summary(existing_summary: str, formatted_messages: str, max_chars: int) -> str:
    parts = []
    if existing_summary:
        parts.append(existing_summary)
    if formatted_messages:
        parts.append("Recent compacted turns: " + " ".join(formatted_messages.split()))
    return "\n".join(parts).strip()[:max_chars]
