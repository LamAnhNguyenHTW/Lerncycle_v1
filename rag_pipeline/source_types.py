"""Shared RAG source type constants."""

from __future__ import annotations


MATERIAL_SOURCE_TYPES = ("pdf", "note", "annotation_comment")
CHAT_MEMORY_SOURCE_TYPE = "chat_memory"
WEB_SOURCE_TYPE = "web"
INTERNAL_SOURCE_TYPES = (*MATERIAL_SOURCE_TYPES, CHAT_MEMORY_SOURCE_TYPE)


def contains_chat_memory(source_types: list[str] | None) -> bool:
    """Return True when the list explicitly requests chat memory."""
    return bool(source_types and CHAT_MEMORY_SOURCE_TYPE in source_types)


def contains_web(source_types: list[str] | None) -> bool:
    """Return True when the list explicitly requests ephemeral web sources."""
    return bool(source_types and WEB_SOURCE_TYPE in source_types)
