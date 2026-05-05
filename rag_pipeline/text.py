"""Text normalization, source extraction, and content hashing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from rag_pipeline.models import SourceRef


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_content(content: str) -> str:
    """Normalize content before chunking and hashing.

    Args:
        content: Raw text.

    Returns:
        Stable, stripped text with compact whitespace.
    """
    text = content.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def build_content_hash(
    content: str,
    source: SourceRef,
    chunking_strategy: str,
    chunking_version: str,
) -> str:
    """Hash normalized content and context for idempotent upserts.

    Args:
        content: Chunk content.
        source: Source reference.
        chunking_strategy: Strategy name.
        chunking_version: Strategy version.

    Returns:
        Hex SHA-256 digest.
    """
    payload = {
        "content": normalize_content(content),
        "source_type": source.source_type,
        "source_id": source.source_id,
        "chunking_strategy": chunking_strategy,
        "chunking_version": chunking_version,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tiptap_to_text(node: dict[str, Any] | None) -> str:
    """Convert TipTap JSON content into stable plain text.

    Args:
        node: TipTap JSON document.

    Returns:
        Plain text suitable for chunking.
    """
    if not node:
        return ""
    lines: list[str] = []
    _collect_tiptap_blocks(node, lines)
    return normalize_content("\n".join(lines))


def annotation_to_text(quote: str | None, comment: str | None) -> str:
    """Normalize an annotation quote/comment pair.

    Args:
        quote: Highlighted source text.
        comment: User comment.

    Returns:
        Combined text for chunking.
    """
    parts = []
    if quote and quote.strip():
        parts.append(f"Quote: {quote.strip()}")
    if comment and comment.strip():
        parts.append(f"Comment: {comment.strip()}")
    return normalize_content("\n".join(parts))


def split_paragraphs(content: str) -> list[str]:
    """Split content into normalized paragraphs.

    Args:
        content: Raw text.

    Returns:
        Non-empty paragraphs.
    """
    normalized = normalize_content(content)
    if not normalized:
        return []
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]


def split_sentences(content: str) -> list[str]:
    """Split text into sentence-like units without external dependencies."""
    normalized = normalize_content(content)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _collect_tiptap_blocks(node: dict[str, Any], lines: list[str]) -> None:
    node_type = node.get("type")
    content = node.get("content")
    if not isinstance(content, list):
        return

    if node_type in {
        "paragraph",
        "heading",
        "blockquote",
        "listItem",
        "taskItem",
        "codeBlock",
    }:
        text = _inline_tiptap_text(node)
        if text:
            lines.append(text)
        return

    for child in content:
        if isinstance(child, dict):
            _collect_tiptap_blocks(child, lines)


def _inline_tiptap_text(node: dict[str, Any]) -> str:
    node_type = node.get("type")
    if node_type == "text":
        text = node.get("text")
        return text if isinstance(text, str) else ""

    content = node.get("content")
    if not isinstance(content, list):
        return ""

    parts = []
    for child in content:
        if isinstance(child, dict):
            parts.append(_inline_tiptap_text(child))
    return normalize_content("".join(parts))
