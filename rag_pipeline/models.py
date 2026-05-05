"""Shared data models for ingestion and chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SourceType = str


@dataclass(frozen=True)
class SourceRef:
    """Reference to a source document in Supabase."""

    user_id: str
    source_type: SourceType
    source_id: str
    pdf_id: str | None = None
    note_id: str | None = None
    annotation_id: str | None = None


@dataclass
class RagChunk:
    """A final chunk ready to persist in Supabase."""

    source: SourceRef
    content: str
    content_hash: str
    page_index: int | None = None
    heading_path: list[str] = field(default_factory=list)
    chunk_kind: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

