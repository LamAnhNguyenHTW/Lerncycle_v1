"""Chunk builders for non-PDF RAG sources."""

from __future__ import annotations

from typing import Any

from rag_pipeline.models import RagChunk, SourceRef
from rag_pipeline.refinement import SemanticRefiner
from rag_pipeline.text import (
    annotation_to_text,
    build_content_hash,
    tiptap_to_text,
)


def chunks_from_note(
    user_id: str,
    note_id: str,
    pdf_id: str | None,
    content: dict[str, Any] | None,
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
) -> list[RagChunk]:
    """Build chunks from a TipTap note document."""
    source = SourceRef(
        user_id=user_id,
        source_type="note",
        source_id=note_id,
        pdf_id=pdf_id,
        note_id=note_id,
    )
    text = tiptap_to_text(content)
    return _chunks_from_text(
        source,
        text,
        refiner,
        chunking_strategy,
        chunking_version,
        {"origin": "note"},
    )


def chunks_from_annotation(
    user_id: str,
    annotation_id: str,
    pdf_id: str,
    quote: str | None,
    comment: str | None,
    metadata: dict[str, Any],
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
) -> list[RagChunk]:
    """Build chunks from an annotation quote/comment pair."""
    source = SourceRef(
        user_id=user_id,
        source_type="annotation_comment",
        source_id=annotation_id,
        pdf_id=pdf_id,
        annotation_id=annotation_id,
    )
    text = annotation_to_text(quote, comment)
    return _chunks_from_text(
        source,
        text,
        refiner,
        chunking_strategy,
        chunking_version,
        {"origin": "annotation_comment", **metadata},
    )


def _chunks_from_text(
    source: SourceRef,
    text: str,
    refiner: SemanticRefiner,
    chunking_strategy: str,
    chunking_version: str,
    metadata: dict[str, Any],
) -> list[RagChunk]:
    chunks = []
    for refined in refiner.refine(text):
        chunks.append(
            RagChunk(
                source=source,
                content=refined,
                content_hash=build_content_hash(
                    refined,
                    source,
                    chunking_strategy,
                    chunking_version,
                ),
                page_index=metadata.get("page_index"),
                chunk_kind="text",
                metadata=metadata,
            )
        )
    return chunks

