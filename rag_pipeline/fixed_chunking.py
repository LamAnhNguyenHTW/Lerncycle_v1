"""Fixed-size chunking baseline — evaluation only, never used by the worker.

The production pipeline chunks with Docling HybridChunker + SemanticRefiner
(`docling_hybrid_semantic_refinement`). This module provides the naive
fixed-size baseline the chunking evaluation compares against. It must not be
wired into `rag_pipeline.worker`; evaluation scripts index its output into a
separate, clearly named Qdrant collection.
"""

from __future__ import annotations

from rag_pipeline.text import normalize_content


CHUNKING_STRATEGY_NAME = "fixed_size_baseline"
CHUNKING_STRATEGY_VERSION = "v1"
DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150


def chunk_fixed_size(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split normalized text into fixed-size windows with character overlap.

    Boundaries are pulled back to the last whitespace inside the window so
    words are never cut, matching the common fixed-size-with-overlap baseline
    from the chunking literature. Deterministic for identical input.

    Args:
        text: Raw source text; normalized before splitting.
        max_chars: Window size; every returned chunk is at most this long.
        overlap_chars: Characters of trailing context repeated at the start
            of the next window. Must be smaller than max_chars.

    Returns:
        Non-empty chunk strings in document order.

    Raises:
        ValueError: If max_chars < 1 or overlap_chars is not in [0, max_chars).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    normalized = normalize_content(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        next_start = end - overlap_chars
        # Guarantee forward progress even for pathological boundary cases.
        if next_start <= start:
            next_start = start + 1
        else:
            # Snap forward to a word boundary so overlap never splits a word.
            boundary = normalized.rfind(" ", start, next_start + 1)
            if boundary > start:
                next_start = boundary + 1
        start = next_start
        while start < text_length and normalized[start].isspace():
            start += 1
    return chunks
