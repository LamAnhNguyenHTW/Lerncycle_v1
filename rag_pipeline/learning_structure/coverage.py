"""Coverage and quality metrics for learning-graph extraction."""

from __future__ import annotations

from typing import Any

from rag_pipeline.learning_structure.models import (
    ChunkForExtraction,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
)


def compute_coverage(
    chunks: list[ChunkForExtraction],
    accepted_topics: list[ExtractedTopic],
    accepted_concepts: list[ExtractedConcept],
    accepted_objectives: list[ExtractedLearningObjective],
    config: Any,
) -> dict[str, float | str]:
    """Compute chunk/page coverage, average confidence, and quality flag."""
    all_items = [*accepted_topics, *accepted_concepts, *accepted_objectives]
    total_chunk_ids = {chunk.chunk_id for chunk in chunks}
    referenced_chunk_ids = {
        chunk_id
        for item in all_items
        for chunk_id in (getattr(item, "chunk_ids", []) or [])
        if chunk_id in total_chunk_ids
    }
    chunk_coverage_ratio = _ratio(len(referenced_chunk_ids), len(total_chunk_ids))

    page_by_chunk = {
        chunk.chunk_id: chunk.page_index
        for chunk in chunks
        if chunk.page_index is not None
    }
    total_pages = set(page_by_chunk.values())
    referenced_pages = {
        page_by_chunk[chunk_id]
        for chunk_id in referenced_chunk_ids
        if chunk_id in page_by_chunk
    }
    page_coverage_ratio = _ratio(len(referenced_pages), len(total_pages))
    confidences = [float(getattr(item, "confidence", 0)) for item in all_items]
    avg_confidence = round(sum(confidences) / len(confidences), 6) if confidences else 0.0
    min_chunk_coverage = float(getattr(config, "learning_graph_min_chunk_coverage", 0.35))
    return {
        "chunk_coverage_ratio": chunk_coverage_ratio,
        "page_coverage_ratio": page_coverage_ratio,
        "avg_confidence": avg_confidence,
        "quality_flag": "low_quality" if chunk_coverage_ratio < min_chunk_coverage else "ok",
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
