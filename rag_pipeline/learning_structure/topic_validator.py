"""Evidence validation gate for final Learning Graph trees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rag_pipeline.learning_structure.models import LearningTreeNode
from rag_pipeline.learning_structure.topic_filter import is_generic_summary
from rag_pipeline.learning_structure.normalizer import normalize_title


ValidationReason = Literal["missing_evidence", "short_summary", "artifact_title", "low_confidence"]
_HARD_ARTIFACT_TITLES = {
    "agenda",
    "themen",
    "gliederung",
    "inhaltsverzeichnis",
    "outline",
    "recap",
    "vielen dank",
    "danke",
    "fragen",
    "q a",
    "kontakt",
    "ueber mich",
    "uber mich",
    "über mich",
    "about",
    "disclaimer",
}


@dataclass(frozen=True)
class ValidationResult:
    """Cleaned tree plus bounded rejection diagnostics."""

    tree: LearningTreeNode
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejection_samples: list[dict[str, Any]] = field(default_factory=list)


def validate_tree(tree: LearningTreeNode, config: Any) -> ValidationResult:
    """Drop invalid topic/subtopic nodes without raising."""
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    min_confidence = float(getattr(config, "learning_graph_min_confidence", 0.4))

    cleaned_children = [
        cleaned
        for child in tree.children
        if (cleaned := _clean_node(child, min_confidence, counts, samples, is_root=True)) is not None
    ]
    return ValidationResult(
        tree=tree.model_copy(update={"children": cleaned_children}),
        rejection_counts=counts,
        rejection_samples=samples,
    )


def _clean_node(
    node: LearningTreeNode,
    min_confidence: float,
    counts: dict[str, int],
    samples: list[dict[str, Any]],
    *,
    is_root: bool,
) -> LearningTreeNode | None:
    reason = _rejection_reason(node, min_confidence)
    if reason is not None:
        _record_rejection(node, reason, counts, samples)
        return None

    cleaned_children = [
        cleaned
        for child in node.children
        if (cleaned := _clean_node(child, min_confidence, counts, samples, is_root=False)) is not None
    ]
    cleaned = node.model_copy(update={"children": cleaned_children})

    if node.type in {"topic", "subtopic"} and node.children and not cleaned_children and set(node.chunk_ids) == _descendant_chunk_ids(node):
        _record_rejection(node, "missing_evidence", counts, samples)
        return None
    return cleaned


def _rejection_reason(node: LearningTreeNode, min_confidence: float) -> ValidationReason | None:
    if node.type not in {"topic", "subtopic"}:
        return None
    if not node.chunk_ids:
        return "missing_evidence"
    summary = (node.summary or "").strip()
    if len(summary) < 40 or is_generic_summary(summary, node.label):
        return "short_summary"
    if normalize_title(node.label) in _HARD_ARTIFACT_TITLES:
        return "artifact_title"
    if node.confidence is not None and node.confidence < min_confidence:
        return "low_confidence"
    return None


def _record_rejection(
    node: LearningTreeNode,
    reason: ValidationReason,
    counts: dict[str, int],
    samples: list[dict[str, Any]],
) -> None:
    counts[reason] = counts.get(reason, 0) + 1
    if len(samples) < 20:
        samples.append({"node_id": node.id, "title": node.label, "reason": reason})


def _descendant_chunk_ids(node: LearningTreeNode) -> set[str]:
    chunks: set[str] = set()
    for child in node.children:
        chunks.update(child.chunk_ids)
        chunks.update(_descendant_chunk_ids(child))
    return chunks
