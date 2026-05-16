from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.learning_structure.models import LearningTreeNode
from rag_pipeline.learning_structure.topic_validator import validate_tree


@dataclass(frozen=True)
class ValidatorConfig:
    learning_graph_min_confidence: float = 0.4


def _node(
    node_id: str,
    label: str,
    node_type: str = "topic",
    *,
    chunk_ids: list[str] | None = None,
    summary: str | None = None,
    confidence: float | None = 0.8,
    children: list[LearningTreeNode] | None = None,
) -> LearningTreeNode:
    return LearningTreeNode(
        id=node_id,
        label=label,
        type=node_type,
        summary=summary or f"{label} teaches a substantive concept with enough detail for a mindmap node.",
        confidence=confidence,
        chunk_ids=chunk_ids if chunk_ids is not None else ["c1"],
        children=children or [],
    )


def _tree(children: list[LearningTreeNode]) -> LearningTreeNode:
    return LearningTreeNode(id="document:source", label="Document", type="document", chunk_ids=[], children=children)


def test_validate_tree_drops_missing_evidence_and_descendants() -> None:
    result = validate_tree(_tree([_node("t1", "Topic", chunk_ids=[], children=[_node("s1", "Sub", "subtopic")])]), ValidatorConfig())

    assert result.tree.children == []
    assert result.rejection_counts["missing_evidence"] == 1


def test_validate_tree_drops_short_summary_denylist_and_low_confidence() -> None:
    short = _node("t1", "Short", summary="Too short.")
    artifact = _node("t2", "Agenda")
    weak = _node("t3", "Weak", confidence=0.2)

    result = validate_tree(_tree([short, artifact, weak]), ValidatorConfig())

    assert result.tree.children == []
    assert result.rejection_counts == {"short_summary": 1, "artifact_title": 1, "low_confidence": 1}


def test_validate_tree_drops_only_bad_subtopic_and_keeps_siblings() -> None:
    good = _node("s1", "Good Sub", "subtopic", chunk_ids=["c2"])
    bad = _node("s2", "Bad Sub", "subtopic", chunk_ids=[])
    parent = _node("t1", "Parent", chunk_ids=["c1"], children=[good, bad])

    result = validate_tree(_tree([parent]), ValidatorConfig())

    assert [child.label for child in result.tree.children[0].children] == ["Good Sub"]


def test_validate_tree_drops_parent_when_all_child_evidence_was_removed_and_parent_has_no_own_evidence() -> None:
    parent = _node(
        "t1",
        "Parent",
        chunk_ids=["child-only"],
        children=[_node("s1", "Bad Sub", "subtopic", chunk_ids=["child-only"], summary="Too short.")],
    )

    result = validate_tree(_tree([parent]), ValidatorConfig())

    assert result.tree.children == []


def test_validate_tree_keeps_container_when_surviving_subtopic_carries_evidence() -> None:
    parent = _node("t1", "Parent", chunk_ids=["c2"], children=[_node("s1", "Good Sub", "subtopic", chunk_ids=["c2"])])

    result = validate_tree(_tree([parent]), ValidatorConfig())

    assert result.tree.children[0].label == "Parent"
    assert result.tree.children[0].chunk_ids == ["c2"]
    assert result.rejection_samples == []
