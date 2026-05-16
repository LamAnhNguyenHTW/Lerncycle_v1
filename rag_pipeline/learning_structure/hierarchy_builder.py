"""Build a hierarchical learning tree from accepted extraction items."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_pipeline.learning_structure.ids import (
    make_concept_id,
    make_objective_id,
    make_topic_id,
)
from rag_pipeline.learning_structure.models import (
    ConsolidatedHierarchy,
    ConsolidatedMainTopic,
    ConsolidatedSubtopic,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
    LearningTreeNode,
)
from rag_pipeline.learning_structure.normalizer import normalize_title


@dataclass
class _TopicNode:
    topic: ExtractedTopic
    id: str
    normalized_title: str
    parent: "_TopicNode | None" = None
    children: list["_TopicNode"] = field(default_factory=list)
    concepts: list[ExtractedConcept] = field(default_factory=list)
    objectives: list[ExtractedLearningObjective] = field(default_factory=list)
    order_index: int = 0


def build_hierarchy(
    user_id: str,
    source_id: str,
    topics: list[ExtractedTopic],
    concepts: list[ExtractedConcept],
    objectives: list[ExtractedLearningObjective],
    config: Any,
    consolidated_hierarchy: ConsolidatedHierarchy | None = None,
) -> LearningTreeNode:
    """Build a document-rooted hierarchy from validated learning items."""
    if consolidated_hierarchy is not None:
        return _build_consolidated_hierarchy(user_id, source_id, topics, consolidated_hierarchy)

    topic_nodes = [_make_topic_node(user_id, source_id, topic) for topic in topics]
    by_title = {node.normalized_title: node for node in topic_nodes}

    for node in topic_nodes:
        parent = _find_heading_parent(node, topic_nodes) or _find_parent_title(node, by_title)
        if parent and parent is not node:
            node.parent = parent
            parent.children.append(node)

    for concept in concepts:
        target = _find_attachment_topic(concept.topic_title, topic_nodes)
        if target:
            target.concepts.append(concept)
    for objective in objectives:
        target = _find_attachment_topic(objective.topic_title, topic_nodes)
        if target:
            target.objectives.append(objective)

    roots = [node for node in topic_nodes if node.parent is None]
    roots = _prune_roots(roots, int(getattr(config, "learning_graph_max_topics_per_doc", 30)))
    _assign_order_indexes(roots)

    return LearningTreeNode(
        id=f"document:{source_id}",
        label="Document",
        type="document",
        chunk_ids=[],
        children=[_to_tree_node(user_id, source_id, root, is_root=True) for root in roots],
    )


def _build_consolidated_hierarchy(
    user_id: str,
    source_id: str,
    topics: list[ExtractedTopic],
    hierarchy: ConsolidatedHierarchy,
) -> LearningTreeNode:
    by_id = {topic.topic_id: topic for topic in topics}
    children = [
        _consolidated_main_node(user_id, source_id, main_topic, by_id, index)
        for index, main_topic in enumerate(hierarchy.main_topics)
    ]
    return LearningTreeNode(
        id=f"document:{source_id}",
        label="Document",
        type="document",
        chunk_ids=[],
        children=children,
    )


def _consolidated_main_node(
    user_id: str,
    source_id: str,
    main_topic: ConsolidatedMainTopic,
    by_id: dict[str, ExtractedTopic],
    order_index: int,
) -> LearningTreeNode:
    subtopics = [
        _consolidated_subtopic_node(user_id, source_id, subtopic, by_id, index)
        for index, subtopic in enumerate(main_topic.subtopics)
    ]
    own_topics = _topics_for_ids(main_topic.source_topic_ids, by_id)
    descendant_topics = [
        topic
        for subtopic in main_topic.subtopics
        for topic in _topics_for_ids(subtopic.source_topic_ids, by_id)
    ]
    evidence_topics = [*own_topics, *descendant_topics]
    return LearningTreeNode(
        id=make_topic_id(user_id, source_id, normalize_title(main_topic.title), 1, _min_page(topic.page_start for topic in evidence_topics)),
        label=main_topic.title,
        type="topic",
        summary=main_topic.summary,
        page_start=_min_page(topic.page_start for topic in evidence_topics),
        page_end=_max_page(topic.page_end for topic in evidence_topics),
        confidence=_max_confidence(evidence_topics),
        order_index=order_index,
        chunk_ids=_union_chunks(topic.chunk_ids for topic in evidence_topics),
        children=subtopics,
    )


def _consolidated_subtopic_node(
    user_id: str,
    source_id: str,
    subtopic: ConsolidatedSubtopic,
    by_id: dict[str, ExtractedTopic],
    order_index: int,
) -> LearningTreeNode:
    evidence_topics = _topics_for_ids(subtopic.source_topic_ids, by_id)
    return LearningTreeNode(
        id=make_topic_id(user_id, source_id, normalize_title(subtopic.title), 2, _min_page(topic.page_start for topic in evidence_topics)),
        label=subtopic.title,
        type="subtopic",
        summary=subtopic.summary,
        page_start=_min_page(topic.page_start for topic in evidence_topics),
        page_end=_max_page(topic.page_end for topic in evidence_topics),
        confidence=_max_confidence(evidence_topics),
        order_index=order_index,
        chunk_ids=_union_chunks(topic.chunk_ids for topic in evidence_topics),
        children=[],
    )


def _topics_for_ids(topic_ids: list[str], by_id: dict[str, ExtractedTopic]) -> list[ExtractedTopic]:
    return [by_id[topic_id] for topic_id in topic_ids if topic_id in by_id]


def _union_chunks(chunks: object) -> list[str]:
    seen: set[str] = set()
    union: list[str] = []
    for chunk_ids in chunks:
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                union.append(chunk_id)
    return union


def _min_page(pages: object) -> int | None:
    values = [page for page in pages if page is not None]
    return min(values) if values else None


def _max_page(pages: object) -> int | None:
    values = [page for page in pages if page is not None]
    return max(values) if values else None


def _max_confidence(topics: list[ExtractedTopic]) -> float | None:
    return max((topic.confidence for topic in topics), default=None)


def _make_topic_node(user_id: str, source_id: str, topic: ExtractedTopic) -> _TopicNode:
    normalized = normalize_title(topic.title)
    topic_id = make_topic_id(user_id, source_id, normalized, topic.level, topic.page_start)
    return _TopicNode(topic=topic, id=topic_id, normalized_title=normalized)


def _find_heading_parent(node: _TopicNode, candidates: list[_TopicNode]) -> _TopicNode | None:
    path = [part.strip() for part in node.topic.heading_path if part.strip()]
    if len(path) < 2:
        return None
    prefixes = [tuple(path[:index]) for index in range(len(path) - 1, 0, -1)]
    for prefix in prefixes:
        for candidate in candidates:
            candidate_path = tuple(part.strip() for part in candidate.topic.heading_path if part.strip())
            if candidate is not node and candidate_path == prefix:
                return candidate
    return None


def _find_parent_title(node: _TopicNode, by_title: dict[str, _TopicNode]) -> _TopicNode | None:
    if not node.topic.parent_title:
        return None
    return by_title.get(normalize_title(node.topic.parent_title))


def _find_attachment_topic(topic_title: str, topic_nodes: list[_TopicNode]) -> _TopicNode | None:
    normalized = normalize_title(topic_title)
    matches = [node for node in topic_nodes if node.normalized_title == normalized]
    if not matches:
        return None
    return max(matches, key=lambda node: (len(node.topic.heading_path), node.topic.level))


def _assign_order_indexes(roots: list[_TopicNode]) -> None:
    counter = 0
    for node in _walk_sorted(roots):
        node.order_index = counter
        counter += 1


def _walk_sorted(nodes: list[_TopicNode]) -> list[_TopicNode]:
    result = []
    for node in sorted(nodes, key=_document_order):
        result.append(node)
        result.extend(_walk_sorted(node.children))
    return result


def _prune_roots(roots: list[_TopicNode], max_topics: int) -> list[_TopicNode]:
    if len(roots) <= max_topics:
        return sorted(roots, key=_document_order)
    ranked = sorted(
        roots,
        key=lambda node: (-_max_descendant_confidence(node), _document_order(node)),
    )
    kept = set(id(node) for node in ranked[:max_topics])
    return sorted([node for node in roots if id(node) in kept], key=_document_order)


def _max_descendant_confidence(node: _TopicNode) -> float:
    values = [node.topic.confidence]
    values.extend(_max_descendant_confidence(child) for child in node.children)
    values.extend(concept.confidence for concept in node.concepts)
    values.extend(objective.confidence for objective in node.objectives)
    return max(values)


def _document_order(node: _TopicNode) -> tuple[int, str]:
    page = node.topic.page_start if node.topic.page_start is not None else 10**9
    return (page, node.topic.title)


def _to_tree_node(
    user_id: str,
    source_id: str,
    node: _TopicNode,
    *,
    is_root: bool,
) -> LearningTreeNode:
    topic = node.topic
    children = [_to_tree_node(user_id, source_id, child, is_root=False) for child in sorted(node.children, key=_document_order)]
    children.extend(_concept_node(user_id, source_id, node.id, concept, index) for index, concept in enumerate(node.concepts))
    children.extend(_objective_node(user_id, source_id, node.id, objective, index) for index, objective in enumerate(node.objectives))
    return LearningTreeNode(
        id=node.id,
        label=topic.title,
        type="topic" if is_root else "subtopic",
        summary=topic.summary,
        page_start=topic.page_start,
        page_end=topic.page_end,
        confidence=topic.confidence,
        order_index=node.order_index,
        chunk_ids=topic.chunk_ids,
        children=children,
    )


def _concept_node(
    user_id: str,
    source_id: str,
    parent_topic_id: str,
    concept: ExtractedConcept,
    order_index: int,
) -> LearningTreeNode:
    normalized = normalize_title(concept.name)
    return LearningTreeNode(
        id=make_concept_id(user_id, source_id, parent_topic_id, normalized),
        label=concept.name,
        type="concept",
        summary=concept.definition or concept.explanation,
        confidence=concept.confidence,
        order_index=order_index,
        chunk_ids=concept.chunk_ids,
        children=[],
    )


def _objective_node(
    user_id: str,
    source_id: str,
    parent_topic_id: str,
    objective: ExtractedLearningObjective,
    order_index: int,
) -> LearningTreeNode:
    normalized = normalize_title(objective.objective)
    return LearningTreeNode(
        id=make_objective_id(user_id, source_id, parent_topic_id, normalized),
        label=objective.objective,
        type="objective",
        confidence=objective.confidence,
        order_index=order_index,
        chunk_ids=objective.chunk_ids,
        children=[],
    )
