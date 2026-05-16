from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.learning_structure.hierarchy_builder import build_hierarchy
from rag_pipeline.learning_structure.ids import make_extracted_topic_id
from rag_pipeline.learning_structure.models import (
    ConsolidatedHierarchy,
    ConsolidatedMainTopic,
    ConsolidatedSubtopic,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
)


@dataclass(frozen=True)
class HierarchyConfig:
    learning_graph_max_topics_per_doc: int = 30


def _topic(title: str, **overrides) -> ExtractedTopic:
    data = {
        "topic_id": make_extracted_topic_id("source", "group", sum(ord(char) for char in title)),
        "title": title,
        "summary": f"{title} teaches the learner a substantive concept from the document with concrete evidence.",
        "level": 1,
        "parent_title": None,
        "chunk_ids": [f"chunk-{title.lower().replace(' ', '-')}"],
        "page_start": 1,
        "page_end": 1,
        "confidence": 0.8,
        "group_id": "group",
        "heading_path": [title],
        "order_hint": None,
    }
    data.update(overrides)
    if "page_start" in overrides and "page_end" not in overrides:
        data["page_end"] = overrides["page_start"]
    return ExtractedTopic(**data)


def _concept(name: str, topic_title: str, **overrides) -> ExtractedConcept:
    data = {
        "name": name,
        "definition": "Definition",
        "explanation": "Explanation",
        "topic_title": topic_title,
        "chunk_ids": ["chunk"],
        "difficulty": "medium",
        "confidence": 0.7,
    }
    data.update(overrides)
    return ExtractedConcept(**data)


def _objective(objective: str, topic_title: str, **overrides) -> ExtractedLearningObjective:
    data = {
        "objective": objective,
        "topic_title": topic_title,
        "bloom_level": "understand",
        "chunk_ids": ["chunk"],
        "confidence": 0.7,
    }
    data.update(overrides)
    return ExtractedLearningObjective(**data)


def test_heading_path_prefix_makes_child_topic() -> None:
    tree = build_hierarchy(
        user_id="user",
        source_id="source",
        topics=[
            _topic("A", heading_path=["A"], page_start=1, page_end=1),
            _topic("B", level=2, heading_path=["A", "B"], page_start=2, page_end=2),
        ],
        concepts=[],
        objectives=[],
        config=HierarchyConfig(),
    )

    assert [node.label for node in tree.children] == ["A"]
    assert tree.children[0].children[0].label == "B"


def test_parent_title_used_when_no_heading_prefix_relationship() -> None:
    tree = build_hierarchy(
        user_id="user",
        source_id="source",
        topics=[
            _topic("Parent", heading_path=["X"], page_start=1),
            _topic("Child", parent_title="Parent", heading_path=["Y"], page_start=2),
        ],
        concepts=[],
        objectives=[],
        config=HierarchyConfig(),
    )

    assert tree.children[0].label == "Parent"
    assert tree.children[0].children[0].label == "Child"


def test_concept_and_objective_attach_to_most_specific_topic() -> None:
    tree = build_hierarchy(
        user_id="user",
        source_id="source",
        topics=[
            _topic("A", heading_path=["A"], page_start=1),
            _topic("B", level=2, heading_path=["A", "B"], page_start=2),
        ],
        concepts=[_concept("Event Log", topic_title="B")],
        objectives=[_objective("Explain B", topic_title="B")],
        config=HierarchyConfig(),
    )

    child = tree.children[0].children[0]
    assert [node.label for node in child.children] == ["Event Log", "Explain B"]
    assert {node.type for node in child.children} == {"concept", "objective"}


def test_order_index_preserves_document_order() -> None:
    tree = build_hierarchy(
        user_id="user",
        source_id="source",
        topics=[
            _topic("Second", page_start=2, heading_path=["Second"]),
            _topic("First", page_start=1, heading_path=["First"]),
        ],
        concepts=[],
        objectives=[],
        config=HierarchyConfig(),
    )

    assert [node.label for node in tree.children] == ["First", "Second"]
    assert [node.order_index for node in tree.children] == [0, 1]


def test_max_topics_per_doc_drops_lowest_ranked_top_level_subtree() -> None:
    tree = build_hierarchy(
        user_id="user",
        source_id="source",
        topics=[
            _topic("Keep", confidence=0.9, page_start=2, heading_path=["Keep"]),
            _topic("Drop", confidence=0.4, page_start=1, heading_path=["Drop"]),
            _topic("Dropped Child", confidence=0.3, parent_title="Drop", page_start=1, heading_path=["Drop", "Dropped Child"]),
        ],
        concepts=[_concept("Only Dropped", topic_title="Dropped Child")],
        objectives=[],
        config=HierarchyConfig(learning_graph_max_topics_per_doc=1),
    )

    assert [node.label for node in tree.children] == ["Keep"]
    assert "Only Dropped" not in str(tree.model_dump())


def test_consolidated_hierarchy_main_topic_inherits_descendant_evidence() -> None:
    own = _topic("Own", chunk_ids=["c1"], page_start=3, page_end=3)
    child = _topic("Child", chunk_ids=["c2"], page_start=5, page_end=6)
    hierarchy = ConsolidatedHierarchy(
        main_topics=[
            ConsolidatedMainTopic(
                title="Main",
                summary="Main teaches a consolidated area that includes its own evidence and child evidence.",
                source_topic_ids=[own.topic_id],
                subtopics=[
                    ConsolidatedSubtopic(
                        title="Sub",
                        summary="Sub teaches a narrower area backed only by its selected source topic.",
                        source_topic_ids=[child.topic_id],
                    )
                ],
            )
        ]
    )

    tree = build_hierarchy("user", "source", [own, child], [], [], HierarchyConfig(), consolidated_hierarchy=hierarchy)

    main = tree.children[0]
    sub = main.children[0]
    assert main.chunk_ids == ["c1", "c2"]
    assert main.page_start == 3
    assert main.page_end == 6
    assert sub.chunk_ids == ["c2"]


def test_consolidated_container_main_topic_inherits_subtopic_evidence() -> None:
    child = _topic("Child", chunk_ids=["c2"], page_start=5, page_end=6)
    hierarchy = ConsolidatedHierarchy(
        main_topics=[
            ConsolidatedMainTopic(
                title="Main",
                summary="Main teaches a consolidated area whose evidence comes from surviving subtopics.",
                source_topic_ids=[],
                subtopics=[
                    ConsolidatedSubtopic(
                        title="Sub",
                        summary="Sub teaches a narrower area backed by its selected source topic.",
                        source_topic_ids=[child.topic_id],
                    )
                ],
            )
        ]
    )

    tree = build_hierarchy("user", "source", [child], [], [], HierarchyConfig(), consolidated_hierarchy=hierarchy)

    assert tree.children[0].chunk_ids == ["c2"]
