from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.learning_structure.chunk_grouper import (
    build_groups,
    group_by_heading,
    group_by_page_window,
)
from rag_pipeline.learning_structure.models import ChunkForExtraction


@dataclass(frozen=True)
class GrouperConfig:
    learning_graph_max_chunks_per_group: int = 2


def _chunk(
    chunk_id: str,
    page: int | None,
    heading_path: list[str] | None = None,
) -> ChunkForExtraction:
    return ChunkForExtraction(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        page_index=page,
        heading_path=heading_path or [],
        content_hash=f"hash-{chunk_id}",
    )


def test_group_by_heading_uses_deepest_non_empty_heading_path_level() -> None:
    chunks = [
        _chunk("c1", 1, ["A", "Intro"]),
        _chunk("c2", 2, ["A", "Intro"]),
        _chunk("c3", 3, ["A", "Details"]),
    ]

    groups = group_by_heading(chunks, max_per_group=8)

    assert [group.group_id for group in groups] == ["heading:a/intro:0", "heading:a/details:0"]
    assert [group.heading_path for group in groups] == [["A", "Intro"], ["A", "Details"]]
    assert [[chunk.chunk_id for chunk in group.chunks] for group in groups] == [["c1", "c2"], ["c3"]]


def test_group_by_page_window_is_fallback_for_empty_or_uniform_headings() -> None:
    chunks = [
        _chunk("c1", 1, []),
        _chunk("c2", 2, []),
        _chunk("c3", 3, []),
        _chunk("c4", 4, []),
    ]

    groups = build_groups(chunks, GrouperConfig(learning_graph_max_chunks_per_group=2))

    assert [group.group_id for group in groups] == ["pages:1-2:0", "pages:3-4:0"]
    assert [[chunk.chunk_id for chunk in group.chunks] for group in groups] == [["c1", "c2"], ["c3", "c4"]]


def test_oversized_heading_group_is_split_by_max_chunks() -> None:
    chunks = [
        _chunk("c1", 1, ["A"]),
        _chunk("c2", 2, ["A"]),
        _chunk("c3", 3, ["A"]),
        _chunk("c4", 4, ["B"]),
    ]

    groups = group_by_heading(chunks, max_per_group=2)

    assert [group.group_id for group in groups] == ["heading:a:0", "heading:a:1", "heading:b:0"]
    assert [[chunk.chunk_id for chunk in group.chunks] for group in groups] == [["c1", "c2"], ["c3"], ["c4"]]


def test_grouping_order_is_deterministic_by_page_then_chunk_id() -> None:
    chunks = [
        _chunk("c3", 2, ["B"]),
        _chunk("c1", 1, ["A"]),
        _chunk("c2", 1, ["A"]),
    ]

    groups = group_by_heading(chunks, max_per_group=8)

    assert [group.group_id for group in groups] == ["heading:a:0", "heading:b:0"]
    assert [[chunk.chunk_id for chunk in group.chunks] for group in groups] == [["c1", "c2"], ["c3"]]


def test_group_by_page_window_handles_missing_pages_deterministically() -> None:
    chunks = [_chunk("c2", None, []), _chunk("c1", 1, []), _chunk("c3", None, [])]

    groups = group_by_page_window(chunks, pages_per_group=2)

    assert [[chunk.chunk_id for chunk in group.chunks] for group in groups] == [["c1"], ["c2", "c3"]]
