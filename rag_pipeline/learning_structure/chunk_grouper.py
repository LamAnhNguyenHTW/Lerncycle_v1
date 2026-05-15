"""Deterministic chunk grouping for learning-graph extraction."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Iterable

from rag_pipeline.learning_structure.models import ChunkForExtraction, ChunkGroup


def group_by_heading(
    chunks: Iterable[ChunkForExtraction],
    max_per_group: int,
) -> list[ChunkGroup]:
    """Group chunks by their deepest non-empty heading path."""
    ordered_chunks = _ordered_chunks(chunks)
    grouped: "OrderedDict[tuple[str, ...], list[ChunkForExtraction]]" = OrderedDict()
    for chunk in ordered_chunks:
        heading_path = _clean_heading_path(chunk.heading_path)
        if not heading_path:
            continue
        grouped.setdefault(tuple(heading_path), []).append(chunk)

    groups: list[ChunkGroup] = []
    for heading_path, heading_chunks in grouped.items():
        groups.extend(
            _split_group(
                prefix=f"heading:{_slug_path(heading_path)}",
                chunks=heading_chunks,
                max_per_group=max_per_group,
                heading_path=list(heading_path),
            )
        )
    return groups


def group_by_page_window(
    chunks: Iterable[ChunkForExtraction],
    pages_per_group: int,
) -> list[ChunkGroup]:
    """Group chunks into deterministic page windows."""
    ordered_chunks = _ordered_chunks(chunks)
    if not ordered_chunks:
        return []

    with_pages = [chunk for chunk in ordered_chunks if chunk.page_index is not None]
    without_pages = [chunk for chunk in ordered_chunks if chunk.page_index is None]
    groups: list[ChunkGroup] = []

    if with_pages:
        current: list[ChunkForExtraction] = []
        window_start = int(with_pages[0].page_index or 0)
        window_end = window_start + max(1, pages_per_group) - 1
        for chunk in with_pages:
            page = int(chunk.page_index or 0)
            if current and page > window_end:
                groups.append(_make_group(f"pages:{window_start}-{window_end}:0", current, []))
                current = []
                window_start = page
                window_end = window_start + max(1, pages_per_group) - 1
            current.append(chunk)
        if current:
            groups.append(_make_group(f"pages:{window_start}-{window_end}:0", current, []))

    if without_pages:
        groups.extend(
            _split_group(
                prefix="pages:unknown",
                chunks=without_pages,
                max_per_group=max(1, pages_per_group),
                heading_path=[],
            )
        )
    return groups


def build_groups(chunks: Iterable[ChunkForExtraction], config: Any) -> list[ChunkGroup]:
    """Choose heading grouping when useful, otherwise fall back to page windows."""
    max_per_group = int(getattr(config, "learning_graph_max_chunks_per_group", 8))
    ordered_chunks = _ordered_chunks(chunks)
    heading_keys = {
        tuple(_clean_heading_path(chunk.heading_path))
        for chunk in ordered_chunks
        if _clean_heading_path(chunk.heading_path)
    }
    if len(heading_keys) > 1:
        return group_by_heading(ordered_chunks, max_per_group=max_per_group)
    return group_by_page_window(ordered_chunks, pages_per_group=max_per_group)


def _split_group(
    prefix: str,
    chunks: list[ChunkForExtraction],
    max_per_group: int,
    heading_path: list[str],
) -> list[ChunkGroup]:
    size = max(1, max_per_group)
    groups = []
    for index in range(0, len(chunks), size):
        groups.append(_make_group(f"{prefix}:{index // size}", chunks[index:index + size], heading_path))
    return groups


def _make_group(
    group_id: str,
    chunks: list[ChunkForExtraction],
    heading_path: list[str],
) -> ChunkGroup:
    pages = [chunk.page_index for chunk in chunks if chunk.page_index is not None]
    return ChunkGroup(
        group_id=group_id,
        chunks=chunks,
        heading_path=heading_path,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        order_hint=min(pages) if pages else None,
    )


def _ordered_chunks(chunks: Iterable[ChunkForExtraction]) -> list[ChunkForExtraction]:
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.page_index is None,
            chunk.page_index if chunk.page_index is not None else 0,
            chunk.chunk_id,
        ),
    )


def _clean_heading_path(heading_path: list[str]) -> list[str]:
    return [part.strip() for part in heading_path if str(part).strip()]


def _slug_path(heading_path: tuple[str, ...]) -> str:
    return "/".join(_slug(part) for part in heading_path)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or "untitled"
