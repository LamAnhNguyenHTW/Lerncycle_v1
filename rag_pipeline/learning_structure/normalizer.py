"""Normalization and duplicate merging for learning-structure items."""

from __future__ import annotations

import string
import unicodedata
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def normalize_title(value: str) -> str:
    """Normalize titles for per-document duplicate detection."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    translation = str.maketrans({char: " " for char in string.punctuation})
    text = text.translate(translation)
    return " ".join(text.split())


def merge_duplicates(items: list[T]) -> list[T]:
    """Merge exact normalized duplicates, unioning evidence and page ranges."""
    merged: dict[str, T] = {}
    order: list[str] = []
    for item in items:
        key = _dedupe_key(item)
        if key not in merged:
            merged[key] = item
            order.append(key)
            continue
        merged[key] = _merge_pair(merged[key], item)
    return [merged[key] for key in order]


def _dedupe_key(item: BaseModel) -> str:
    for attr in ("title", "name", "objective"):
        value = getattr(item, attr, None)
        if value:
            return normalize_title(str(value))
    return normalize_title(str(item))


def _merge_pair(first: T, second: T) -> T:
    winner = first if float(getattr(first, "confidence", 0)) >= float(getattr(second, "confidence", 0)) else second
    data = winner.model_dump()
    data["chunk_ids"] = sorted(set(getattr(first, "chunk_ids", []) or []) | set(getattr(second, "chunk_ids", []) or []))
    page_starts = [
        page
        for page in [getattr(first, "page_start", None), getattr(second, "page_start", None)]
        if page is not None
    ]
    page_ends = [
        page
        for page in [getattr(first, "page_end", None), getattr(second, "page_end", None)]
        if page is not None
    ]
    if page_starts:
        data["page_start"] = min(page_starts)
    if page_ends:
        data["page_end"] = max(page_ends)
    return type(winner).model_validate(data)
