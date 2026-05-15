"""Deterministic IDs for learning-graph nodes."""

from __future__ import annotations

import hashlib


def make_topic_id(
    user_id: str,
    source_id: str,
    normalized_title: str,
    level: int,
    page_start: int | None,
) -> str:
    """Return a stable SHA1 ID for a learning topic."""
    return _sha1_parts(user_id, source_id, normalized_title, str(level), str(page_start or 0))


def make_concept_id(
    user_id: str,
    source_id: str,
    parent_topic_id: str,
    normalized_name: str,
) -> str:
    """Return a stable SHA1 ID for a learning concept scoped to one topic."""
    return _sha1_parts(user_id, source_id, parent_topic_id, normalized_name)


def make_objective_id(
    user_id: str,
    source_id: str,
    parent_topic_id: str,
    normalized_objective: str,
) -> str:
    """Return a stable SHA1 ID for a learning objective scoped to one topic."""
    return _sha1_parts(user_id, source_id, parent_topic_id, normalized_objective)


def _sha1_parts(*parts: str) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
