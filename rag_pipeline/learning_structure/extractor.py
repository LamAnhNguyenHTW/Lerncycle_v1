"""LLM extraction of learning topics, concepts, and objectives from chunk groups."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_pipeline.learning_structure.models import (
    ChunkGroup,
    ExtractedConcept,
    ExtractedLearningObjective,
    ExtractedTopic,
)
from rag_pipeline.llm_client import OpenAILlmClient


LEARNING_EXTRACTION_SYSTEM_PROMPT = (
    "Extract a hierarchical learning structure from the provided chunks. Return strict "
    "JSON only with keys topics, concepts, and objectives. Use no outside knowledge. "
    "Every item must cite chunk_ids from the provided group. Topics require title, "
    "level, chunk_ids, and confidence. Concepts require name, topic_title, chunk_ids, "
    "difficulty, and confidence. Objectives require objective, topic_title, bloom_level, "
    "chunk_ids, and confidence. Do not include markdown."
)


class LearningExtractionError(Exception):
    """Raised when a learning-structure extraction cannot be parsed or validated."""


class LearningExtraction(BaseModel):
    """Validated extraction payload for one chunk group."""

    model_config = ConfigDict(extra="forbid")

    topics: list[ExtractedTopic] = Field(default_factory=list)
    concepts: list[ExtractedConcept] = Field(default_factory=list)
    objectives: list[ExtractedLearningObjective] = Field(default_factory=list)


class _LlmTopic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str = ""
    level: int = Field(ge=1)
    parent_title: str | None = None
    chunk_ids: list[str] = Field(min_length=1)
    page_start: int | None = None
    page_end: int | None = None
    confidence: float = Field(ge=0, le=1)


class _LlmExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: list[_LlmTopic] = Field(default_factory=list)
    concepts: list[ExtractedConcept] = Field(default_factory=list)
    objectives: list[ExtractedLearningObjective] = Field(default_factory=list)


class LearningExtractor:
    """Extract validated learning-graph items from a deterministic chunk group."""

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    def extract_from_group(
        self,
        group: ChunkGroup,
        *,
        llm_client: Any = None,
    ) -> LearningExtraction:
        """Extract learning topics, concepts, and objectives from one group."""
        if not group.chunks:
            return LearningExtraction()

        client = llm_client or self.llm_client or OpenAILlmClient()
        try:
            raw = client.complete(
                system_prompt=LEARNING_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=_build_prompt(group),
            )
            payload = json.loads(_strip_json_fence(raw))
            llm_extraction = _LlmExtraction.model_validate(_coerce_payload(payload, group))
            return LearningExtraction(
                topics=[
                    ExtractedTopic(
                        **topic.model_dump(),
                        group_id=group.group_id,
                        heading_path=group.heading_path,
                        order_hint=group.order_hint,
                    )
                    for topic in llm_extraction.topics
                ],
                concepts=llm_extraction.concepts,
                objectives=llm_extraction.objectives,
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise LearningExtractionError(str(exc)) from exc


def _build_prompt(group: ChunkGroup) -> str:
    chunks = []
    for chunk in group.chunks:
        heading = " > ".join(chunk.heading_path)
        page = chunk.page_index if chunk.page_index is not None else ""
        chunks.append(
            "\n".join(
                [
                    f"chunk_id: {chunk.chunk_id}",
                    f"page_index: {page}",
                    f"heading_path: {heading}",
                    "text:",
                    chunk.text[:3500],
                ]
            )
        )
    return "\n\n".join(
        [
            f"group_id: {group.group_id}",
            "Return JSON shape:",
            '{"topics":[],"concepts":[],"objectives":[]}',
            "Chunks:",
            "\n\n---\n\n".join(chunks),
        ]
    )


def _coerce_payload(payload: Any, group: ChunkGroup) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    chunk_ids = [chunk.chunk_id for chunk in group.chunks]
    default_topic = _default_topic_title(payload, group)
    return {
        "topics": [
            topic
            for item in _as_list(payload.get("topics"))
            if (topic := _coerce_topic(item, group, chunk_ids)) is not None
        ],
        "concepts": [
            concept
            for item in _as_list(payload.get("concepts"))
            if (concept := _coerce_concept(item, default_topic, chunk_ids)) is not None
        ],
        "objectives": [
            objective
            for item in _as_list(payload.get("objectives"))
            if (objective := _coerce_objective(item, default_topic, chunk_ids)) is not None
        ],
    }


def _coerce_topic(
    item: Any,
    group: ChunkGroup,
    group_chunk_ids: list[str],
) -> dict[str, Any] | None:
    if isinstance(item, str):
        data: dict[str, Any] = {"title": item}
    elif isinstance(item, dict):
        data = dict(item)
    else:
        return None

    title = _first_text(data, "title", "topic", "name", "heading", "label", "id")
    if not title:
        return None

    level = data.get("level")
    if not isinstance(level, int) or level < 1:
        level = max(1, len(group.heading_path))

    return {
        "title": title,
        "summary": _first_text(data, "summary", "description", "explanation") or "",
        "level": level,
        "parent_title": _first_text(data, "parent_title", "parent"),
        "chunk_ids": _coerce_chunk_ids(data, group_chunk_ids),
        "page_start": data.get("page_start", group.page_start),
        "page_end": data.get("page_end", group.page_end),
        "confidence": _coerce_confidence(data.get("confidence")),
    }


def _coerce_concept(
    item: Any,
    default_topic: str,
    group_chunk_ids: list[str],
) -> dict[str, Any] | None:
    if isinstance(item, str):
        data: dict[str, Any] = {"name": item}
    elif isinstance(item, dict):
        data = dict(item)
    else:
        return None

    name = _first_text(data, "name", "concept", "title", "term", "label")
    if not name:
        return None

    difficulty = data.get("difficulty")
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    definition = _first_text(data, "definition", "description") or ""
    return {
        "name": name,
        "definition": definition,
        "explanation": _first_text(data, "explanation", "description") or definition,
        "topic_title": _first_text(data, "topic_title", "topic", "parent_title") or default_topic,
        "chunk_ids": _coerce_chunk_ids(data, group_chunk_ids),
        "difficulty": difficulty,
        "confidence": _coerce_confidence(data.get("confidence")),
    }


def _coerce_objective(
    item: Any,
    default_topic: str,
    group_chunk_ids: list[str],
) -> dict[str, Any] | None:
    if isinstance(item, str):
        data: dict[str, Any] = {"objective": item}
    elif isinstance(item, dict):
        data = dict(item)
    else:
        return None

    objective = _first_text(data, "objective", "text", "description", "name", "title")
    if not objective:
        return None

    bloom_level = data.get("bloom_level")
    if bloom_level not in {"remember", "understand", "apply", "analyze", "evaluate", "create"}:
        bloom_level = "understand"
    return {
        "objective": objective,
        "topic_title": _first_text(data, "topic_title", "topic", "parent_title") or default_topic,
        "bloom_level": bloom_level,
        "chunk_ids": _coerce_chunk_ids(data, group_chunk_ids),
        "confidence": _coerce_confidence(data.get("confidence")),
    }


def _default_topic_title(payload: dict[str, Any], group: ChunkGroup) -> str:
    for item in _as_list(payload.get("topics")):
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, dict):
            title = _first_text(item, "title", "topic", "name", "heading", "label")
            if title:
                return title
    if group.heading_path:
        return group.heading_path[-1]
    return "Learning Topic"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _first_text(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_chunk_ids(data: dict[str, Any], fallback: list[str]) -> list[str]:
    for key in ("chunk_ids", "chunks", "evidence_chunk_ids", "source_chunk_ids"):
        value = data.get(key)
        if isinstance(value, list):
            chunk_ids = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if chunk_ids:
                return chunk_ids
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return fallback


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    return 0.6


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text
