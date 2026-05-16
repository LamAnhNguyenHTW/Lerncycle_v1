"""Pydantic models for hierarchical learning-graph extraction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkForExtraction(_StrictModel):
    """A Supabase chunk prepared for learning-structure extraction."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_index: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    content_hash: str | None = None


class ChunkGroup(_StrictModel):
    """A deterministic group of chunks submitted to one LLM extraction call."""

    group_id: str = Field(min_length=1)
    chunks: list[ChunkForExtraction] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    order_hint: int | None = None


class ExtractedTopic(_StrictModel):
    """One validated topic extracted from a chunk group."""

    topic_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=40)
    level: int = Field(ge=1)
    parent_title: str | None = None
    chunk_ids: list[str] = Field(min_length=1)
    page_start: int | None = None
    page_end: int | None = None
    confidence: float = Field(ge=0, le=1)
    group_id: str = Field(min_length=1)
    heading_path: list[str] = Field(default_factory=list)
    order_hint: int | None = None
    merged_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _page_range_is_ordered(self) -> "ExtractedTopic":
        if self.page_start is not None and self.page_end is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be <= page_end")
        return self


class ExtractedConcept(_StrictModel):
    """One concept attached to an extracted topic."""

    name: str = Field(min_length=1)
    definition: str = ""
    explanation: str = ""
    topic_id: str | None = None
    topic_title: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    confidence: float = Field(ge=0, le=1)


class ExtractedLearningObjective(_StrictModel):
    """One learning objective attached to an extracted topic."""

    objective: str = Field(min_length=1)
    topic_id: str | None = None
    topic_title: str = Field(min_length=1)
    bloom_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
    chunk_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExtractionReport(_StrictModel):
    """Bounded extraction report safe to persist in job metadata."""

    total_groups: int = Field(ge=0)
    successful_groups: int = Field(ge=0)
    failed_groups: int = Field(ge=0)
    accepted_topics: int = Field(ge=0)
    accepted_concepts: int = Field(ge=0)
    accepted_objectives: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    rejected_samples: list[dict[str, Any]] = Field(default_factory=list)
    avg_confidence: float = Field(ge=0, le=1)
    chunk_coverage_ratio: float = Field(ge=0, le=1)
    page_coverage_ratio: float = Field(ge=0, le=1)
    quality_flag: Literal["ok", "low_quality"]

    @field_validator("rejected_samples", mode="before")
    @classmethod
    def _cap_rejected_samples(cls, value: Any) -> list[dict[str, Any]]:
        samples = value if isinstance(value, list) else []
        capped = []
        for sample in samples[:20]:
            text = str(sample)
            if len(text) > 500:
                sample = {"truncated": text[:470]}
            capped.append(sample if isinstance(sample, dict) else {"value": str(sample)[:500]})
        return capped


class ConsolidatedSubtopic(_StrictModel):
    """One document-level subtopic produced by consolidation."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=40)
    source_topic_ids: list[str] = Field(min_length=1)


class ConsolidatedMainTopic(_StrictModel):
    """One document-level main topic produced by consolidation."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=40)
    source_topic_ids: list[str] = Field(default_factory=list)
    subtopics: list[ConsolidatedSubtopic] = Field(default_factory=list)


class ConsolidatedHierarchy(_StrictModel):
    """Document-level main-topic to subtopic hierarchy."""

    main_topics: list[ConsolidatedMainTopic] = Field(default_factory=list)


class LearningTreeNode(_StrictModel):
    """Tree node returned by learning-graph retrieval."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: Literal["document", "topic", "subtopic", "concept", "objective"]
    summary: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    order_index: int | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    children: list["LearningTreeNode"] = Field(default_factory=list)
