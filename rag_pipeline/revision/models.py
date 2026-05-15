"""Pydantic models for revision generation (flashcards + mock tests)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeneratedFlashcard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    front: str = Field(min_length=1, max_length=500)
    back: str = Field(min_length=1, max_length=2000)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("front", "back")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class FlashcardBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cards: list[GeneratedFlashcard] = Field(default_factory=list)


class GeneratedMockQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1, max_length=1000)
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str = Field(default="", max_length=2000)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("prompt", "explanation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, value: list[str]) -> list[str]:
        cleaned = [str(c).strip() for c in value]
        if any(not c for c in cleaned):
            raise ValueError("choices must be non-empty")
        return cleaned


class MockTestBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    questions: list[GeneratedMockQuestion] = Field(default_factory=list)
