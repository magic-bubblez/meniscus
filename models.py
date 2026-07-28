"""Pydantic response models used for LLM structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """One entity extracted from an event."""

    name: str
    aliases: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Entity + fact extraction response shape (one call produces both)."""

    entities: list[ExtractedEntity]
    facts: list[str] = Field(default_factory=list)


class ThreadSummary(BaseModel):
    """Thread summarization response shape."""

    title: str
    summary: str
