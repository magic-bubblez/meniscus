"""Pydantic response models used for LLM structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """One entity extracted from an event."""

    name: str
    aliases: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Entity extraction response shape."""

    entities: list[ExtractedEntity]


class ThreadSummary(BaseModel):
    """Thread summarization response shape."""

    title: str
    summary: str
