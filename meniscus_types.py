"""Shared enum types for Meniscus."""

from __future__ import annotations

from enum import StrEnum


class ExtractionStatus(StrEnum):
    """Processing status for an event."""

    PENDING = "pending"
    COMPLETED = "completed"


class DecisionType(StrEnum):
    """Thread assignment decision type."""

    EXISTING_THREAD = "existing_thread"
    NEW_THREAD = "new_thread"
