"""Shared enum types for Meniscus."""

from __future__ import annotations

from enum import StrEnum


class ExtractionStatus(StrEnum):
    """Processing status for an event."""

    PENDING = "pending"
    COMPLETED = "completed"
