"""Shared normalization for user-facing time-window bounds."""

from __future__ import annotations

import re
from datetime import datetime, time, timezone

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_time_bound(value: str, *, is_end: bool = False) -> str:
    """Normalize an ISO date or timestamp to a canonical UTC timestamp.

    A date-only lower bound means the start of that day. A date-only upper
    bound means the end of that day, so `--until 2026-01-15` includes events
    recorded throughout January 15.
    """

    cleaned = value.strip()
    if _DATE_ONLY_RE.fullmatch(cleaned):
        parsed_date = datetime.fromisoformat(cleaned).date()
        boundary_time = time.max if is_end else time.min
        return datetime.combine(
            parsed_date,
            boundary_time,
            tzinfo=timezone.utc,
        ).isoformat()

    parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def normalize_time_window(
    start: str | None,
    end: str | None,
) -> tuple[str | None, str | None]:
    """Normalize optional bounds and reject an inverted window."""

    normalized_start = (
        normalize_time_bound(start, is_end=False) if start is not None else None
    )
    normalized_end = (
        normalize_time_bound(end, is_end=True) if end is not None else None
    )
    if (
        normalized_start is not None
        and normalized_end is not None
        and normalized_start > normalized_end
    ):
        raise ValueError("start must be before or equal to end")
    return normalized_start, normalized_end

