from __future__ import annotations

import pytest

from meniscus.time_bounds import normalize_time_bound, normalize_time_window


def test_date_only_bounds_cover_whole_day():
    assert normalize_time_bound("2026-01-15") == "2026-01-15T00:00:00+00:00"
    assert normalize_time_bound(
        "2026-01-15",
        is_end=True,
    ) == "2026-01-15T23:59:59.999999+00:00"


def test_timestamp_bound_is_normalized_to_utc():
    assert normalize_time_bound(
        "2026-01-15T05:30:00+05:30"
    ) == "2026-01-15T00:00:00+00:00"


def test_inverted_window_rejected():
    with pytest.raises(ValueError, match="start must be before"):
        normalize_time_window("2026-01-16", "2026-01-15")
