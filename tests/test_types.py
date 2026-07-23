from __future__ import annotations

from meniscus_types import DecisionType, ExtractionStatus


def test_extraction_status_values():
    assert ExtractionStatus.PENDING == "pending"
    assert ExtractionStatus.COMPLETED == "completed"


def test_decision_type_values():
    assert DecisionType.EXISTING_THREAD == "existing_thread"
    assert DecisionType.NEW_THREAD == "new_thread"
