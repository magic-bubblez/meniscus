from __future__ import annotations

import pytest

from meniscus.config import ENTITY_CAP
from meniscus.entity_extractor import build_extraction_prompt, extract_entities
from meniscus.exceptions import ModelUnavailableError
from meniscus.models import ExtractedEntity, ExtractionResult


class FakeModel:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.prompt = None
        self.response_model = None

    def generate_structured(self, prompt, response_model):
        self.prompt = prompt
        self.response_model = response_model
        if self.error is not None:
            raise self.error
        return self.result


def _insert_event(conn):
    cursor = conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('cli', 'learning about linked lists', "
        "'2026-01-01T00:00:00+00:00', 'extract-event')"
    )
    return cursor.lastrowid


def test_build_extraction_prompt():
    prompt = build_extraction_prompt("cli", "learning about linked lists", "2026-01-01T00:00:00+00:00")
    assert "Source: cli" in prompt
    assert "Content: learning about linked lists" in prompt
    assert "Recorded: 2026-01-01T00:00:00+00:00" in prompt
    assert str(ENTITY_CAP) in prompt
    assert "[ENTITY_CAP]" not in prompt
    assert "Lean toward completeness" in prompt


def test_build_extraction_prompt_special_chars():
    prompt = build_extraction_prompt("cli", 'curly {braces}\n"quotes"', "2026-01-01T00:00:00+00:00")
    assert 'curly {braces}\n"quotes"' in prompt


def test_extract_entities_basic(conn):
    event_id = _insert_event(conn)
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="linked list"),
            ExtractedEntity(name="pointer"),
        ]
    )
    model = FakeModel(result=result)

    extracted = extract_entities(conn, event_id, model)

    assert extracted == result
    assert model.response_model is ExtractionResult
    assert "linked lists" in model.prompt


def test_extract_entities_truncates_to_cap(conn):
    event_id = _insert_event(conn)
    result = ExtractionResult(
        entities=[ExtractedEntity(name=f"entity {i}") for i in range(ENTITY_CAP + 10)]
    )

    extracted = extract_entities(conn, event_id, FakeModel(result=result))

    assert len(extracted.entities) == ENTITY_CAP
    assert extracted.entities[-1].name == f"entity {ENTITY_CAP - 1}"


def test_extract_entities_event_not_found(conn):
    with pytest.raises(ValueError):
        extract_entities(conn, 999, FakeModel(result=ExtractionResult(entities=[])))


def test_extract_entities_model_unavailable(conn):
    event_id = _insert_event(conn)
    with pytest.raises(ModelUnavailableError):
        extract_entities(
            conn,
            event_id,
            FakeModel(error=ModelUnavailableError("unavailable")),
        )
