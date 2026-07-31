from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import ExtractedEntity, ExtractionResult


def test_extracted_entity_defaults():
    entity = ExtractedEntity(name="python")
    assert entity.aliases == []


def test_extracted_entity_with_aliases():
    entity = ExtractedEntity(name="machine learning", aliases=["ML", "ml"])
    assert entity.aliases == ["ML", "ml"]


def test_extraction_result_from_json():
    result = ExtractionResult.model_validate_json(
        '{"entities": [{"name": "python", "aliases": ["py"]}]}'
    )
    assert result.entities[0].name == "python"


def test_extraction_result_missing_entities_field():
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate_json("{}")
