"""Shared JSON-schema helpers for provider implementations."""

from __future__ import annotations

from typing import Any


def strictify(schema: Any) -> Any:

    if isinstance(schema, dict):
        cleaned = {
            key: strictify(value)
            for key, value in schema.items()
            if key != "default"
        }
        if cleaned.get("type") == "object" or "properties" in cleaned:
            properties = cleaned.get("properties", {})
            cleaned["additionalProperties"] = False
            cleaned["required"] = list(properties.keys())
        return cleaned
    if isinstance(schema, list):
        return [strictify(item) for item in schema]
    return schema


def strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences some models wrap around JSON."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()
