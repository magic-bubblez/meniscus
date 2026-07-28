
from __future__ import annotations

import json
import os
import time
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from config import (
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    OPENROUTER_MODEL,
    RETRY_BASE_DELAY_SECONDS,
)
from exceptions import ModelUnavailableError
from model_interface import ModelInterface

T = TypeVar("T", bound=BaseModel)

_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(ModelInterface):
    """Structured generation via OpenRouter's OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ModelUnavailableError("OPENROUTER_API_KEY environment variable is not set.")
        self._api_key = api_key
        self._model = model or OPENROUTER_MODEL

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # Bound the reservation: without this OpenRouter holds the model's
            # entire output budget per request and 402s on a low balance.
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _strictify(response_model.model_json_schema()),
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(_URL, headers=headers, json=body, timeout=120)
                if response.status_code == 429:
                    # Rate/credit limit — do not retry (it won't recover immediately).
                    raise ModelUnavailableError(
                        f"OpenRouter rate/credit limit (429): {response.text[:300]}"
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return response_model.model_validate_json(_strip_fences(content))
            except ModelUnavailableError:
                raise
            except (ValidationError, Exception) as exc:
                # Retryable: network/5xx, and ALSO a truncated/parse failure —
                # OpenRouter occasionally returns an incomplete body, which a
                # retry resolves. Only give up after MAX_RETRIES.
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ModelUnavailableError(
            f"OpenRouter unavailable after {MAX_RETRIES} attempts: {last_error}"
        )


def _strictify(schema: Any) -> Any:
    """Make a Pydantic JSON schema satisfy OpenAI/OpenRouter strict mode.

    Strict mode requires every object to set additionalProperties=false and to
    list all properties in `required`; it also rejects the `default` keyword.
    """

    if isinstance(schema, dict):
        cleaned = {
            key: _strictify(value)
            for key, value in schema.items()
            if key != "default"
        }
        if cleaned.get("type") == "object" or "properties" in cleaned:
            properties = cleaned.get("properties", {})
            cleaned["additionalProperties"] = False
            cleaned["required"] = list(properties.keys())
        return cleaned
    if isinstance(schema, list):
        return [_strictify(item) for item in schema]
    return schema


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` fences some models wrap around JSON."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()
