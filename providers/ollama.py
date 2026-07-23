"""Ollama LLM provider (local, OpenAI-compatible chat completions).

Talks to a locally-running Ollama server. No API key, no network egress, no
per-token cost. Reuses the same strict-JSON-schema structured-output path as the
OpenRouter provider, so extraction/summarization get the same guaranteed shape.
"""

from __future__ import annotations

import time
from typing import TypeVar

import requests
from pydantic import BaseModel

from config import (
    MAX_RETRIES,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RETRY_BASE_DELAY_SECONDS,
)
from exceptions import ModelUnavailableError
from model_interface import ModelInterface
from providers.openrouter import _strictify, _strip_fences

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(ModelInterface):
    """Structured generation via a local Ollama OpenAI-compatible endpoint."""

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        body = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _strictify(response_model.model_json_schema()),
                },
            },
        }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                # Local inference is slower than a hosted API, so allow a generous
                # timeout (a cold 7B on CPU/Metal can take tens of seconds).
                response = requests.post(OLLAMA_URL, json=body, timeout=300)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return response_model.model_validate_json(_strip_fences(content))
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ModelUnavailableError(
            f"Ollama unavailable after {MAX_RETRIES} attempts (is the Ollama "
            f"server running and `{OLLAMA_MODEL}` pulled?): {last_error}"
        )
