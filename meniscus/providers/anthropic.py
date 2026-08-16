
from __future__ import annotations

import time
from http import HTTPStatus
from typing import TypeVar

import requests
from pydantic import BaseModel

from meniscus.config import (
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MODEL_REQUEST_TIMEOUT_SECONDS,
    MODEL_VALIDATION_TIMEOUT_SECONDS,
    RETRY_BASE_DELAY_SECONDS,
)
from meniscus.exceptions import ModelUnavailableError
from meniscus.model_interface import ModelInterface
from meniscus.providers._schema import strictify, strip_fences

T = TypeVar("T", bound=BaseModel)

_URL = "https://api.anthropic.com/v1/messages"
_MODELS_URL = "https://api.anthropic.com/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(ModelInterface):
    """Structured generation via Claude's native API (guaranteed JSON schema output).

    Anthropic's OpenAI-compatibility shim ignores `response_format`/`strict`, so
    this talks to /v1/messages directly using `output_config.format` instead.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        body = {
            "model": self._model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": strictify(response_model.model_json_schema()),
                }
            },
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    _URL,
                    headers=headers,
                    json=body,
                    timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                    raise ModelUnavailableError(
                        f"Anthropic rate/credit limit (429): {response.text[:300]}"
                    )
                if 400 <= response.status_code < 500:
                    raise ModelUnavailableError(
                        f"Anthropic request rejected ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                response.raise_for_status()
                content = response.json()["content"][0]["text"]
                return response_model.model_validate_json(strip_fences(content))
            except ModelUnavailableError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ModelUnavailableError(
            f"Anthropic unavailable after {MAX_RETRIES} attempts: {last_error}"
        )

    def validate_key(self) -> tuple[bool, str]:
        """GET /v1/models — confirms auth with zero generation cost."""

        headers = {"x-api-key": self._api_key, "anthropic-version": _ANTHROPIC_VERSION}
        try:
            response = requests.get(
                _MODELS_URL,
                headers=headers,
                timeout=MODEL_VALIDATION_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            return False, f"could not reach Anthropic: {exc}"

        if response.status_code == HTTPStatus.OK:
            payload = response.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            available = {
                str(row.get("id", "")) for row in rows if isinstance(row, dict)
            }
            if not available:
                return False, "key works, but Anthropic returned no available models"
            if self._model not in available:
                return False, f'key works, but model "{self._model}" is unavailable'
            return True, f'model "{self._model}" verified'
        if response.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
            return False, f"authentication rejected ({response.status_code})"
        return False, f"unexpected response ({response.status_code}): {response.text[:200]}"
