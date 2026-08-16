
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


def _authentication_rejected(response: requests.Response) -> bool:
    return response.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN) or (
        response.status_code == HTTPStatus.BAD_REQUEST
        and "api key" in response.text.lower()
    )


class OpenAICompatProvider(ModelInterface):
    """Structured generation against any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        max_tokens_field: str = "max_tokens",
        auth_url: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._max_tokens_field = max_tokens_field
        self._auth_url = auth_url

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            self._max_tokens_field: MAX_OUTPUT_TOKENS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": strictify(response_model.model_json_schema()),
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self._url,
                    headers=headers,
                    json=body,
                    timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                    raise ModelUnavailableError(
                        f"Rate/credit limit (429): {response.text[:300]}"
                    )
                if 400 <= response.status_code < 500:
                    raise ModelUnavailableError(
                        f"Model request rejected ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return response_model.model_validate_json(strip_fences(content))
            except ModelUnavailableError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ModelUnavailableError(
            f"Model backend unavailable after {MAX_RETRIES} attempts: {last_error}"
        )

    def validate_key(self) -> tuple[bool, str]:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._auth_url:
            try:
                auth_response = requests.get(
                    self._auth_url,
                    headers=headers,
                    timeout=MODEL_VALIDATION_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                return False, f"could not reach {self._auth_url}: {exc}"
            if _authentication_rejected(auth_response):
                return False, f"authentication rejected ({auth_response.status_code})"
            if auth_response.status_code != 200:
                return False, (
                    f"authentication check failed ({auth_response.status_code}): "
                    f"{auth_response.text[:200]}"
                )

        try:
            response = requests.get(
                f"{self._base_url}/models",
                headers=headers,
                timeout=MODEL_VALIDATION_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            return False, f"could not reach {self._base_url}: {exc}"

        if response.status_code == HTTPStatus.OK:
            payload = response.json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            available = {
                str(row.get("id", "")).removeprefix("models/")
                for row in rows
                if isinstance(row, dict)
            }
            if not available:
                return False, "key works, but the provider returned no available models"
            expected = self._model.removeprefix("models/")
            if expected not in available:
                return False, f'key works, but model "{self._model}" is unavailable'
            return True, f'model "{self._model}" verified'
        if _authentication_rejected(response):
            return False, f"authentication rejected ({response.status_code})"
        return False, f"unexpected response ({response.status_code}): {response.text[:200]}"
