
from __future__ import annotations

import os
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from config import ANTHROPIC_MODEL, MAX_OUTPUT_TOKENS, MAX_RETRIES
from exceptions import ModelUnavailableError
from model_interface import ModelInterface

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(ModelInterface):
    """Structured generation via the Anthropic Messages API."""

    def __init__(self, model: str | None = None) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelUnavailableError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        # The SDK already retries 429s and 5xx with exponential backoff, so the
        # provider does not need its own retry loop.
        self._client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        self._model = model or ANTHROPIC_MODEL

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                output_format=response_model,
            )
        except anthropic.APIStatusError as exc:
            raise ModelUnavailableError(
                f"Anthropic API error ({exc.status_code}): {exc}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ModelUnavailableError(f"Anthropic connection error: {exc}") from exc

        parsed = response.parsed_output
        if parsed is None:
            # A refusal or a truncated response yields no parsed object; treat it
            # like any other unavailability so the caller leaves the event pending.
            raise ModelUnavailableError(
                f"Anthropic returned no parsed output (stop_reason={response.stop_reason})."
            )
        return parsed
