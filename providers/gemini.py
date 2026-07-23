"""Gemini LLM provider."""

from __future__ import annotations

import os
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from config import GEMINI_MODEL, MAX_RETRIES, RETRY_BASE_DELAY_SECONDS
from exceptions import ModelUnavailableError
from model_interface import ModelInterface

T = TypeVar("T", bound=BaseModel)

_GEMINI_MODEL_NAME = GEMINI_MODEL


class GeminiProvider(ModelInterface):
    """Gemini implementation of structured generation."""

    def __init__(self) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ModelUnavailableError("GEMINI_API_KEY environment variable is not set.")
        self._client = genai.Client(api_key=api_key)

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        from google import genai as genai_module

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=_GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=genai_module.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_model,
                        temperature=0,
                    ),
                )
                try:
                    return response_model.model_validate_json(response.text)
                except ValidationError as exc:
                    raise ModelUnavailableError(
                        f"LLM output failed schema validation: {exc}"
                    ) from exc
            except ModelUnavailableError:
                raise
            except Exception as exc:
                # A quota/rate error (429) will NOT recover on an immediate retry,
                # and each retry burns another request against the quota. Fail
                # fast with a clear message instead of tripling the damage.
                if _is_quota_error(exc):
                    raise ModelUnavailableError(_quota_message(exc)) from exc
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ModelUnavailableError(
            f"Gemini API unavailable after {MAX_RETRIES} attempts: {last_error}"
        )


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _quota_message(exc: Exception) -> str:
    return (
        "Gemini quota/rate limit hit (HTTP 429) — no request was retried, to avoid "
        "burning more of your quota. Free tiers are small (e.g. gemini-2.5-flash "
        "is ~20 requests/day). Wait for the limit to reset, then run `men process` "
        "to finish the pending events, or switch to a higher-quota model/plan. "
        f"Details: {exc}"
    )
