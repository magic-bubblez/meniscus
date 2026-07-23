"""Gemini embedding provider."""

from __future__ import annotations

import os
import time

from config import EMBEDDING_DIMENSIONS, MAX_RETRIES, RETRY_BASE_DELAY_SECONDS
from embedding_interface import EmbeddingInterface
from exceptions import EmbeddingDimensionMismatchError, ModelUnavailableError

_GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-001"


class GeminiEmbeddingProvider(EmbeddingInterface):
    """Gemini embedding implementation."""

    def __init__(self) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ModelUnavailableError("GEMINI_API_KEY environment variable is not set.")
        self._client = genai.Client(api_key=api_key)

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSIONS

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in a single API call (the bulk-import win)."""

        if not texts:
            return []

        from google import genai as genai_module

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                result = self._client.models.embed_content(
                    model=_GEMINI_EMBEDDING_MODEL_NAME,
                    contents=texts,
                    config=genai_module.types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIMENSIONS,
                    ),
                )
                vectors = [list(e.values) for e in result.embeddings]
                if len(vectors) != len(texts):
                    raise ModelUnavailableError(
                        f"Expected {len(texts)} embeddings, got {len(vectors)}."
                    )
                for vector in vectors:
                    if len(vector) != EMBEDDING_DIMENSIONS:
                        raise EmbeddingDimensionMismatchError(
                            f"Gemini returned a {len(vector)}-dimensional vector, "
                            f"expected {EMBEDDING_DIMENSIONS}."
                        )
                return vectors
            except (EmbeddingDimensionMismatchError, ModelUnavailableError):
                raise
            except Exception as exc:
                # Fail fast on quota/rate errors so retries don't burn more quota.
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    raise ModelUnavailableError(
                        "Gemini embedding quota/rate limit hit (HTTP 429); not "
                        f"retried to preserve quota. Details: {exc}"
                    ) from exc
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ModelUnavailableError(
            "Gemini Embedding API unavailable after "
            f"{MAX_RETRIES} attempts: {last_error}"
        )
