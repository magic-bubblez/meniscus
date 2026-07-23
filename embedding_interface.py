"""Abstract interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingInterface(ABC):
    """Abstract base class for embedding model providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Native vector dimension produced by this provider."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Compute an embedding vector for text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts at once.

        Default implementation loops `embed`; providers that support a native
        batch endpoint (e.g. Gemini) override this to collapse N calls into a
        handful, which is the big win for bulk import.
        """

        return [self.embed(text) for text in texts]
