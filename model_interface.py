"""Abstract interface for structured LLM generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelInterface(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        """Generate and parse a structured response."""
