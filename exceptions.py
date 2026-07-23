"""Meniscus exception hierarchy."""


class MeniscusError(Exception):
    """Base exception for all Meniscus errors."""


class ModelUnavailableError(MeniscusError):
    """Raised when an LLM or embedding model is persistently unavailable."""


class EmbeddingDimensionMismatchError(MeniscusError):
    """Raised when embedding dimensions disagree across config, provider, or DB."""


class EmbeddingBackendUnavailableError(MeniscusError):
    """Raised when a configured embedding provider lacks a vector backend."""


class DuplicateEventError(MeniscusError):
    """Raised internally when a duplicate event is detected."""


class DatabaseError(MeniscusError):
    """Raised for unrecoverable database setup or schema errors."""
