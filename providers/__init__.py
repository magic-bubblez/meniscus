"""Provider registry for LLM and embedding implementations."""

from __future__ import annotations

from collections.abc import Callable

from config import DEFAULT_MODEL_PROVIDER, EMBEDDING_DIMENSIONS, EMBEDDING_PROVIDER
from embedding_interface import EmbeddingInterface
from exceptions import EmbeddingDimensionMismatchError, MeniscusError
from model_interface import ModelInterface

_LLM_REGISTRY: dict[str, Callable[[], ModelInterface]] = {}
_EMBEDDING_REGISTRY: dict[str, Callable[[], EmbeddingInterface]] = {}


def _register_llm(name: str, factory: Callable[[], ModelInterface]) -> None:
    _LLM_REGISTRY[name] = factory


def _register_embedding(name: str, factory: Callable[[], EmbeddingInterface]) -> None:
    _EMBEDDING_REGISTRY[name] = factory


def _make_gemini_llm() -> ModelInterface:
    from providers.gemini import GeminiProvider

    return GeminiProvider()


def _make_openrouter_llm() -> ModelInterface:
    from providers.openrouter import OpenRouterProvider

    return OpenRouterProvider()


def _make_ollama_llm() -> ModelInterface:
    from providers.ollama import OllamaProvider

    return OllamaProvider()


def _make_gemini_embedding() -> EmbeddingInterface:
    from providers.gemini_embedding import GeminiEmbeddingProvider

    return GeminiEmbeddingProvider()


def _make_local_embedding() -> EmbeddingInterface:
    from providers.local_embedding import LocalEmbeddingProvider

    return LocalEmbeddingProvider()


_register_llm("gemini", _make_gemini_llm)
_register_llm("openrouter", _make_openrouter_llm)
_register_llm("ollama", _make_ollama_llm)
_register_embedding("gemini", _make_gemini_embedding)
_register_embedding("local", _make_local_embedding)


def get_model(provider_name: str | None = None) -> ModelInterface:
    """Instantiate an LLM provider."""

    name = provider_name or DEFAULT_MODEL_PROVIDER
    factory = _LLM_REGISTRY.get(name)
    if factory is None:
        raise MeniscusError(
            f"Unknown LLM provider: {name!r}. "
            f"Registered providers: {sorted(_LLM_REGISTRY)}"
        )
    return factory()


def get_embedding_model(provider_name: str | None = None) -> EmbeddingInterface | None:
    """Instantiate and validate an embedding provider."""

    name = provider_name or EMBEDDING_PROVIDER
    if name == "none":
        return None

    factory = _EMBEDDING_REGISTRY.get(name)
    if factory is None:
        raise MeniscusError(
            f"Unknown embedding provider: {name!r}. "
            f"Registered providers: {sorted(_EMBEDDING_REGISTRY)} or 'none'"
        )

    provider = factory()
    if provider.dimension != EMBEDDING_DIMENSIONS:
        raise EmbeddingDimensionMismatchError(
            f"Embedding provider {name!r} produces {provider.dimension}-dimensional "
            f"vectors, but EMBEDDING_DIMENSIONS is {EMBEDDING_DIMENSIONS}. "
            f"Set EMBEDDING_DIMENSIONS={provider.dimension} for this provider, "
            f"or select a provider whose native dimension is {EMBEDDING_DIMENSIONS}."
        )
    return provider
