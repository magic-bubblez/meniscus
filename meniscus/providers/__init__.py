"""Provider registry for LLM and embedding implementations."""

from __future__ import annotations

import os
from collections.abc import Callable

from meniscus import config
from meniscus.embedding_interface import EmbeddingInterface
from meniscus.exceptions import (
    EmbeddingDimensionMismatchError,
    MeniscusError,
    ModelUnavailableError,
)
from meniscus.model_interface import ModelInterface

_EMBEDDING_REGISTRY: dict[str, Callable[[], EmbeddingInterface]] = {}


def _register_embedding(name: str, factory: Callable[[], EmbeddingInterface]) -> None:
    _EMBEDDING_REGISTRY[name] = factory


def _make_local_embedding() -> EmbeddingInterface:
    from meniscus.providers.local_embedding import LocalEmbeddingProvider

    return LocalEmbeddingProvider()


_register_embedding("local", _make_local_embedding)


def build_model(
    provider_name: str,
    model: str,
    api_key: str | None,
    base_url: str | None = None,
) -> ModelInterface:
    """Build a provider from explicit setup-time credentials without saving them."""

    from meniscus.providers.openai_compat import OpenAICompatProvider

    if provider_name == "anthropic":
        from meniscus.providers.anthropic import AnthropicProvider

        if not api_key:
            raise ModelUnavailableError("ANTHROPIC_API_KEY is not set.")
        return AnthropicProvider(api_key=api_key, model=model)

    if provider_name == config.CUSTOM_PROVIDER:
        resolved_base_url = base_url
        auth_url = None
    else:
        preset = config.PROVIDER_PRESETS.get(provider_name)
        if preset is None:
            raise MeniscusError(
                f"Unknown provider: {provider_name!r}. "
                f"Known providers: {sorted(config.PROVIDER_PRESETS)} or 'custom'."
            )
        resolved_base_url = preset.base_url
        auth_url = preset.auth_url
        max_tokens_field = preset.max_tokens_field

    if not resolved_base_url:
        raise MeniscusError(
            f"Provider {provider_name!r} does not expose an OpenAI-compatible URL."
        )

    if provider_name == config.CUSTOM_PROVIDER:
        max_tokens_field = "max_tokens"
    return OpenAICompatProvider(
        base_url=resolved_base_url,
        api_key=api_key,
        model=model,
        max_tokens_field=max_tokens_field,
        auth_url=auth_url,
    )


def get_model(provider_name: str | None = None, model: str | None = None) -> ModelInterface:
    """Instantiate an LLM provider: resolve base_url + key from the preset/custom config.

    Reads `config.*` at call time (not import time) so a config.toml rewritten
    mid-process (e.g. by `men init`) takes effect immediately.

    `model` is an optional per-call override; ordinary commands use the single
    model selected by the user during setup.
    """

    name = provider_name or config.DEFAULT_MODEL_PROVIDER
    if not name:
        raise ModelUnavailableError(
            "No LLM provider configured. Run `men init` to set one up."
        )

    resolved_model = model or config.MODEL
    if not resolved_model:
        raise ModelUnavailableError(
            "No model configured. Run `men config set` (or `men init`) to choose one."
        )

    if name == config.CUSTOM_PROVIDER:
        if not config.CUSTOM_BASE_URL:
            raise MeniscusError(
                'provider = "custom" requires [model] base_url in ~/.meniscus/config.toml'
            )
        base_url = config.CUSTOM_BASE_URL
        api_key_env = config.CUSTOM_API_KEY_ENV
    else:
        preset = config.PROVIDER_PRESETS.get(name)
        if preset is None:
            raise MeniscusError(
                f"Unknown provider: {name!r}. "
                f"Known providers: {sorted(config.PROVIDER_PRESETS)} or 'custom'."
            )
        base_url = None
        api_key_env = preset.api_key_env

    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise ModelUnavailableError(f"{api_key_env} environment variable is not set.")

    return build_model(
        provider_name=name,
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
    )


def get_embedding_model(provider_name: str | None = None) -> EmbeddingInterface | None:
    """Instantiate and validate an embedding provider."""

    name = provider_name or config.EMBEDDING_PROVIDER
    if name == "none":
        return None

    factory = _EMBEDDING_REGISTRY.get(name)
    if factory is None:
        raise MeniscusError(
            f"Unknown embedding provider: {name!r}. "
            f"Registered providers: {sorted(_EMBEDDING_REGISTRY)} or 'none'"
        )

    provider = factory()
    if provider.dimension != config.EMBEDDING_DIMENSIONS:
        raise EmbeddingDimensionMismatchError(
            f"Embedding provider {name!r} produces {provider.dimension}-dimensional "
            f"vectors, but EMBEDDING_DIMENSIONS is {config.EMBEDDING_DIMENSIONS}. "
            f"Set EMBEDDING_DIMENSIONS={provider.dimension} for this provider, "
            f"or select a provider whose native dimension is {config.EMBEDDING_DIMENSIONS}."
        )
    return provider


def get_model_if_available() -> ModelInterface | None:
    try:
        return get_model()
    except (ModelUnavailableError, ImportError):
        return None


def get_embedding_if_available() -> EmbeddingInterface | None:
    try:
        return get_embedding_model()
    except (ModelUnavailableError, ImportError):
        return None
