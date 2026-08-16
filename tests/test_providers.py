from __future__ import annotations

import pytest

from meniscus import config, providers
from meniscus.exceptions import EmbeddingDimensionMismatchError, MeniscusError
from meniscus.providers.local_embedding import LocalEmbeddingProvider


def test_get_embedding_model_none():
    assert providers.get_embedding_model("none") is None


def test_get_model_unknown_provider():
    with pytest.raises(MeniscusError):
        providers.get_model("missing")


def test_get_embedding_model_unknown_provider():
    with pytest.raises(MeniscusError):
        providers.get_embedding_model("missing")


def test_local_embedding_dimension_property():
    assert LocalEmbeddingProvider().dimension == 768


def test_get_embedding_model_dimension_mismatch(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSIONS", 384)
    with pytest.raises(EmbeddingDimensionMismatchError):
        providers.get_embedding_model("local")


def test_get_embedding_model_dimension_match(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSIONS", 768)
    provider = providers.get_embedding_model("local")
    assert isinstance(provider, LocalEmbeddingProvider)


def test_setup_exposes_every_supported_provider_without_model_profiles():
    from meniscus.cli.main import _PROVIDER_MENU

    assert [provider for provider, _description in _PROVIDER_MENU] == [
        "openrouter",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "xai",
        "huggingface",
        "ollama",
        "custom",
    ]
    assert not hasattr(config, "PROVIDER_MODEL_PROFILES")
    assert set(config.PROVIDER_DEFAULT_MODELS) == {
        provider for provider, _description in _PROVIDER_MENU if provider != "custom"
    }


def test_openai_uses_its_completion_token_parameter(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = providers.get_model("openai", "gpt-5-mini")

    assert model._max_tokens_field == "max_completion_tokens"


def test_compatible_providers_use_standard_max_tokens(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    model = providers.get_model("gemini", "gemini-3.5-flash-lite")

    assert model._max_tokens_field == "max_tokens"


def test_openrouter_validation_checks_the_authenticated_key_endpoint(monkeypatch):
    class Response:
        status_code = 401
        text = "unauthorized"

    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("meniscus.providers.openai_compat.requests.get", get)
    model = providers.build_model(
        "openrouter",
        "google/gemini-3.5-flash-lite",
        "invalid-key",
    )

    ok, detail = model.validate_key()

    assert not ok
    assert detail == "authentication rejected (401)"
    assert [url for url, _kwargs in calls] == ["https://openrouter.ai/api/v1/key"]


def test_xai_validation_recognizes_its_invalid_key_response(monkeypatch):
    class Response:
        status_code = 400
        text = '{"error":"Incorrect API key provided."}'

    monkeypatch.setattr(
        "meniscus.providers.openai_compat.requests.get",
        lambda *_args, **_kwargs: Response(),
    )
    model = providers.build_model("xai", "grok-4.3", "xyz")

    ok, detail = model.validate_key()

    assert not ok
    assert detail == "authentication rejected (400)"
