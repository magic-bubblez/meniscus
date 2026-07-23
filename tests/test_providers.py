from __future__ import annotations

import pytest

import providers
from exceptions import EmbeddingDimensionMismatchError, MeniscusError
from providers.local_embedding import LocalEmbeddingProvider


def test_get_embedding_model_none():
    assert providers.get_embedding_model("none") is None


def test_get_model_unknown_provider():
    with pytest.raises(MeniscusError):
        providers.get_model("missing")


def test_get_embedding_model_unknown_provider():
    with pytest.raises(MeniscusError):
        providers.get_embedding_model("missing")


def test_local_embedding_dimension_property():
    assert LocalEmbeddingProvider().dimension == 384


def test_get_embedding_model_dimension_mismatch(monkeypatch):
    monkeypatch.setattr(providers, "EMBEDDING_DIMENSIONS", 768)
    with pytest.raises(EmbeddingDimensionMismatchError):
        providers.get_embedding_model("local")


def test_get_embedding_model_dimension_match(monkeypatch):
    monkeypatch.setattr(providers, "EMBEDDING_DIMENSIONS", 384)
    provider = providers.get_embedding_model("local")
    assert isinstance(provider, LocalEmbeddingProvider)
