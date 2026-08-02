from __future__ import annotations

from meniscus import config


def test_all_config_constants_exist():
    names = [
        "ENTITY_CAP",
        "CHUNK_SIZE_WORDS",
        "VECTOR_CANDIDATE_K",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_DIMENSIONS",
        "HYBRID_ALPHA",
        "DEFAULT_MODEL_PROVIDER",
        "DB_PATH",
    ]
    for name in names:
        assert getattr(config, name) is not None


def test_config_value_ranges():
    assert config.ENTITY_CAP > 0
    assert config.CHUNK_SIZE_WORDS > 0
    assert config.VECTOR_CANDIDATE_K > 0
    assert config.EMBEDDING_DIMENSIONS > 0
    assert 0 <= config.HYBRID_ALPHA <= 1
