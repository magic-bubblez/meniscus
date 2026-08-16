from __future__ import annotations

import os

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
        "PROVIDER_DEFAULT_MODELS",
    ]
    for name in names:
        assert getattr(config, name) is not None


def test_config_value_ranges():
    assert config.ENTITY_CAP > 0
    assert config.CHUNK_SIZE_WORDS > 0
    assert config.VECTOR_CANDIDATE_K > 0
    assert config.EMBEDDING_DIMENSIONS > 0
    assert 0 <= config.HYBRID_ALPHA <= 1


def test_load_env_never_reads_an_unrelated_working_directory(tmp_path, monkeypatch):
    from meniscus import home

    variable = "MENISCUS_UNRELATED_TEST_KEY"
    unrelated = tmp_path / "project"
    memory_home = tmp_path / "memory"
    unrelated.mkdir()
    memory_home.mkdir()
    (unrelated / ".env").write_text(f"{variable}=wrong\n")
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(home, "MENISCUS_HOME", memory_home)
    monkeypatch.delenv(variable, raising=False)

    home.load_env()

    assert variable not in os.environ
