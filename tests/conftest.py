from __future__ import annotations

from pathlib import Path

import pytest

from meniscus import config, db


@pytest.fixture(autouse=True)
def _default_model_config(monkeypatch: pytest.MonkeyPatch):
    """Tests that build a FakeModel directly still need provenance fields set --
    production ships no default, so tests supply their own, like a real config.toml would."""
    monkeypatch.setattr(config, "DEFAULT_MODEL_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "MODEL", "test-model")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def conn(db_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    connection = db.get_connection(db_path)
    db.init_db(connection)
    try:
        yield connection
    finally:
        connection.close()
