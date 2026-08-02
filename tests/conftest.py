from __future__ import annotations

from pathlib import Path

import pytest

from meniscus import db


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
