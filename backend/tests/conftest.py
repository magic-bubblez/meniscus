import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest

from app import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    # Point every test at its own throwaway SQLite file instead of
    # backend/data/meniscus.db, which the dev server and seed data use.
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_meniscus.db")
    db.initialize_db()
    yield
