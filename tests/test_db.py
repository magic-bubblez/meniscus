from __future__ import annotations

import sqlite3

import pytest

import db
from exceptions import (
    EmbeddingBackendUnavailableError,
    EmbeddingDimensionMismatchError,
)


def test_get_db_path_env_override(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "meniscus.db"
    monkeypatch.setenv("MENISCUS_DB_PATH", str(path))

    resolved = db.get_db_path()

    assert resolved == path.resolve()
    assert resolved.parent.exists()


def test_get_connection_wal_mode(db_path, monkeypatch):
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    conn = db.get_connection(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_get_connection_foreign_keys_on(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_get_connection_row_factory(conn):
    conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('test', 'content', '2026-01-01T00:00:00+00:00', 'hash-row')"
    )
    row = conn.execute("SELECT id FROM events WHERE content_hash = 'hash-row'").fetchone()
    assert row["id"] == 1


def test_init_db_creates_all_tables(conn):
    expected = {
        "events",
        "entities",
        "entity_aliases",
        "event_entity_edges",
        "threads",
        "event_thread_edges",
        "assignment_log",
        "events_fts",
        "meta",
    }
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert expected <= names


def test_init_db_creates_all_indexes(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    names = {row["name"] for row in rows}
    assert {
        "idx_events_extraction_status",
        "idx_events_content_hash",
        "idx_events_timestamp",
        "idx_events_source_id",
        "idx_entities_normalized_form",
        "idx_entity_aliases_normalized_form",
        "idx_event_entity_edges_entity_id",
        "idx_event_thread_edges_thread_id",
    } <= names


def test_init_db_idempotent(conn):
    db.init_db(conn)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 0


def test_init_db_fts_trigger_fires_on_insert(conn):
    conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('test', 'linked lists', '2026-01-01T00:00:00+00:00', 'hash-fts')"
    )
    row = conn.execute(
        "SELECT rowid FROM events_fts WHERE events_fts MATCH 'linked'"
    ).fetchone()
    assert row is not None


def test_init_db_fts_trigger_fires_on_delete(conn):
    conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('test', 'delete marker', '2026-01-01T00:00:00+00:00', 'hash-delete')"
    )
    conn.execute("DELETE FROM events WHERE content_hash = 'hash-delete'")
    row = conn.execute(
        "SELECT rowid FROM events_fts WHERE events_fts MATCH 'marker'"
    ).fetchone()
    assert row is None


def test_transactional_commits_on_success(conn):
    with db.transactional(conn) as txn:
        txn.execute(
            "INSERT INTO entities (canonical_name, normalized_form, created_at) "
            "VALUES ('Python', 'python', '2026-01-01T00:00:00+00:00')"
        )
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1


def test_transactional_rolls_back_on_exception(conn):
    with pytest.raises(ValueError):
        with db.transactional(conn) as txn:
            txn.execute(
                "INSERT INTO entities (canonical_name, normalized_form, created_at) "
                "VALUES ('Python', 'python', '2026-01-01T00:00:00+00:00')"
            )
            raise ValueError("boom")
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0


def test_unique_constraints(conn):
    conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('test', 'a', '2026-01-01T00:00:00+00:00', 'same')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO events (source, content, timestamp, content_hash) "
            "VALUES ('test', 'b', '2026-01-01T00:00:00+00:00', 'same')"
        )


def test_event_thread_edges_single_membership(conn):
    conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('test', 'a', '2026-01-01T00:00:00+00:00', 'event')"
    )
    conn.execute(
        "INSERT INTO threads (created_at, updated_at) "
        "VALUES ('2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO threads (created_at, updated_at) "
        "VALUES ('2026-01-02T00:00:00+00:00', '2026-01-02T00:00:00+00:00')"
    )
    conn.execute("INSERT INTO event_thread_edges (event_id, thread_id) VALUES (1, 1)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO event_thread_edges (event_id, thread_id) VALUES (1, 2)")


def test_init_db_records_configured_dimension(conn):
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'embedding_dimensions'"
    ).fetchone()
    assert row["value"] == str(db.EMBEDDING_DIMENSIONS)


def test_init_db_dimension_mismatch_raises(db_path, monkeypatch):
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    monkeypatch.setattr(db, "EMBEDDING_DIMENSIONS", 768)
    conn = db.get_connection(db_path)
    db.init_db(conn)
    conn.close()

    monkeypatch.setattr(db, "EMBEDDING_DIMENSIONS", 384)
    conn = db.get_connection(db_path)
    try:
        with pytest.raises(EmbeddingDimensionMismatchError):
            db.init_db(conn)
    finally:
        conn.close()


def test_init_db_raises_when_provider_configured_and_sqlite_vec_missing(
    db_path, monkeypatch
):
    import sys

    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "gemini")
    # Genuinely simulate a missing extension: make `import sqlite_vec` fail.
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)
    conn = db.get_connection(db_path)
    try:
        with pytest.raises(EmbeddingBackendUnavailableError):
            db.init_db(conn)
    finally:
        conn.close()


def test_init_db_skips_vec0_when_provider_none(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'event_embeddings'"
    ).fetchone()
    assert row is None
