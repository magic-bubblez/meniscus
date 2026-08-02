from __future__ import annotations

from pathlib import Path

import pytest

from meniscus.event_intake import (
    _normalize_timestamp,
    compute_content_hash,
    ingest_directory,
    ingest_event,
    ingest_file,
)


def test_compute_content_hash_source_sensitive():
    assert compute_content_hash("same", "a") != compute_content_hash("same", "b")


def test_normalize_timestamp_variants():
    assert _normalize_timestamp("2026-07-15T12:00:00Z") == "2026-07-15T12:00:00+00:00"
    assert _normalize_timestamp("2026-07-15T12:00:00") == "2026-07-15T12:00:00+00:00"
    assert _normalize_timestamp("2026-07-15T17:30:00+05:30") == "2026-07-15T12:00:00+00:00"


def test_ingest_event_basic(conn):
    event_ids = ingest_event(
        conn,
        "hello",
        "cli",
        timestamp="2026-07-15T12:00:00Z",
    )
    assert len(event_ids) == 1
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_ids[0],)).fetchone()
    assert row["timestamp"] == "2026-07-15T12:00:00+00:00"
    assert row["extraction_status"] == "pending"


def test_ingest_event_dedup(conn):
    assert ingest_event(conn, "hello", "cli")
    assert ingest_event(conn, "hello", "cli") == []


def test_ingest_file_txt(conn, tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("file content")
    event_ids = ingest_file(conn, str(path))
    assert len(event_ids) == 1
    row = conn.execute("SELECT source FROM events WHERE id = ?", (event_ids[0],)).fetchone()
    assert row["source"] == "file:note.txt"


def test_ingest_file_unsupported_extension(conn, tmp_path: Path):
    path = tmp_path / "note.pdf"
    path.write_text("file content")
    with pytest.raises(ValueError):
        ingest_file(conn, str(path))


def test_ingest_directory(conn, tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "c.py").write_text("c")
    event_ids = ingest_directory(conn, str(tmp_path))
    assert len(event_ids) == 2
