"""Integration coverage for the REAL sqlite-vec KNN path.

The default `conn` fixture runs in EMBEDDING_PROVIDER="none" mode, so the actual
vec0 KNN query is never executed there. These tests build a genuine vec-enabled
database and exercise the KNN queries end to end -- catching things a mock never
would (e.g. vec0 rejecting a secondary `ORDER BY`).
"""

from __future__ import annotations

import struct

import pytest

import db


def _sqlite_vec_ok() -> bool:
    try:
        import sqlite3

        import sqlite_vec

        probe = sqlite3.connect(":memory:")
        try:
            probe.enable_load_extension(True)
            sqlite_vec.load(probe)
        finally:
            probe.close()
        return True
    except Exception:
        return False


requires_vec = pytest.mark.skipif(
    not _sqlite_vec_ok(), reason="sqlite-vec not installed/loadable"
)


def _seed(conn) -> None:
    vectors = {1: [1.0, 0.0, 0.0], 2: [0.9, 0.1, 0.0], 3: [0.0, 1.0, 0.0]}
    for event_id, vec in vectors.items():
        ts = f"2026-01-0{event_id}T00:00:00+00:00"
        conn.execute(
            "INSERT INTO events (source, content, timestamp, extraction_status, content_hash) "
            "VALUES ('t', ?, ?, 'completed', ?)",
            (f"content {event_id}", ts, f"hash-{event_id}"),
        )
        conn.execute(
            "INSERT INTO event_embeddings (event_id, embedding) VALUES (?, ?)",
            (event_id, struct.pack("3f", *vec)),
        )
        conn.execute(
            "INSERT INTO threads (title, summary, created_at, updated_at) VALUES ('', '', ?, ?)",
            (ts, ts),
        )
        conn.execute(
            "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
            (event_id, event_id),
        )
    conn.commit()


@pytest.fixture
def vec_conn(db_path, monkeypatch):
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(db, "EMBEDDING_DIMENSIONS", 3)
    connection = db.get_connection(db_path)
    db.init_db(connection)
    _seed(connection)
    try:
        yield connection
    finally:
        connection.close()


@requires_vec
def test_get_vector_candidates_runs_real_knn(vec_conn):
    from thread_assigner import get_vector_candidates

    # Query near event 1; its own id is excluded, the close event 2 is a candidate.
    candidates = get_vector_candidates(vec_conn, event_id=1, event_embedding=[1.0, 0.0, 0.0])
    assert isinstance(candidates, set)
    assert 2 in candidates          # thread of the near neighbor
    assert 1 not in candidates      # self excluded


@requires_vec
def test_vector_event_scores_runs_real_knn(vec_conn):
    from retrieval import _vector_event_scores

    scores = _vector_event_scores(vec_conn, [1.0, 0.0, 0.0], limit=5)
    ids = [event_id for event_id, _ in scores]
    assert ids, "real vec0 KNN must return results without OperationalError"
    assert ids[0] == 1              # itself is the nearest neighbor


@requires_vec
def test_search_with_real_embeddings(vec_conn):
    """End-to-end search through the vector door on a real vec0 table."""
    from retrieval import search

    class FakeEmbedding:
        dimension = 3

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    hits = search(vec_conn, text="anything", limit=5, embedding_model=FakeEmbedding())
    assert any("vector" in hit.matched_by for hit in hits)
    assert 1 in {hit.event_id for hit in hits}
