from __future__ import annotations

import json

import thread_assigner
from thread_assigner import (
    assign_thread,
    compute_hours_between,
    compute_idf,
    cosine_similarity,
)


def _insert_event(conn, content="event", timestamp="2026-01-01T00:00:00+00:00"):
    cursor = conn.execute(
        "INSERT INTO events (source, content, timestamp, extraction_status, content_hash) "
        "VALUES ('test', ?, ?, 'completed', ?)",
        (content, timestamp, f"hash-{content}-{timestamp}"),
    )
    return cursor.lastrowid


def _insert_entity(conn, name):
    cursor = conn.execute(
        "INSERT INTO entities (canonical_name, normalized_form, created_at) "
        "VALUES (?, ?, '2026-01-01T00:00:00+00:00')",
        (name, name),
    )
    return cursor.lastrowid


def _link(conn, event_id, entity_id):
    conn.execute(
        "INSERT OR IGNORE INTO event_entity_edges (event_id, entity_id) VALUES (?, ?)",
        (event_id, entity_id),
    )


def test_compute_idf_single_entity_single_event(conn):
    event_id = _insert_event(conn)
    entity_id = _insert_entity(conn, "python")
    _link(conn, event_id, entity_id)

    weights = compute_idf(conn, [entity_id])

    assert weights[entity_id] == 1.0


def test_compute_hours_between():
    assert compute_hours_between(
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T02:30:00+00:00",
    ) == 2.5


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_assign_thread_no_candidates_creates_new(conn):
    event_id = _insert_event(conn)
    entity_id = _insert_entity(conn, "python")
    _link(conn, event_id, entity_id)

    thread_id = assign_thread(conn, event_id, [entity_id])

    assert thread_id == 1
    log = conn.execute("SELECT * FROM assignment_log").fetchone()
    assert log["decision_type"] == "new_thread"


def test_assign_thread_existing_thread(conn, monkeypatch):
    monkeypatch.setattr(thread_assigner, "ASSIGNMENT_THRESHOLD", 0.1)
    event1 = _insert_event(conn, "first", "2026-01-01T00:00:00+00:00")
    entity_id = _insert_entity(conn, "python")
    _link(conn, event1, entity_id)
    thread1 = assign_thread(conn, event1, [entity_id])

    event2 = _insert_event(conn, "second", "2026-01-01T00:30:00+00:00")
    _link(conn, event2, entity_id)
    thread2 = assign_thread(conn, event2, [entity_id])

    assert thread2 == thread1
    log = conn.execute(
        "SELECT * FROM assignment_log WHERE event_id = ?", (event2,)
    ).fetchone()
    assert log["decision_type"] == "existing_thread"
    scores = json.loads(log["candidate_scores"])
    assert str(thread1) in scores


def test_assign_thread_score_tie_selects_lowest_thread_id(conn, monkeypatch):
    monkeypatch.setattr(thread_assigner, "ASSIGNMENT_THRESHOLD", 0.1)
    event_a = _insert_event(conn, "a", "2026-01-01T00:00:00+00:00")
    event_b = _insert_event(conn, "b", "2026-01-01T00:00:00+00:00")
    entity_id = _insert_entity(conn, "python")
    _link(conn, event_a, entity_id)
    _link(conn, event_b, entity_id)
    thread_a = assign_thread(conn, event_a, [entity_id])
    # Force a separate second thread with same entity and timestamp.
    conn.execute("DELETE FROM event_thread_edges WHERE event_id = ?", (event_b,))
    conn.execute(
        "INSERT INTO threads (title, summary, created_at, updated_at) "
        "VALUES ('', '', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    thread_b = conn.execute("SELECT MAX(id) FROM threads").fetchone()[0]
    conn.execute(
        "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
        (event_b, thread_b),
    )

    event_c = _insert_event(conn, "c", "2026-01-01T00:00:00+00:00")
    _link(conn, event_c, entity_id)
    assigned = assign_thread(conn, event_c, [entity_id])

    assert assigned == min(thread_a, thread_b)
