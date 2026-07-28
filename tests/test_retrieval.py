from __future__ import annotations

import json

from click.testing import CliRunner

import db
from cli.main import cli
from retrieval import (
    _sanitize_fts_query,
    get_thread_detail,
    group_into_episodes,
    recent,
    search,
)


class FakeEmbedding:
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def _add_event(
    conn,
    content: str,
    timestamp: str,
    *,
    source: str = "test",
    entity_names: list[str] | None = None,
    thread_id: int | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO events (source, content, timestamp, extraction_status, content_hash) "
        "VALUES (?, ?, ?, 'completed', ?)",
        (source, content, timestamp, f"{source}:{content}:{timestamp}"),
    )
    event_id = int(cursor.lastrowid)
    for name in entity_names or []:
        normalized = " ".join(sorted(name.lower().split()))
        row = conn.execute(
            "SELECT id FROM entities WHERE normalized_form = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            entity_id = int(
                conn.execute(
                    "INSERT INTO entities (canonical_name, normalized_form, created_at) "
                    "VALUES (?, ?, ?)",
                    (name, normalized, timestamp),
                ).lastrowid
            )
        else:
            entity_id = int(row["id"])
        conn.execute(
            "INSERT INTO event_entity_edges (event_id, entity_id) VALUES (?, ?)",
            (event_id, entity_id),
        )
    if thread_id is not None:
        conn.execute(
            "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
            (event_id, thread_id),
        )
    return event_id


def _add_fact(conn, event_id: int, text: str, entity_names: list[str] | None = None) -> int:
    extraction_id = int(
        conn.execute(
            "INSERT INTO extractions (event_id, provider, model, prompt_version, extracted_at) "
            "VALUES (?, 'p', 'm', 'v', '2026-01-01')",
            (event_id,),
        ).lastrowid
    )
    fact_id = int(
        conn.execute(
            "INSERT INTO facts (event_id, extraction_id, text, position, created_at) "
            "VALUES (?, ?, ?, 0, '2026-01-01')",
            (event_id, extraction_id, text),
        ).lastrowid
    )
    for name in entity_names or []:
        normalized = " ".join(sorted(name.lower().split()))
        row = conn.execute(
            "SELECT id FROM entities WHERE normalized_form = ?", (normalized,)
        ).fetchone()
        entity_id = (
            int(row["id"])
            if row is not None
            else int(
                conn.execute(
                    "INSERT INTO entities (canonical_name, normalized_form, created_at) "
                    "VALUES (?, ?, '2026-01-01')",
                    (name, normalized),
                ).lastrowid
            )
        )
        conn.execute(
            "INSERT INTO fact_entity_edges (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, entity_id),
        )
    return fact_id


def _add_thread(
    conn,
    title: str,
    created_at: str,
    updated_at: str,
    *,
    summary: str = "",
) -> int:
    return int(
        conn.execute(
            "INSERT INTO threads (title, summary, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (title, summary, created_at, updated_at),
        ).lastrowid
    )


def test_search_entity_door(conn):
    thread_id = _add_thread(conn, "Auth", "2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00")
    event_id = _add_event(
        conn,
        "debugged jwt refresh",
        "2026-01-01T01:00:00+00:00",
        entity_names=["auth"],
        thread_id=thread_id,
    )
    _add_event(conn, "read sqlite notes", "2026-01-01T02:00:00+00:00")

    hits = search(conn, entity="auth")

    assert [hit.event_id for hit in hits] == [event_id]
    assert hits[0].matched_by == ["entity"]


def test_search_text_door_sanitized(conn):
    thread_id = _add_thread(conn, "Operators", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    event_id = _add_event(
        conn,
        "C++ operator overload notes",
        "2026-01-01T00:00:00+00:00",
        thread_id=thread_id,
    )

    hits = search(conn, text='"operator" -*')

    assert [hit.event_id for hit in hits] == [event_id]
    assert hits[0].matched_by == ["text"]
    assert _sanitize_fts_query('"foo -bar AND baz*') == '"foo" "bar" "AND" "baz"'


def test_search_vector_door(conn, monkeypatch):
    thread_id = _add_thread(conn, "Vectors", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    event_id = _add_event(conn, "semantic neighbor", "2026-01-01T00:00:00+00:00", thread_id=thread_id)

    def fake_vector_scores(_conn, query_embedding, limit):
        assert query_embedding == [1.0, 0.0, 0.0]
        return [(event_id, 0.9)]

    monkeypatch.setattr("retrieval._vector_event_scores", fake_vector_scores)

    hits = search(conn, text="meaning", embedding_model=FakeEmbedding())

    assert [hit.event_id for hit in hits] == [event_id]
    assert hits[0].matched_by == ["vector"]
    assert hits[0].scores["vector_score"] == 0.9


def test_search_temporal_filter(conn):
    old_thread = _add_thread(conn, "Old Auth", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    new_thread = _add_thread(conn, "New Auth", "2026-01-15T00:00:00+00:00", "2026-01-15T00:00:00+00:00")
    _add_event(conn, "old auth", "2026-01-01T00:00:00+00:00", entity_names=["auth"], thread_id=old_thread)
    new_event = _add_event(
        conn,
        "new auth",
        "2026-01-15T00:00:00+00:00",
        entity_names=["auth"],
        thread_id=new_thread,
    )

    hits = search(
        conn,
        entity="auth",
        start="2026-01-10T00:00:00+00:00",
        end="2026-01-20T00:00:00+00:00",
    )

    assert [hit.event_id for hit in hits] == [new_event]


def test_search_date_only_end_includes_whole_day(conn):
    thread_id = _add_thread(
        conn,
        "Auth",
        "2026-01-15T00:00:00+00:00",
        "2026-01-15T23:30:00+00:00",
    )
    event_id = _add_event(
        conn,
        "late auth work",
        "2026-01-15T23:30:00+00:00",
        entity_names=["auth"],
        thread_id=thread_id,
    )

    hits = search(conn, entity="auth", start="2026-01-15", end="2026-01-15")

    assert [hit.event_id for hit in hits] == [event_id]


def test_search_dedup_and_matched_by(conn):
    thread_id = _add_thread(conn, "Auth", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    event_id = _add_event(
        conn,
        "auth token rotation",
        "2026-01-01T00:00:00+00:00",
        entity_names=["auth"],
        thread_id=thread_id,
    )

    hits = search(conn, text="auth", entity="auth")

    assert [hit.event_id for hit in hits] == [event_id]
    assert hits[0].matched_by == ["entity", "text"]


def test_search_limit(conn):
    thread_id = _add_thread(conn, "Auth", "2026-01-01T00:00:00+00:00", "2026-01-03T00:00:00+00:00")
    for index in range(3):
        _add_event(
            conn,
            f"auth event {index}",
            f"2026-01-0{index + 1}T00:00:00+00:00",
            entity_names=["auth"],
            thread_id=thread_id,
        )

    hits = search(conn, entity="auth", limit=2)

    assert len(hits) == 2


def test_get_thread_detail_ordering(conn):
    thread_id = _add_thread(conn, "DB", "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00")
    second = _add_event(conn, "second", "2026-01-02T00:00:00+00:00", entity_names=["sqlite"], thread_id=thread_id)
    first = _add_event(conn, "first", "2026-01-01T00:00:00+00:00", entity_names=["database"], thread_id=thread_id)

    episode = get_thread_detail(conn, thread_id)

    assert episode is not None
    assert [event.event_id for event in episode.events] == [first, second]
    assert episode.events[0].entities == ["database"]


def test_get_thread_detail_missing(conn):
    assert get_thread_detail(conn, 999) is None


def test_recent_window(conn):
    old_thread = _add_thread(conn, "Old", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    new_thread = _add_thread(conn, "New", "2026-01-10T00:00:00+00:00", "2026-01-10T00:00:00+00:00")
    _add_event(conn, "old", "2026-01-01T00:00:00+00:00", thread_id=old_thread)
    _add_event(conn, "new", "2026-01-10T00:00:00+00:00", thread_id=new_thread)

    episodes = recent(
        conn,
        start="2026-01-05T00:00:00+00:00",
        end="2026-01-15T00:00:00+00:00",
    )

    assert [episode.thread_id for episode in episodes] == [new_thread]


def test_recent_no_window(conn):
    first = _add_thread(conn, "First", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    second = _add_thread(conn, "Second", "2026-01-02T00:00:00+00:00", "2026-01-02T00:00:00+00:00")

    episodes = recent(conn, limit=1)

    assert [episode.thread_id for episode in episodes] == [second]
    assert first != second


def test_group_into_episodes_full_episode(conn):
    first_thread = _add_thread(conn, "One", "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00")
    second_thread = _add_thread(conn, "Two", "2026-01-03T00:00:00+00:00", "2026-01-03T00:00:00+00:00")
    match_event = _add_event(conn, "matching auth", "2026-01-04T00:00:00+00:00", entity_names=["auth"], thread_id=first_thread)
    other_event = _add_event(conn, "same episode context", "2026-01-02T00:00:00+00:00", thread_id=first_thread)
    _add_event(conn, "other auth", "2026-01-03T00:00:00+00:00", entity_names=["auth"], thread_id=second_thread)

    hits = search(conn, entity="auth", limit=1)
    episodes = group_into_episodes(conn, hits)

    assert len(episodes) == 1
    assert episodes[0].thread_id == first_thread
    assert [event.event_id for event in episodes[0].events] == [other_event, match_event]
    assert episodes[0].matched_by == ["entity"]


def test_search_episode_order_preserves_relevance(conn):
    older_thread = _add_thread(
        conn,
        "Older but relevant",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00+00:00",
    )
    newer_thread = _add_thread(
        conn,
        "Newer but less relevant",
        "2026-02-01T00:00:00+00:00",
        "2026-02-01T00:00:00+00:00",
    )
    older_event = _add_event(
        conn,
        "older",
        "2026-01-01T00:00:00+00:00",
        thread_id=older_thread,
    )
    newer_event = _add_event(
        conn,
        "newer",
        "2026-02-01T00:00:00+00:00",
        thread_id=newer_thread,
    )
    hits = [
        get_thread_detail(conn, older_thread).events[0],
        get_thread_detail(conn, newer_thread).events[0],
    ]
    hits[0].matched_by = ["vector"]
    hits[0].scores = {"vector_score": 0.95}
    hits[1].matched_by = ["vector"]
    hits[1].scores = {"vector_score": 0.80}

    episodes = group_into_episodes(conn, hits)

    assert older_event != newer_event
    assert [episode.thread_id for episode in episodes] == [
        older_thread,
        newer_thread,
    ]


def test_search_no_embedding_model(conn):
    thread_id = _add_thread(conn, "Auth", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00")
    event_id = _add_event(conn, "auth fallback", "2026-01-01T00:00:00+00:00", thread_id=thread_id)

    hits = search(conn, text="auth", embedding_model=None)

    assert [hit.event_id for hit in hits] == [event_id]
    assert hits[0].matched_by == ["text"]


def test_ask_json_flag_no_llm(tmp_path, monkeypatch):
    db_path = tmp_path / "ask.db"
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    monkeypatch.setattr("providers.EMBEDDING_PROVIDER", "none")
    conn = db.get_connection(db_path)
    db.init_db(conn)
    event_id = _add_event(conn, "auth cli fallback", "2026-01-01T00:00:00+00:00")
    _add_fact(conn, event_id, "The person used auth cli fallback.", ["auth"])
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ask", "--json", "auth"],
        env={"MENISCUS_DB_PATH": str(db_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["facts"][0]["event_id"] == event_id


def test_search_text_window_recall(conn):
    """In-window matches must surface even when out-of-window matches would
    fill the relevance limit first (Fix 2)."""

    # 25 older "auth" events outside the window (lower rowids sort first on tie).
    for i in range(25):
        _add_event(conn, "auth notes", f"2025-03-{(i % 28) + 1:02d}T09:00:00+00:00")
    # 3 "auth" events inside the window.
    inside = [
        _add_event(conn, "auth notes", "2026-06-10T09:00:00+00:00"),
        _add_event(conn, "auth notes", "2026-06-12T09:00:00+00:00"),
        _add_event(conn, "auth notes", "2026-06-15T09:00:00+00:00"),
    ]

    hits = search(
        conn,
        text="auth",
        start="2026-06-01",
        end="2026-06-30",
        limit=5,
    )

    returned = {hit.event_id for hit in hits}
    assert returned == set(inside)
    assert all("2026-06" in hit.timestamp for hit in hits)
