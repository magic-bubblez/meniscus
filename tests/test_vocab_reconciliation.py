from __future__ import annotations

from models import ExtractedEntity, ExtractionResult
from vocab_reconciliation import normalize, reconcile_entities


def _insert_event(conn, hash_value="event"):
    cursor = conn.execute(
        "INSERT INTO events (source, content, timestamp, content_hash) "
        "VALUES ('test', 'content', '2026-01-01T00:00:00+00:00', ?)",
        (hash_value,),
    )
    return cursor.lastrowid


def test_normalize_examples():
    assert normalize("Linked-List") == "linked list"
    assert normalize("user_space") == "space user"
    assert normalize("JWT") == "jwt"
    assert normalize("A/B Testing") == "a/b testing"
    assert normalize("  hello  world  ") == "hello world"
    assert normalize("A") == ""
    assert normalize("") == ""


def test_reconcile_creates_new_entity_and_edge(conn):
    event_id = _insert_event(conn)
    result = ExtractionResult(
        entities=[ExtractedEntity(name="linked list", aliases=["list node"])]
    )

    entity_ids = reconcile_entities(conn, event_id, result)

    assert len(entity_ids) == 1
    entity = conn.execute("SELECT * FROM entities").fetchone()
    assert entity["canonical_name"] == "linked list"
    alias = conn.execute("SELECT * FROM entity_aliases").fetchone()
    assert alias["normalized_form"] == "list node"


def test_reconcile_reuses_existing_entity(conn):
    event_id = _insert_event(conn)
    conn.execute(
        "INSERT INTO entities (canonical_name, normalized_form, created_at) "
        "VALUES ('linked list', 'linked list', '2026-01-01T00:00:00+00:00')"
    )

    entity_ids = reconcile_entities(
        conn,
        event_id,
        ExtractionResult(entities=[ExtractedEntity(name="List Linked")]),
    )

    assert entity_ids == [1]
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1


def test_reconcile_matches_alias(conn):
    event_id = _insert_event(conn)
    conn.execute(
        "INSERT INTO entities (canonical_name, normalized_form, created_at) "
        "VALUES ('machine learning', 'learning machine', "
        "'2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO entity_aliases (entity_id, alias, normalized_form) "
        "VALUES (1, 'ML', 'ml')"
    )

    entity_ids = reconcile_entities(
        conn,
        event_id,
        ExtractionResult(entities=[ExtractedEntity(name="ML")]),
    )

    assert entity_ids == [1]


def test_reconcile_deduplicates_returned_ids(conn):
    event_id = _insert_event(conn)
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="python"),
            ExtractedEntity(name="Python"),
        ]
    )

    entity_ids = reconcile_entities(conn, event_id, result)

    assert entity_ids == [1]


def test_reconcile_skips_empty_normalized_entity(conn):
    event_id = _insert_event(conn)

    entity_ids = reconcile_entities(
        conn,
        event_id,
        ExtractionResult(entities=[ExtractedEntity(name="A")]),
    )

    assert entity_ids == []
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
