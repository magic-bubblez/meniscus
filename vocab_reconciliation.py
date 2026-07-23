"""Deterministic vocabulary reconciliation for extracted entities."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from models import ExtractionResult


def normalize(text: str) -> str:
    """Normalize text into the matching form used for entity lookup."""

    lowered = text.lower()
    replaced = lowered.replace("-", " ").replace("_", " ")
    tokens = replaced.strip().split()
    filtered = [token for token in tokens if len(token) > 1]
    return " ".join(sorted(filtered))


def reconcile_entities(
    conn: sqlite3.Connection,
    event_id: int,
    extraction_result: ExtractionResult,
) -> list[int]:
    """Reconcile extracted entities and create event-entity edges."""

    entity_ids: list[int] = []
    now = datetime.now(timezone.utc).isoformat()

    for extracted in extraction_result.entities:
        normalized = normalize(extracted.name)
        if not normalized:
            continue

        entity_id = _find_entity_id(conn, normalized)

        if entity_id is None:
            cursor = conn.execute(
                "INSERT INTO entities (canonical_name, normalized_form, created_at) "
                "VALUES (?, ?, ?)",
                (extracted.name, normalized, now),
            )
            entity_id = int(cursor.lastrowid)

        _insert_aliases(conn, entity_id, extracted.aliases)
        conn.execute(
            "INSERT OR IGNORE INTO event_entity_edges (event_id, entity_id) "
            "VALUES (?, ?)",
            (event_id, entity_id),
        )

        if entity_id not in entity_ids:
            entity_ids.append(entity_id)

    return entity_ids


def _find_entity_id(conn: sqlite3.Connection, normalized: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM entities WHERE normalized_form = ?",
        (normalized,),
    ).fetchone()
    if row is not None:
        return int(row["id"])

    alias_row = conn.execute(
        "SELECT entity_id FROM entity_aliases WHERE normalized_form = ?",
        (normalized,),
    ).fetchone()
    if alias_row is not None:
        return int(alias_row["entity_id"])

    return None


def _insert_aliases(
    conn: sqlite3.Connection,
    entity_id: int,
    aliases: list[str],
) -> None:
    for alias_text in aliases:
        alias_normalized = normalize(alias_text)
        if not alias_normalized:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases "
            "(entity_id, alias, normalized_form) VALUES (?, ?, ?)",
            (entity_id, alias_text, alias_normalized),
        )
