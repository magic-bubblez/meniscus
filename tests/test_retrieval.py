from __future__ import annotations

import json

from click.testing import CliRunner

import db
from cli.main import cli


def _add_event(conn, content: str, timestamp: str, *, source: str = "test") -> int:
    cursor = conn.execute(
        "INSERT INTO events (source, content, timestamp, extraction_status, content_hash) "
        "VALUES (?, ?, ?, 'completed', ?)",
        (source, content, timestamp, f"{source}:{content}:{timestamp}"),
    )
    return int(cursor.lastrowid)


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
