from __future__ import annotations

import json

from click.testing import CliRunner

import config
import db
from cli import main as cli_main
from cli.main import AskAnswer, RetrievalParams, cli


class SequencedModel:
    def __init__(self, responses):
        self._responses = iter(responses)

    def generate_structured(self, prompt, response_model):
        response = next(self._responses)
        assert isinstance(response, response_model)
        return response


def _open_cli_db(tmp_path, monkeypatch):
    path = tmp_path / "cli.db"
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    conn = db.get_connection(path)
    db.init_db(conn)
    return path, conn


def _insert_event(conn, content, timestamp, source="test"):
    return int(
        conn.execute(
            "INSERT INTO events "
            "(source, content, timestamp, extraction_status, content_hash) "
            "VALUES (?, ?, ?, 'completed', ?)",
            (source, content, timestamp, f"{source}:{content}:{timestamp}"),
        ).lastrowid
    )


def _insert_fact(conn, event_id, text, entities):
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
    for name in entities:
        entity_id = int(
            conn.execute(
                "INSERT INTO entities (canonical_name, normalized_form, created_at) "
                "VALUES (?, ?, '2026-01-01')",
                (name, name),
            ).lastrowid
        )
        conn.execute(
            "INSERT INTO fact_entity_edges (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, entity_id),
        )
    return fact_id


def test_list_events_filters_compose_and_date_end_is_inclusive(tmp_path, monkeypatch):
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    _insert_event(conn, "previous day", "2026-01-14T23:59:00+00:00", "agent")
    in_window = _insert_event(
        conn,
        "late target",
        "2026-01-15T23:30:00+00:00",
        "agent",
    )
    _insert_event(conn, "wrong source", "2026-01-15T12:00:00+00:00", "file")
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        cli,
        [
            "list",
            "events",
            "--since",
            "2026-01-15",
            "--until",
            "2026-01-15",
            "--source",
            "agent",
        ],
        env={"MENISCUS_DB_PATH": str(path)},
    )

    assert result.exit_code == 0
    assert str(in_window) in result.output
    assert "late target" in result.output
    assert "previous day" not in result.output
    assert "wrong source" not in result.output


def test_ask_unanswered_appends_deterministic_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_COLLAPSE", False)  # assert fact-text guidance, not collapse
    _path, conn = _open_cli_db(tmp_path, monkeypatch)
    event_id = _insert_event(conn, "Worked on auth tokens", "2026-01-15T12:00:00+00:00")
    _insert_fact(conn, event_id, "The person worked on auth tokens.", ["auth"])
    conn.commit()
    model = SequencedModel(
        [
            RetrievalParams(
                text="auth",
                start="2026-01-15",
                end="2026-01-16",
            ),
            AskAnswer(
                answered=False,
                answer="I don't have anything that answers that directly.",
            ),
        ]
    )
    monkeypatch.setattr(
        cli_main,
        "_get_retrieval_resources",
        lambda: (conn, model, None, True),
    )

    result = CliRunner().invoke(cli, ["ask", "Did I choose OAuth?"])

    assert result.exit_code == 0
    assert "I don't have anything that answers that directly." in result.output
    assert "The person worked on auth tokens." in result.output
    assert f"men show event {event_id}" in result.output
    assert (
        "men list events --since 2026-01-15 --until 2026-01-16"
        in result.output
    )
    assert "men status" in result.output


def test_ask_json_does_not_synthesize(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SOURCE_COLLAPSE", False)  # assert raw fact passthrough, not collapse
    _path, conn = _open_cli_db(tmp_path, monkeypatch)
    event_id = _insert_event(conn, "auth", "2026-01-15T00:00:00+00:00")
    _insert_fact(conn, event_id, "auth work happened.", ["auth"])
    conn.commit()
    model = SequencedModel([RetrievalParams(text="auth")])
    monkeypatch.setattr(
        cli_main,
        "_get_retrieval_resources",
        lambda: (conn, model, None, True),
    )

    result = CliRunner().invoke(cli, ["ask", "--json", "auth"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["facts"][0]["text"] == "auth work happened."
