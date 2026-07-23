from __future__ import annotations

import json

from click.testing import CliRunner

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


def _insert_thread(conn, title, created_at, updated_at):
    return int(
        conn.execute(
            "INSERT INTO threads (title, summary, created_at, updated_at) "
            "VALUES (?, '', ?, ?)",
            (title, created_at, updated_at),
        ).lastrowid
    )


def _insert_event(conn, content, timestamp, source="test", thread_id=None):
    event_id = int(
        conn.execute(
            "INSERT INTO events "
            "(source, content, timestamp, extraction_status, content_hash) "
            "VALUES (?, ?, ?, 'completed', ?)",
            (source, content, timestamp, f"{source}:{content}:{timestamp}"),
        ).lastrowid
    )
    if thread_id is not None:
        conn.execute(
            "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
            (event_id, thread_id),
        )
    return event_id


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


def test_list_threads_uses_any_event_in_window(tmp_path, monkeypatch):
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    touched_thread = _insert_thread(
        conn,
        "Touched in window",
        "2026-01-01T00:00:00+00:00",
        "2026-02-01T00:00:00+00:00",
    )
    outside_thread = _insert_thread(
        conn,
        "Outside",
        "2026-02-01T00:00:00+00:00",
        "2026-02-01T00:00:00+00:00",
    )
    _insert_event(
        conn,
        "inside historical slice",
        "2026-01-15T12:00:00+00:00",
        thread_id=touched_thread,
    )
    _insert_event(
        conn,
        "outside historical slice",
        "2026-02-01T12:00:00+00:00",
        thread_id=outside_thread,
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        cli,
        [
            "list",
            "threads",
            "--since",
            "2026-01-15",
            "--until",
            "2026-01-15",
        ],
        env={"MENISCUS_DB_PATH": str(path)},
    )

    assert result.exit_code == 0
    assert "Touched in window" in result.output
    assert "Outside" not in result.output


def test_ask_unanswered_appends_deterministic_guidance(tmp_path, monkeypatch):
    _path, conn = _open_cli_db(tmp_path, monkeypatch)
    thread_id = _insert_thread(
        conn,
        "Authentication work",
        "2026-01-15T00:00:00+00:00",
        "2026-01-16T00:00:00+00:00",
    )
    _insert_event(
        conn,
        "Worked on auth tokens",
        "2026-01-15T12:00:00+00:00",
        thread_id=thread_id,
    )
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
    assert f"{thread_id}: Authentication work (2026-01-15)" in result.output
    assert f"men show thread {thread_id}" in result.output
    assert (
        "men list events --since 2026-01-15 --until 2026-01-16"
        in result.output
    )
    assert (
        "men list threads --since 2026-01-15 --until 2026-01-16"
        in result.output
    )
    assert "men status" in result.output


def test_ask_json_does_not_synthesize(tmp_path, monkeypatch):
    _path, conn = _open_cli_db(tmp_path, monkeypatch)
    thread_id = _insert_thread(
        conn,
        "Auth",
        "2026-01-15T00:00:00+00:00",
        "2026-01-15T00:00:00+00:00",
    )
    _insert_event(
        conn,
        "auth",
        "2026-01-15T00:00:00+00:00",
        thread_id=thread_id,
    )
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
    assert payload["episodes"][0]["thread_id"] == thread_id
