from __future__ import annotations

import json
import shutil
import stat
import subprocess

from click.testing import CliRunner

from meniscus import config
from meniscus import db
from meniscus.cli import main as cli_main
from meniscus.cli.main import AskAnswer, RetrievalParams, cli


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


def test_init_uses_provider_default_model_for_both_jobs(tmp_path, monkeypatch):
    from meniscus import home

    meniscus_home = tmp_path / ".meniscus"
    monkeypatch.setattr(home, "MENISCUS_HOME", meniscus_home)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cli_main, "_init_doctor", lambda: None)
    monkeypatch.setattr(
        cli_main,
        "_validate_provider_selection",
        lambda *_args: (True, 'model "gemini-3.5-flash-lite" verified'),
    )
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    result = CliRunner().invoke(
        cli,
        ["init"],
        input="4\nn\n\n3\n",
    )

    assert result.exit_code == 0, result.output
    assert "Extraction model" not in result.output
    assert "Synthesis model" not in result.output
    assert "Model (used for memory processing and answers)" not in result.output
    assert "gemini-3.5-flash-lite" in result.output
    config_text = (meniscus_home / "config.toml").read_text()
    assert 'provider = "gemini"' in config_text
    assert 'model = "gemini-3.5-flash-lite"' in config_text
    assert "extraction_model" not in config_text
    assert "synthesis_model" not in config_text


def test_provider_prompt_has_no_default(tmp_path, monkeypatch):
    from meniscus import home

    meniscus_home = tmp_path / ".meniscus"
    monkeypatch.setattr(home, "MENISCUS_HOME", meniscus_home)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli_main, "_init_doctor", lambda: None)
    monkeypatch.setattr(
        cli_main,
        "_validate_provider_selection",
        lambda *_args: (True, 'model "gemini-3.5-flash-lite" verified'),
    )
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    result = CliRunner().invoke(cli, ["init"], input="4\ntest-key\n\n3\n")

    assert result.exit_code == 0, result.output
    provider_step = result.output.split("Step 2", 1)[0]
    assert "Choice [" not in provider_step
    assert "Choice:" in provider_step


def test_invalid_provider_credentials_are_not_saved(tmp_path, monkeypatch):
    from meniscus import home

    meniscus_home = tmp_path / ".meniscus"
    monkeypatch.setattr(home, "MENISCUS_HOME", meniscus_home)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        cli_main,
        "_validate_provider_selection",
        lambda *_args: (False, "authentication rejected (401)"),
    )

    result = CliRunner().invoke(cli, ["config", "set"], input="6\nxyz\n\n")

    assert result.exit_code != 0
    assert "Nothing was saved" in result.output
    assert not (meniscus_home / ".env").exists()
    assert not (meniscus_home / "config.toml").exists()


def test_provider_recommendation_can_be_overridden(tmp_path, monkeypatch):
    from meniscus import home

    meniscus_home = tmp_path / ".meniscus"
    selected = {}
    monkeypatch.setattr(home, "MENISCUS_HOME", meniscus_home)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    def validate(provider, model, api_key, base_url):
        selected.update(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        return True, f'model "{model}" verified'

    monkeypatch.setattr(cli_main, "_validate_provider_selection", validate)

    result = CliRunner().invoke(
        cli,
        ["config", "set"],
        input="6\nvalid-key\nn\ngrok-custom\n",
    )

    assert result.exit_code == 0, result.output
    assert selected == {
        "provider": "xai",
        "model": "grok-custom",
        "api_key": "valid-key",
        "base_url": None,
    }
    assert 'model = "grok-custom"' in (meniscus_home / "config.toml").read_text()
    env_path = meniscus_home / ".env"
    assert 'XAI_API_KEY="valid-key"' in env_path.read_text()
    assert stat.S_IMODE(meniscus_home.stat().st_mode) == 0o700
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((meniscus_home / "config.toml").stat().st_mode) == 0o600


def test_sql_command_cannot_write_through_pragma(tmp_path, monkeypatch):
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    conn.close()

    result = CliRunner().invoke(
        cli,
        ["sql", "PRAGMA user_version=1"],
        env={"MENISCUS_DB_PATH": str(path)},
    )

    assert result.exit_code != 0
    conn = db.get_connection(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    conn.close()


def test_add_without_model_saves_pending_event(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_MODEL_PROVIDER", None)
    monkeypatch.setattr(config, "MODEL", None)
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    path = tmp_path / "pending.db"

    result = CliRunner().invoke(
        cli,
        ["add", "Remember this later"],
        env={"MENISCUS_DB_PATH": str(path)},
    )

    assert result.exit_code == 0, result.output
    assert "pending" in result.output
    conn = db.get_connection(path)
    row = conn.execute(
        "SELECT content, extraction_status FROM events"
    ).fetchone()
    conn.close()
    assert tuple(row) == ("Remember this later", "pending")


def test_process_without_model_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_MODEL_PROVIDER", None)
    monkeypatch.setattr(config, "MODEL", None)
    monkeypatch.setattr(db, "EMBEDDING_PROVIDER", "none")
    path = tmp_path / "pending.db"

    result = CliRunner().invoke(
        cli,
        ["process"],
        env={"MENISCUS_DB_PATH": str(path)},
    )

    assert result.exit_code == 0, result.output
    assert "No model configured" in result.output


def test_mcp_wiring_uses_user_scope_for_claude(monkeypatch):
    calls = []

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"claude", "codex"} else None,
    )
    monkeypatch.setattr(cli_main.click, "confirm", lambda *args, **kwargs: True)

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    cli_main._wire_mcp_tools("/Users/test/.local/bin/men-mcp")

    assert calls == [
        [
            "claude",
            "mcp",
            "add",
            "--scope",
            "user",
            "meniscus",
            "--",
            "/Users/test/.local/bin/men-mcp",
        ],
        [
            "codex",
            "mcp",
            "add",
            "meniscus",
            "--",
            "/Users/test/.local/bin/men-mcp",
        ],
    ]
