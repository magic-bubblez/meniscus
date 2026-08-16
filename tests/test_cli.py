from __future__ import annotations

import json
import os
import platform
import plistlib
import shutil
import stat
import subprocess
from pathlib import Path

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


def _stub_service_install(tmp_path, monkeypatch, system, *, running=True, which=True):
    """Install a fake supervisor toolchain and return the recorded command list."""

    import shutil
    import time

    from meniscus import home

    tool_python = tmp_path / "uv-tools" / "meniscus" / "bin" / "python"
    base_python = tmp_path / "homebrew" / "bin" / "python3"
    tool_python.parent.mkdir(parents=True, exist_ok=True)
    base_python.parent.mkdir(parents=True, exist_ok=True)
    base_python.touch()
    if not tool_python.exists():
        tool_python.symlink_to(base_python)

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["launchctl", "print"]:
            return subprocess.CompletedProcess(
                args, 0, "state = running\n" if running else "state = exited\n", ""
            )
        if args[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(
                args, 0, "active\n" if running else "failed\n", ""
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr(home, "MENISCUS_HOME", tmp_path / ".meniscus")
    monkeypatch.setattr(cli_main.sys, "executable", str(tool_python))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/systemctl" if which else None)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    # Collapse the stop-confirmation poll so tests do not wait on real time.
    monkeypatch.setattr(cli_main, "_SERVICE_STOP_TIMEOUT_SECONDS", 0.0)
    return calls, tool_python


def test_launchagent_preserves_uv_tool_python_symlink(tmp_path, monkeypatch):
    calls, tool_python = _stub_service_install(tmp_path, monkeypatch, "Darwin")

    cli_main._install_background_service()

    seed = next(args for args in calls if "--once" in args)
    assert seed[0] == str(tool_python)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"
    with plist_path.open("rb") as plist_file:
        plist = plistlib.load(plist_file)
    assert plist["ProgramArguments"][0] == str(tool_python)
    assert [
        "launchctl",
        "bootstrap",
        f"gui/{os.getuid()}",
        str(plist_path),
    ] in calls


def test_launchagent_reports_success_only_when_job_stays_running(tmp_path, monkeypatch, capsys):
    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)

    cli_main._install_background_service()

    assert "installed and started" in capsys.readouterr().out


def test_launchagent_reports_failure_when_job_dies(tmp_path, monkeypatch, capsys):
    """`bootstrap` succeeding only means launchd accepted the job, not that it runs."""

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=False)

    cli_main._install_background_service()

    assert "did not stay running" in capsys.readouterr().out


def test_linux_installs_systemd_user_unit(tmp_path, monkeypatch, capsys):
    calls, tool_python = _stub_service_install(tmp_path, monkeypatch, "Linux")

    cli_main._install_background_service()

    unit = tmp_path / ".config" / "systemd" / "user" / "ai.meniscus.watch.service"
    body = unit.read_text()
    assert f"ExecStart={tool_python} -m meniscus.cli.main watch --distill" in body
    assert f"Environment=MENISCUS_HOME={tmp_path / '.meniscus'}" in body
    assert ["systemctl", "--user", "enable", "--now", "ai.meniscus.watch.service"] in calls
    assert "installed and started" in capsys.readouterr().out


def _stub_linger(monkeypatch, linger_output, *, loginctl=True):
    """Point loginctl at a fixed Linger answer, leaving systemctl behaviour intact."""

    import shutil
    import subprocess

    real_run = subprocess.run

    def run(args, **kwargs):
        if args[:2] == ["loginctl", "show-user"]:
            return subprocess.CompletedProcess(args, 0, linger_output, "")
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: None if (name == "loginctl" and not loginctl) else "/usr/bin/x",
    )


def test_linux_warns_when_linger_is_disabled(tmp_path, monkeypatch, capsys):
    """systemd stops user services at logout; say so where it is actionable."""

    _stub_service_install(tmp_path, monkeypatch, "Linux", running=True)
    _stub_linger(monkeypatch, "Linger=no\n")

    cli_main._install_background_service()

    output = capsys.readouterr().out
    assert "pause when you log out" in output
    assert "enable-linger" in output


def test_linux_silent_when_linger_is_enabled(tmp_path, monkeypatch, capsys):
    _stub_service_install(tmp_path, monkeypatch, "Linux", running=True)
    _stub_linger(monkeypatch, "Linger=yes\n")

    cli_main._install_background_service()

    assert "enable-linger" not in capsys.readouterr().out


def test_linux_silent_when_linger_cannot_be_determined(tmp_path, monkeypatch, capsys):
    """An unreadable answer is not a problem worth reporting."""

    _stub_service_install(tmp_path, monkeypatch, "Linux", running=True)
    _stub_linger(monkeypatch, "", loginctl=False)

    cli_main._install_background_service()

    assert "enable-linger" not in capsys.readouterr().out


def test_macos_never_mentions_linger(tmp_path, monkeypatch, capsys):
    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)
    _stub_linger(monkeypatch, "Linger=no\n")

    cli_main._install_background_service()

    assert "enable-linger" not in capsys.readouterr().out


def test_linux_without_systemd_falls_back_to_manual(tmp_path, monkeypatch, capsys):
    _stub_service_install(tmp_path, monkeypatch, "Linux", which=False)

    cli_main._install_background_service()

    assert "men watch --catch-up" in capsys.readouterr().out
    assert not (tmp_path / ".config" / "systemd").exists()


def test_unsupported_platform_falls_back_to_manual(tmp_path, monkeypatch, capsys):
    _stub_service_install(tmp_path, monkeypatch, "Windows")

    cli_main._install_background_service()

    assert "men watch --catch-up" in capsys.readouterr().out


def test_watch_start_installs_the_service(tmp_path, monkeypatch):
    calls, tool_python = _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)

    result = CliRunner().invoke(cli, ["watch", "--start"])

    assert result.exit_code == 0
    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"
    assert plist_path.exists()
    assert "installed and started" in result.output


def test_watch_start_and_stop_are_mutually_exclusive(tmp_path, monkeypatch):
    _stub_service_install(tmp_path, monkeypatch, "Darwin")

    result = CliRunner().invoke(cli, ["watch", "--start", "--stop"])

    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_watch_start_stop_start_round_trips(tmp_path, monkeypatch):
    """Capture must be switchable off and back on without re-running `men init`."""

    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)
    CliRunner().invoke(cli, ["watch", "--start"])
    assert plist_path.exists()

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=False)
    CliRunner().invoke(cli, ["watch", "--stop"])
    assert not plist_path.exists()

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)
    CliRunner().invoke(cli, ["watch", "--start"])
    assert plist_path.exists()


def test_watch_stop_removes_the_launchagent(tmp_path, monkeypatch):
    """Stopping must delete the definition: KeepAlive and RunAtLoad bring it back otherwise."""

    calls, _ = _stub_service_install(tmp_path, monkeypatch, "Darwin", running=False)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(b"stub")

    result = CliRunner().invoke(cli, ["watch", "--stop"])

    assert result.exit_code == 0
    assert not plist_path.exists()
    assert ["launchctl", "bootout", f"gui/{os.getuid()}/ai.meniscus.watch"] in calls
    assert "stopped and removed" in result.output


def test_watch_stop_removes_the_systemd_unit(tmp_path, monkeypatch):
    calls, _ = _stub_service_install(tmp_path, monkeypatch, "Linux", running=False)
    unit_path = tmp_path / ".config" / "systemd" / "user" / "ai.meniscus.watch.service"
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("stub")

    result = CliRunner().invoke(cli, ["watch", "--stop"])

    assert result.exit_code == 0
    assert not unit_path.exists()
    assert ["systemctl", "--user", "disable", "--now", "ai.meniscus.watch.service"] in calls
    assert "stopped and removed" in result.output


def test_stop_waits_for_a_service_that_is_still_shutting_down(monkeypatch):
    """A job signalled to stop may take seconds to exit; that is not a failed stop."""

    import time

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli_main, "_SERVICE_STOP_TIMEOUT_SECONDS", 5.0)
    states = iter([True, True, False])

    assert cli_main._still_running_after_grace(lambda: next(states)) is False


def test_stop_gives_up_after_the_timeout(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli_main, "_SERVICE_STOP_TIMEOUT_SECONDS", 0.0)

    assert cli_main._still_running_after_grace(lambda: True) is True


def test_stop_reports_success_when_a_slow_service_finally_exits(tmp_path, monkeypatch, capsys):
    """The end-to-end shape of the bug: stop worked, but the report said otherwise."""

    _stub_service_install(tmp_path, monkeypatch, "Darwin")
    monkeypatch.setattr(cli_main, "_SERVICE_STOP_TIMEOUT_SECONDS", 5.0)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(b"stub")

    states = iter([True, True, False])
    monkeypatch.setattr(cli_main, "_launchagent_running", lambda: next(states))

    cli_main._uninstall_background_service()

    assert "stopped and removed" in capsys.readouterr().out


def test_watch_stop_when_nothing_is_installed(tmp_path, monkeypatch):
    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=False)

    result = CliRunner().invoke(cli, ["watch", "--stop"])

    assert result.exit_code == 0
    assert "not installed" in result.output


def test_watch_stop_reports_a_service_that_survives(tmp_path, monkeypatch):
    """Removing the file is not proof the process died."""

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_bytes(b"stub")

    result = CliRunner().invoke(cli, ["watch", "--stop"])

    assert "still running" in result.output


def test_watch_stop_does_not_touch_the_database(tmp_path, monkeypatch):
    """`--stop` is service-only: it must not open or migrate memory as a side effect."""

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=False)

    def fail():
        raise AssertionError("--stop opened the database")

    monkeypatch.setattr(db, "get_connection", fail)

    result = CliRunner().invoke(cli, ["watch", "--stop"])

    assert result.exit_code == 0


def test_install_then_stop_round_trips(tmp_path, monkeypatch):
    """The full service lifecycle: install leaves a definition, stop removes it."""

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=True)
    plist_path = tmp_path / "Library" / "LaunchAgents" / "ai.meniscus.watch.plist"

    cli_main._install_background_service()
    assert plist_path.exists()

    _stub_service_install(tmp_path, monkeypatch, "Darwin", running=False)
    CliRunner().invoke(cli, ["watch", "--stop"])

    assert not plist_path.exists()


def test_write_probe_passes_on_a_healthy_database(tmp_path, monkeypatch):
    _, conn = _open_cli_db(tmp_path, monkeypatch)
    try:
        ok, name, detail = cli_main._probe_write_path(conn)

        assert ok
        assert name == "write path"
        assert "accept" in detail
    finally:
        conn.close()


def test_write_probe_leaves_no_trace(tmp_path, monkeypatch):
    _, conn = _open_cli_db(tmp_path, monkeypatch)
    try:
        cli_main._probe_write_path(conn)

        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    finally:
        conn.close()


def test_write_probe_catches_a_broken_write_path(tmp_path, monkeypatch):
    """The check that would have caught a rejected insert behind a clean-looking schema."""

    _, conn = _open_cli_db(tmp_path, monkeypatch)
    try:
        conn.executescript(
            "CREATE TRIGGER reject_writes BEFORE INSERT ON events BEGIN "
            "SELECT RAISE(ABORT, 'no such table: main.events_fts'); END;"
        )

        ok, _name, detail = cli_main._probe_write_path(conn)

        assert not ok
        assert "cannot record new events" in detail
    finally:
        conn.close()


def test_status_reports_last_event_age(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "none")
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    recent = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    _insert_event(conn, "recent turn", recent)
    conn.commit()
    conn.close()

    result = CliRunner().invoke(cli, ["status"], env={"MENISCUS_DB_PATH": str(path)})

    assert result.exit_code == 0
    assert "Last event:" in result.output
    assert "10m ago" in result.output


def test_status_flags_a_stalled_capture(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "none")
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    stale = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    _insert_event(conn, "old turn", stale)
    conn.commit()
    conn.close()

    result = CliRunner().invoke(cli, ["status"], env={"MENISCUS_DB_PATH": str(path)})

    assert "5d ago" in result.output
    assert "capture may have stopped" in result.output


def test_status_on_an_empty_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "none")
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    conn.close()

    result = CliRunner().invoke(cli, ["status"], env={"MENISCUS_DB_PATH": str(path)})

    assert result.exit_code == 0
    assert "nothing captured yet" in result.output


def test_show_event_reports_the_model_that_wrote_its_facts(tmp_path, monkeypatch):
    path, conn = _open_cli_db(tmp_path, monkeypatch)
    event_id = _insert_event(conn, "a turn worth remembering", "2026-01-15T12:00:00+00:00")
    _insert_fact(conn, event_id, "The person shipped Meniscus.", [])
    conn.commit()
    conn.close()

    result = CliRunner().invoke(
        cli, ["show", "event", str(event_id)], env={"MENISCUS_DB_PATH": str(path)}
    )

    assert result.exit_code == 0
    assert "Facts by: p/m (v) at 2026-01-01" in result.output
