"""End-to-end user-workflow test — every command, as a user would run it.

Real Gemini needs a live key/quota, so this uses a stand-in model to document
and LOCK the full journey: add -> (auto-process) -> entities + threads form ->
status -> list -> show -> ask -> process -> duplicate -> rebuild.

If this passes, the machinery works; a real-world failure then localizes to the
live model/API, not the pipeline.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import config as config_mod
import db as db_mod
import providers as providers_mod
import thread_assigner as thread_assigner_mod
from cli.main import AskAnswer, RetrievalParams, cli
from models import ExtractedEntity, ExtractionResult, ThreadSummary

_KEYWORDS = ["jwt", "auth", "token", "refresh", "groceries", "store"]


class WorkflowModel:
    """Deterministic stand-in for Gemini across all four structured calls."""

    def generate_structured(self, prompt, response_model):
        name = response_model.__name__
        if name == "ExtractionResult":
            # Extract from the CONTENT only, not the whole prompt (the prompt's
            # examples mention words like "refresh token"). The real model reads
            # the event content; the fake must too.
            content = prompt.split("Content:", 1)[-1].lower()
            entities = [
                ExtractedEntity(name=k, aliases=[]) for k in _KEYWORDS if k in content
            ]
            return ExtractionResult(
                entities=entities or [ExtractedEntity(name="note", aliases=[])],
                facts=[content.strip() or "a note"],
            )
        if name == "ThreadSummary":
            return ThreadSummary(title="A work session", summary="Some work happened.")
        if name == "RetrievalParams":
            return RetrievalParams(text="jwt")
        if name == "AskAnswer":
            return AskAnswer(answered=True, answer="You worked on JWT auth tokens.")
        raise AssertionError(f"unexpected response model: {name}")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MENISCUS_DB_PATH", str(tmp_path / "workflow.db"))
    # Run in disabled-embeddings mode so no sqlite-vec / API key is needed.
    for module in (config_mod, db_mod, providers_mod, thread_assigner_mod):
        monkeypatch.setattr(module, "EMBEDDING_PROVIDER", "none", raising=False)
    monkeypatch.setattr(providers_mod, "get_model", lambda *a, **k: WorkflowModel())
    monkeypatch.setattr(providers_mod, "get_embedding_model", lambda *a, **k: None)
    return CliRunner()


def _run(runner, *args) -> str:
    result = runner.invoke(cli, list(args))
    assert result.exit_code == 0, f"`men {' '.join(args)}` failed:\n{result.output}"
    return result.output


def test_full_user_journey(runner):
    # 1. Fresh system is empty.
    assert "Events:   0" in _run(runner, "status")

    # 2. Capture three notes: two related (jwt) + one unrelated (groceries).
    added = _run(runner, "add", "learning about jwt auth refresh tokens")
    assert "fact" in added.lower()  # facts extracted, not left pending
    _run(runner, "add", "more jwt refresh token debugging today")
    _run(runner, "add", "bought groceries at the store")

    # 3. All processed into facts.
    status = _run(runner, "status")
    assert "3 completed, 0 pending" in status
    assert "Facts:" in status and "Facts:    0" not in status
    assert "Entities:" in status and "Entities: 0" not in status

    # 4. Events are listable and inspectable.
    events = _run(runner, "list", "events", "--source", "cli")
    assert "jwt" in events.lower()
    shown = _run(runner, "show", "event", "1")
    assert "jwt" in shown.lower()

    # 5. Ask returns a synthesized, grounded answer over facts.
    answer = _run(runner, "ask", "what did i do with jwt?")
    assert "jwt" in answer.lower()

    # 6. Nothing left to process.
    assert "caught up" in _run(runner, "process").lower()

    # 7. Duplicate capture is rejected.
    assert "duplicate" in _run(
        runner, "add", "learning about jwt auth refresh tokens"
    ).lower()

    # 8. Dormant thread grouping can still be rebuilt for a comparison run.
    rebuilt = _run(runner, "rebuild-threads")
    assert "rebuilt" in rebuilt.lower() and "3 event(s)" in rebuilt.lower()
