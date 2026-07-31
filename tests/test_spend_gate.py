from __future__ import annotations

import pytest

import config
import spend_gate
from event_intake import ingest_event
from exceptions import SpendCeilingError
from models import ExtractedEntity, ExtractionResult
from pipeline import process_pending_events

MODEL = "deepseek/deepseek-v4-flash"

_ITEMS = [
    ("cli", "learning about python generators", "2026-01-01T00:00:00+00:00"),
    ("cli", "more python generator patterns today", "2026-01-01T01:00:00+00:00"),
]
_PENDING = [(i + 1, s, c, t) for i, (s, c, t) in enumerate(_ITEMS)]


class FakeModel:
    def generate_structured(self, prompt, response_model):
        return ExtractionResult(
            entities=[ExtractedEntity(name="python")],
            facts=["The person is learning Python."],
        )


class ExplodingModel:
    def generate_structured(self, prompt, response_model):
        raise AssertionError("model must not be called once the spend gate aborts")


def test_estimate_cost_is_positive_and_monotonic():
    both = spend_gate.estimate_cost_usd(_ITEMS, MODEL)
    one = spend_gate.estimate_cost_usd(_ITEMS[:1], MODEL)
    assert both > one > 0


def test_unknown_model_uses_default_price():
    assert spend_gate.estimate_cost_usd(_ITEMS, "some/unlisted-model") > 0


def test_enforce_over_budget_aborts(monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 0.0001)
    with pytest.raises(SpendCeilingError) as exc:
        spend_gate.enforce_spend_ceiling(_PENDING, MODEL)
    assert "MAX_RUN_COST_USD" in str(exc.value)
    assert spend_gate.ALLOW_LARGE_RUN_ENV in str(exc.value)


def test_enforce_under_budget_proceeds(monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 100.0)
    assert spend_gate.enforce_spend_ceiling(_PENDING, MODEL) > 0


def test_budget_zero_disables_gate(monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 0.0)
    assert spend_gate.enforce_spend_ceiling(_PENDING, MODEL) > 0


def test_override_param_bypasses(monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 0.0001)
    assert spend_gate.enforce_spend_ceiling(_PENDING, MODEL, allow_large_run=True) > 0


def test_override_env_bypasses(monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 0.0001)
    monkeypatch.setenv(spend_gate.ALLOW_LARGE_RUN_ENV, "1")
    assert spend_gate.enforce_spend_ceiling(_PENDING, MODEL) > 0


def test_process_pending_events_aborts_before_any_model_call(conn, monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 0.0001)
    ingest_event(conn, "one", "cli", timestamp="2026-01-01T00:00:00+00:00")
    ingest_event(conn, "two", "cli", timestamp="2026-01-01T01:00:00+00:00")

    with pytest.raises(SpendCeilingError):
        process_pending_events(conn, ExplodingModel(), None)

    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE extraction_status = 'pending'"
    ).fetchone()[0] == 2


def test_process_pending_events_override_proceeds(conn, monkeypatch):
    monkeypatch.setattr(config, "MAX_RUN_COST_USD", 0.0001)
    ingest_event(conn, "one", "cli", timestamp="2026-01-01T00:00:00+00:00")

    processed = process_pending_events(conn, FakeModel(), None, allow_large_run=True)

    assert len(processed) == 1
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
