from __future__ import annotations

from models import ExtractedEntity, ExtractionResult, ThreadSummary
from pipeline import ingest_and_process, process_pending_events


class FakeModel:
    def generate_structured(self, prompt, response_model):
        if response_model is ExtractionResult:
            return ExtractionResult(entities=[ExtractedEntity(name="python")])
        if response_model is ThreadSummary:
            return ThreadSummary(title="Python work", summary="Worked on Python.")
        raise AssertionError(response_model)


def test_ingest_and_process_without_embeddings(conn):
    event_ids = ingest_and_process(
        conn,
        content="learning python",
        source="cli",
        model=FakeModel(),
        embedding_model=None,
    )

    assert len(event_ids) == 1
    event = conn.execute("SELECT extraction_status FROM events").fetchone()
    assert event["extraction_status"] == "completed"
    assert conn.execute("SELECT COUNT(*) FROM event_thread_edges").fetchone()[0] == 1


def test_process_pending_events(conn):
    from event_intake import ingest_event

    ingest_event(conn, "one", "cli", timestamp="2026-01-01T00:00:00+00:00")
    ingest_event(conn, "two", "cli", timestamp="2026-01-01T01:00:00+00:00")

    processed = process_pending_events(conn, FakeModel(), None)

    assert len(processed) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE extraction_status = 'completed'"
    ).fetchone()[0] == 2


def test_ingest_and_process_no_model_saves_pending(conn):
    """Capture must not fail when no model is available (Fix 1)."""

    event_ids = ingest_and_process(
        conn,
        content="a quick thought",
        source="cli",
        model=None,
        embedding_model=None,
    )

    assert len(event_ids) == 1
    row = conn.execute(
        "SELECT extraction_status FROM events WHERE id = ?", (event_ids[0],)
    ).fetchone()
    assert row["extraction_status"] == "pending"
    assert conn.execute("SELECT COUNT(*) FROM event_thread_edges").fetchone()[0] == 0


def test_process_pending_events_no_model_is_noop(conn):
    from event_intake import ingest_event

    ingest_event(conn, "later", "cli", timestamp="2026-01-01T00:00:00+00:00")
    assert process_pending_events(conn, None, None) == []
    assert conn.execute(
        "SELECT COUNT(*) FROM events WHERE extraction_status = 'pending'"
    ).fetchone()[0] == 1


def test_import_and_process_parallel(conn, tmp_path):
    """Batch import: parallel extract, serial assign, progress callback."""
    from pipeline import import_and_process

    notes = tmp_path / "notes"
    (notes / "sub").mkdir(parents=True)          # also proves recursion
    (notes / "a.md").write_text("learning about python generators")
    (notes / "b.md").write_text("more python generator patterns today")
    (notes / "sub" / "c.md").write_text("nested note about python")

    progress: list[tuple[int, int]] = []
    ids = import_and_process(
        conn, str(notes), FakeModel(), None,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert len(ids) == 3
    completed = conn.execute(
        "SELECT COUNT(*) FROM events WHERE extraction_status = 'completed'"
    ).fetchone()[0]
    assert completed == 3
    assert conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0] >= 1
    assert progress[-1] == (3, 3)               # progress reported to completion
    # every thread got a summary title
    titles = [r[0] for r in conn.execute("SELECT title FROM threads").fetchall()]
    assert all(t for t in titles)
