from __future__ import annotations

from exceptions import ModelUnavailableError
from models import ThreadSummary
from thread_summarizer import summarize_thread


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.prompt = None

    def generate_structured(self, prompt, response_model):
        self.prompt = prompt
        if self.error is not None:
            raise self.error
        return self.result


def _thread_with_event(conn, content="worked on auth"):
    conn.execute(
        "INSERT INTO events (source, content, timestamp, extraction_status, content_hash) "
        "VALUES ('test', ?, '2026-01-01T00:00:00+00:00', 'completed', ?)",
        (content, f"hash-{content}"),
    )
    event_id = conn.execute("SELECT MAX(id) FROM events").fetchone()[0]
    conn.execute(
        "INSERT INTO threads (title, summary, created_at, updated_at) "
        "VALUES ('', '', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    thread_id = conn.execute("SELECT MAX(id) FROM threads").fetchone()[0]
    conn.execute(
        "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
        (event_id, thread_id),
    )
    conn.commit()
    return thread_id


def test_summarize_thread_basic(conn):
    thread_id = _thread_with_event(conn)
    model = FakeModel(
        result=ThreadSummary(title="Auth work", summary="Worked on auth.")
    )
    summarize_thread(
        conn,
        thread_id,
        model,
    )

    row = conn.execute("SELECT title, summary FROM threads WHERE id = ?", (thread_id,)).fetchone()
    assert row["title"] == "Auth work"
    assert row["summary"] == "Worked on auth."
    assert "invent nothing" in model.prompt
    assert "Don't expose the machinery" in model.prompt


def test_summarize_thread_model_unavailable(conn):
    thread_id = _thread_with_event(conn, "a long fallback title for the thread")
    summarize_thread(conn, thread_id, FakeModel(error=ModelUnavailableError("down")))

    row = conn.execute("SELECT title, summary FROM threads WHERE id = ?", (thread_id,)).fetchone()
    assert row["title"] == "a long fallback title for the thread"
    assert row["summary"] == ""
