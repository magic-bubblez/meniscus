"""LLM-based thread summarization."""

from __future__ import annotations

import sqlite3

from db import transactional
from exceptions import ModelUnavailableError
from model_interface import ModelInterface
from models import ThreadSummary

_TITLE_MAX_LENGTH = 80

SUMMARIZATION_PROMPT_TEMPLATE: str = """\
# Your role
You are given all the events in one episode of a person's memory -- a coherent
stretch of their activity on a single topic, in the order it happened -- and you
write a short title and summary for it. These are how the person navigates this
episode later: the title is what they see in a list of episodes; the summary is
the quick orientation they (or an assistant reading their memory) get before the
full detail. Write both to be recognized at a glance and trusted as accurate.

# What you are given
The episode's events, each a timestamp and the text recorded at that moment, in
chronological order. Together they are one arc of activity on one topic. Read them
as a whole -- the episode is the story they tell, not any single event.

# The title
A short, specific label -- think of a good subject line -- that lets the person
pick this episode out of a list and know at once what it was. Name the actual
subject, not a generic category: a title like "Debugging JWT refresh-token expiry"
tells them what it was, whereas "Work session" tells them nothing. Keep it under
{max_title_length} characters, capitalized as you naturally would, with no
trailing punctuation.

# The summary
One to three sentences capturing what this episode was actually about and what
came of it -- what was done, learned, decided, or resolved. Give the throughline,
not a retelling of every event: if the episode moved from a problem to its cause
to a fix, that arc is the summary. Stay grounded in the events -- describe only
what they show, and never add an outcome or detail they don't contain. If the
episode is just beginning and holds little yet, one honest sentence about its
subject is enough; don't inflate it.

# Bounds
- Describe only what the events contain; invent nothing -- no outcomes,
  resolutions, or details that aren't there.
- Summarize; don't judge or editorialize.
- Don't expose the machinery -- no "this thread", "these events", "the episode";
  write about the subject itself.

# What you return
A structured result with a title and a summary.

Events (chronological):
{events_text}
"""


TITLE_MAX_LENGTH = _TITLE_MAX_LENGTH  # public alias for callers


def render_events(event_rows) -> tuple[str, str]:
    """Build (events_text, first_content) from a thread's event rows."""

    events_text = "\n\n".join(
        f"[{row['timestamp']}] ({row['source']}) {row['content']}"
        for row in event_rows
    )
    first_content = event_rows[0]["content"] if event_rows else ""
    return events_text, first_content


def summarize_from_text(events_text: str, model: ModelInterface) -> tuple[str, str]:
    """LLM-only summarization (no DB access) → (title, summary).

    Raises ModelUnavailableError on failure; callers decide on the fallback. This
    is the parallel-safe core used by the batch import path.
    """

    prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(
        max_title_length=_TITLE_MAX_LENGTH,
        events_text=events_text,
    )
    result = model.generate_structured(prompt, ThreadSummary)
    return result.title[:_TITLE_MAX_LENGTH], result.summary


def deterministic_title(content: str) -> str:
    """Fallback title when the model is unavailable."""

    return _deterministic_title(content)


def write_thread_summary(
    conn: sqlite3.Connection, thread_id: int, title: str, summary: str
) -> None:
    """Persist a thread's title/summary in a short transaction."""

    with transactional(conn) as txn:
        txn.execute(
            "UPDATE threads SET title = ?, summary = ? WHERE id = ?",
            (title, summary, thread_id),
        )
        _update_centroid(txn, thread_id)


def summarize_thread(
    conn: sqlite3.Connection,
    thread_id: int,
    model: ModelInterface,
) -> None:
    """Summarize a thread and persist title/summary (serial path)."""

    event_rows = conn.execute(
        "SELECT e.id, e.content, e.timestamp, e.source "
        "FROM events e "
        "JOIN event_thread_edges ete ON e.id = ete.event_id "
        "WHERE ete.thread_id = ? "
        "ORDER BY e.timestamp ASC",
        (thread_id,),
    ).fetchall()
    if not event_rows:
        return

    events_text, first_content = render_events(event_rows)
    try:
        title, summary = summarize_from_text(events_text, model)
    except ModelUnavailableError:
        title, summary = deterministic_title(first_content), ""

    write_thread_summary(conn, thread_id, title, summary)


def _deterministic_title(content: str) -> str:
    if len(content) <= _TITLE_MAX_LENGTH:
        return content
    truncate_at = _TITLE_MAX_LENGTH - 3
    last_space = content.rfind(" ", 0, truncate_at)
    if last_space == -1:
        return content[:truncate_at] + "..."
    return content[:last_space] + "..."


def _update_centroid(conn: sqlite3.Connection, thread_id: int) -> None:
    """No-op placeholder; centroids are computed on demand in v1."""

    return None
