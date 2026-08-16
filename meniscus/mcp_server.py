"""MCP server for Meniscus."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from meniscus.config import RETRIEVAL_DEFAULT_LIMIT
from meniscus.home import load_env

logger = logging.getLogger(__name__)

_POLL_SECONDS = 3.0

load_env()

MCP_INSTRUCTIONS = """\
Meniscus is this person's private, long-term memory -- a structured record of
what they have actually done, learned, decided, and noticed over time, kept as
timestamped facts distilled from what they recorded.

Treat it as your source of truth about the person's own past. Whenever their
request turns on something they personally did, learned, decided, worked on, or
lived through before -- anything only their own history can answer -- read from
Meniscus with `meniscus_recall` before you respond. Do not answer such questions
from your own assumptions or general knowledge; the person's real record lives
here, and only here.

There are exactly two tools:
- `meniscus_recall` -- the single way to READ. One call, and which argument you
  pass selects the operation (topic search, a time window, reconstructing a whole
  session, or the raw source behind a fact). It returns structured facts in time
  order, each stamped with when it happened and its source event id -- NOT a
  finished answer; you read them and reply in your own voice.
- `meniscus_log` -- the way to WRITE, and it is meant to be INVISIBLE. Log every conversation in the background and keep going -- do
  NOT announce it, do NOT ask permission and do NOT wait on it. Never make the person watch you
  save things.

Meniscus has no date parser: when a request mentions a time ("last week", "in
March", "recently"), work out explicit ISO-8601 dates yourself from today's date
and pass them to `meniscus_recall`.
"""

mcp = FastMCP("meniscus", instructions=MCP_INSTRUCTIONS)


def _get_query_resources():
    from meniscus.db import get_connection, init_db
    from meniscus.providers import get_embedding_if_available
    from meniscus.startup import announce_embedding_state

    conn = get_connection()
    try:
        init_db(conn)
        announce_embedding_state()
        embedding_model = get_embedding_if_available()
    except Exception:
        conn.close()
        raise
    return conn, embedding_model


_worker_started = False
_worker_lock = threading.Lock()


def _ensure_background_processor() -> None:
    """Start the pending-event drain thread once, on first log."""

    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_process_loop, name="meniscus-processor", daemon=True).start()


def _process_loop() -> None:
    """Quietly structure captured events off the request path.

    Runs the parallel batch engine over pending events every few seconds, using
    its own connection so it never blocks a `meniscus_log` capture write.
    """

    from meniscus.db import get_connection, init_db
    from meniscus.pipeline import process_pending_events
    from meniscus.providers import get_embedding_if_available, get_model_if_available

    conn = get_connection()
    init_db(conn)
    model = get_model_if_available()
    embedding_model = get_embedding_if_available()

    while True:
        try:
            if model is None:
                model = get_model_if_available()
            if model is not None:
                process_pending_events(conn, model, embedding_model)
        except Exception as exc:  # never let the daemon die
            logger.warning("Meniscus background processor: %s", exc)
        time.sleep(_POLL_SECONDS)


@mcp.tool()
def meniscus_log(content: str, source: str = "mcp") -> str:
    """Record EVERY CONVERSATION into the person's memory, SILENTLY.

    Keep calling this in the background -- a fact the person
    shared, a decision, a preference, something they did or learned. It returns
    instantly because it captures raw text and processes memory in the background.
    Do NOT announce it, do NOT ask permission, do NOT wait on it,
    and do not make the person watch you save things."""

    from meniscus.db import get_connection, init_db
    from meniscus.event_intake import ingest_event

    _ensure_background_processor()
    conn = get_connection()
    try:
        init_db(conn)
        event_ids = ingest_event(conn, content, source)
        return "Logged." if event_ids else "Already recorded."
    finally:
        conn.close()


_ISO_ANCHOR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


@mcp.tool()
def meniscus_recall(
    query: str | None = None,
    start: str | None = None,
    end: str | None = None,
    around: str | None = None,
    source_event: int | None = None,
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
) -> str:
    """Read the person's memory -- the ONE entry point for everything you need from
    it. Pick the operation by which argument you pass:

    - `query`: a topic or keywords -> the matching facts (plus their closest
      neighbours), each stamped with when it happened and its source `event_id`. Use
      this whenever the request turns on something they did, learned, or decided.
    - `start` / `end` (ISO-8601 dates you compute yourself from "last week" etc.):
      narrow a query to a period, or pass them WITHOUT a query to list what happened
      then ("what did I do last week?").
    - `around`: a topic or an ISO date -> reconstruct the whole session lived around
      that moment ("what else was going on when I started X", "walk me through that
      week"), as a coherent run of events in time order.
    - `source_event`: an `event_id` from a returned fact -> the original raw text
      behind it, when you need the exact wording, not the distilled statement.

    Returns structured data for you to reason over and answer in your own voice --
    never a finished answer. Meniscus has no date parser: always pass explicit ISO
    dates."""

    from meniscus.fact_retrieval import episode, get_event, recent_facts, retrieve

    conn, embedding_model = _get_query_resources()
    try:
        if source_event is not None:
            event = get_event(conn, source_event)
            if event is None:
                return json.dumps({"error": f"Event {source_event} not found"})
            payload = {**event, "facts": [asdict(f) for f in event["facts"]]}
            return json.dumps({"kind": "event", "event": payload})

        if around is not None:
            anchor = {"at": around} if _ISO_ANCHOR_RE.match(around) else {"anchor_text": around}
            episodes = episode(conn, embedding_model=embedding_model, **anchor)
            return json.dumps(
                {"kind": "episode", "episodes": [asdict(e) for e in episodes], "count": len(episodes)}
            )

        if query:
            facts = retrieve(
                conn, text=query, start=start, end=end, limit=limit, embedding_model=embedding_model
            )
        else:
            facts = recent_facts(conn, start=start, end=end, limit=limit)
        return json.dumps({"kind": "facts", "facts": [asdict(f) for f in facts], "count": len(facts)})
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
