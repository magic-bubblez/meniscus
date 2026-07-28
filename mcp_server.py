"""MCP server for Meniscus."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from config import RETRIEVAL_DEFAULT_LIMIT

logger = logging.getLogger(__name__)

# Background processor drains pending events so logging never blocks on LLM calls.
_POLL_SECONDS = 3.0

load_dotenv()

MCP_INSTRUCTIONS = """\
Meniscus is this person's private, long-term memory -- a structured record of
what they have actually done, learned, decided, and noticed over time, kept as
timestamped facts distilled from what they recorded.

Treat it as your source of truth about the person's own past. Whenever their
request turns on something they personally did, learned, decided, worked on, or
lived through before -- anything only their own history can answer -- retrieve
from Meniscus with the tools below before you respond. Do not answer such
questions from your own assumptions or general knowledge; the person's real
record lives here, and only here.

Two things hold across every tool:
- A "fact" is a single self-contained statement, stamped with when it happened
  and the id of the source event it came from. Facts come back in time order.
- Meniscus has no date parser. When a request mentions a time ("last week", "in
  March", "recently"), work out explicit ISO-8601 dates yourself from today's
  date and pass them.

The tools return structured facts, not finished answers. You read them and reply
in your own voice -- grounded in what the records actually show. To see the raw
source text behind a fact, call `meniscus_event` with its `event_id`.

Logging (`meniscus_log`) is meant to be INVISIBLE. When something is worth
remembering, log it in the background and keep going -- do NOT announce it, ask
permission, or wait on it. The call returns instantly; the memory is organized
in the background. Never make the person watch you save things.
"""

mcp = FastMCP("meniscus", instructions=MCP_INSTRUCTIONS)


def _get_resources():
    from db import get_connection, init_db
    from exceptions import ModelUnavailableError
    from providers import get_embedding_model, get_model
    from startup import announce_embedding_state

    conn = get_connection()
    try:
        init_db(conn)
        announce_embedding_state()
        # A missing model saves the event as pending rather than failing intake.
        try:
            model = get_model()
        except (ModelUnavailableError, ImportError):
            model = None
        try:
            embedding_model = get_embedding_model()
        except (ModelUnavailableError, ImportError):
            embedding_model = None
    except Exception:
        conn.close()
        raise
    return conn, model, embedding_model


def _get_query_resources():
    from db import get_connection, init_db
    from exceptions import ModelUnavailableError
    from providers import get_embedding_model
    from startup import announce_embedding_state

    conn = get_connection()
    try:
        init_db(conn)
        announce_embedding_state()
        try:
            embedding_model = get_embedding_model()
        except (ModelUnavailableError, ImportError):
            embedding_model = None
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

    from db import get_connection, init_db
    from exceptions import ModelUnavailableError
    from pipeline import process_pending_events
    from providers import get_embedding_model, get_model

    conn = get_connection()
    init_db(conn)
    try:
        model = get_model()
    except (ModelUnavailableError, ImportError):
        model = None
    try:
        embedding_model = get_embedding_model()
    except (ModelUnavailableError, ImportError):
        embedding_model = None

    while True:
        try:
            if model is not None:
                process_pending_events(conn, model, embedding_model)
        except Exception as exc:  # never let the daemon die
            logger.warning("Meniscus background processor: %s", exc)
        time.sleep(_POLL_SECONDS)


@mcp.tool()
def meniscus_log(content: str, source: str = "mcp") -> str:
    """Record something worth remembering into the person's memory, SILENTLY.

    Call this in the background after a meaningful moment -- a fact the person
    shared, a decision, a preference, something they did or learned. It returns
    instantly (it only captures the text; entities, threading, and summaries are
    built in the background). Do NOT announce it, ask permission, or wait on it,
    and do not make the person watch you save things."""

    from db import get_connection, init_db
    from event_intake import ingest_event

    _ensure_background_processor()
    conn = get_connection()
    try:
        init_db(conn)
        event_ids = ingest_event(conn, content, source)  # Stage 1 only — instant
        return "Logged." if event_ids else "Already recorded."
    finally:
        conn.close()


@mcp.tool()
def meniscus_query(
    query: str,
    entity: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
) -> str:
    """Search the person's memory for facts about a topic. Use this whenever the
    person's request refers to something they may have done, learned, or decided
    before. Pass `query` as the topic or keywords. To limit to a time period,
    compute explicit ISO-8601 dates yourself from phrases like "last week" using
    today's date, and pass `start`/`end`. Returns the matching facts plus their
    closest neighbours, in time order, each with its `event_id` -- structured
    data for you to reason over, NOT a finished answer; you write that."""

    from fact_retrieval import retrieve

    conn, embedding_model = _get_query_resources()
    try:
        facts = retrieve(
            conn,
            text=query,
            entity=entity,
            start=start,
            end=end,
            limit=limit,
            embedding_model=embedding_model,
        )
        return json.dumps({"facts": [asdict(f) for f in facts], "count": len(facts)})
    finally:
        conn.close()


@mcp.tool()
def meniscus_recent(
    start: str | None = None,
    end: str | None = None,
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
) -> str:
    """List the person's facts from their memory, most recent first, optionally
    limited to a time window. Use this for time-based questions that name no
    particular topic ("what did I do last week?", "what have I been working on
    lately?"). To limit to a period, compute explicit ISO-8601 `start`/`end`
    dates yourself from today's date; omit them for the most recent facts
    overall. Returns structured facts for you to read and summarize -- not a
    finished answer."""

    from db import get_connection, init_db
    from fact_retrieval import recent_facts

    conn = get_connection()
    init_db(conn)
    try:
        facts = recent_facts(conn, start=start, end=end, limit=limit)
        return json.dumps({"facts": [asdict(f) for f in facts], "count": len(facts)})
    finally:
        conn.close()


@mcp.tool()
def meniscus_event(event_id: int) -> str:
    """Fetch one raw source event by its id, with the facts distilled from it.
    The ids come from the `event_id` on any fact returned by meniscus_query or
    meniscus_recent -- call this when you need the original wording behind a fact
    rather than the distilled statement. Returns that single event, or an error
    if no event has that id."""

    from db import get_connection, init_db
    from fact_retrieval import get_event

    conn = get_connection()
    init_db(conn)
    try:
        event = get_event(conn, event_id)
        if event is None:
            return json.dumps({"error": f"Event {event_id} not found"})
        payload = {**event, "facts": [asdict(f) for f in event["facts"]]}
        return json.dumps({"event": payload})
    finally:
        conn.close()


@mcp.tool()
def meniscus_episode(
    query: str | None = None,
    event_id: int | None = None,
    at: str | None = None,
) -> str:
    """Reconstruct the session the person lived around a moment -- "what else was I
    doing around X", "walk me through that session/week". Anchor it with `query` (a
    topic), `event_id`, or `at` (an ISO date/time you compute from today). Returns the
    contiguous run of events around that anchor, in time order, each with its facts --
    the coherent episode, not scattered matches. Use this for browsing a period; use
    meniscus_query for topic search."""

    from fact_retrieval import episode

    conn, embedding_model = _get_query_resources()
    try:
        episodes = episode(
            conn,
            anchor_text=query,
            anchor_event_id=event_id,
            at=at,
            embedding_model=embedding_model,
        )
        return json.dumps(
            {"episodes": [asdict(e) for e in episodes], "count": len(episodes)}
        )
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
