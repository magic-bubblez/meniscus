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

# The background processor drains pending (captured-but-unstructured) events
# every few seconds, so logging never blocks on LLM calls.
_POLL_SECONDS = 3.0

load_dotenv()

MCP_INSTRUCTIONS = """\
Meniscus is this person's private, long-term memory -- a structured record of
what they have actually done, learned, decided, and noticed over time, kept as
timestamped events grouped into episodes (coherent stretches of activity on one
topic).

Treat it as your source of truth about the person's own past. Whenever their
request turns on something they personally did, learned, decided, worked on, or
lived through before -- anything only their own history can answer -- retrieve
from Meniscus with the tools below before you respond. Do not answer such
questions from your own assumptions or general knowledge; the person's real
record lives here, and only here.

Two things hold across every tool:
- An "episode" is a coherent stretch of the person's activity on one topic,
  returned as a title, a short summary, and its events in time order. The events
  are the ground truth.
- Meniscus has no date parser. When a request mentions a time ("last week", "in
  March", "recently"), work out explicit ISO-8601 dates yourself from today's
  date and pass them.

The tools return structured episodes, not finished answers. You read them and
reply in your own voice -- grounded in what the records actually show.

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
        # A missing model must not make logging fail: intake is deterministic,
        # so the event is saved as pending for a later `men process`.
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
    """Search the person's memory for episodes about a topic. Use this whenever
    the person's request refers to something they may have done, learned, or
    decided before. Pass `query` as the topic or keywords. To limit to a time
    period, compute explicit ISO-8601 dates yourself from phrases like "last
    week" using today's date, and pass `start`/`end`. Returns matching episodes
    -- each a title, a summary, and its events in time order -- as structured
    data for you to reason over, NOT a finished answer; you write that."""

    from retrieval import group_into_episodes, search

    conn, embedding_model = _get_query_resources()
    try:
        hits = search(
            conn,
            text=query,
            entity=entity,
            start=start,
            end=end,
            limit=limit,
            embedding_model=embedding_model,
        )
        episodes = group_into_episodes(conn, hits)
        return json.dumps({"episodes": [asdict(e) for e in episodes], "count": len(episodes)})
    finally:
        conn.close()


@mcp.tool()
def meniscus_recent(
    start: str | None = None,
    end: str | None = None,
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
) -> str:
    """List the person's episodes from their memory, most recent first,
    optionally limited to a time window. Use this for time-based questions that
    name no particular topic ("what did I do last week?", "what have I been
    working on lately?"). To limit to a period, compute explicit ISO-8601
    `start`/`end` dates yourself from today's date; omit them for the most recent
    episodes overall. Returns structured episodes for you to read and summarize
    -- not a finished answer."""

    from db import get_connection, init_db
    from retrieval import recent

    conn = get_connection()
    init_db(conn)
    try:
        episodes = recent(conn, start=start, end=end, limit=limit)
        return json.dumps({"episodes": [asdict(e) for e in episodes], "count": len(episodes)})
    finally:
        conn.close()


@mcp.tool()
def meniscus_thread(thread_id: int) -> str:
    """Fetch one complete episode by its id, with every event in it in
    chronological order. The ids come from meniscus_query or meniscus_recent,
    which return episodes in summary form -- call this when one of them looks
    relevant and you need its full detail rather than just the summary. Returns
    that single episode, or nothing if no episode has that id."""

    from db import get_connection, init_db
    from retrieval import get_thread_detail

    conn = get_connection()
    init_db(conn)
    try:
        episode = get_thread_detail(conn, thread_id)
        if episode is None:
            return json.dumps({"error": f"Thread {thread_id} not found"})
        return json.dumps({"episode": asdict(episode)})
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
