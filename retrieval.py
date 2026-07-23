"""Deterministic retrieval over Meniscus indexes."""

from __future__ import annotations

import logging
import re
import sqlite3
import struct
from dataclasses import dataclass, field

import config
from embedding_interface import EmbeddingInterface
from exceptions import ModelUnavailableError
from thread_assigner import cosine_similarity, get_event_embedding
from time_bounds import normalize_time_window
from vocab_reconciliation import normalize

logger = logging.getLogger(__name__)

_FTS_TOKEN_RE = re.compile(r"\w+")

# When a time window is present the vector door oversamples (vec0 KNN cannot
# join on timestamp), then the caller time-filters and trims.
_WINDOW_OVERSAMPLE = 10
_VECTOR_MAX_K = 500


@dataclass
class EventHit:
    """One matching event, annotated with the episode it belongs to."""

    event_id: int
    thread_id: int | None
    content: str
    timestamp: str
    source: str
    entities: list[str] = field(default_factory=list)
    matched_by: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class Episode:
    """A thread packaged for handoff."""

    thread_id: int
    title: str
    summary: str
    events: list[EventHit] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    matched_by: list[str] = field(default_factory=list)
    best_score: float = 0.0


def _sanitize_fts_query(raw: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression."""

    return " ".join(f'"{token}"' for token in _FTS_TOKEN_RE.findall(raw))


def search(
    conn: sqlite3.Connection,
    text: str | None = None,
    entity: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = config.RETRIEVAL_DEFAULT_LIMIT,
    embedding_model: EmbeddingInterface | None = None,
) -> list[EventHit]:
    """Find events matching a topic, optionally within a time window."""

    limit = max(int(limit), 0)
    if limit == 0:
        return []

    # Normalize the window once; the doors apply it themselves so that time is a
    # true filter on the candidate set, not a post-filter over a relevance-
    # truncated list (which would silently drop in-window matches ranked beyond
    # `limit`).
    start_bound, end_bound = normalize_time_window(start, end)
    windowed = start_bound is not None or end_bound is not None

    hits: dict[int, dict] = {}
    query_embedding: list[float] | None = None
    vector_door_fired = False
    text_door_fired = False

    if entity:
        # The entity door is exhaustive, so it is safe to bound it in Python.
        for event_id in _entity_event_ids(conn, entity):
            _record_hit(hits, event_id, "entity")

    if text:
        text_scores = _text_event_scores(conn, text, limit, start_bound, end_bound)
        text_door_fired = bool(text_scores)
        for event_id, rank in text_scores:
            _record_hit(hits, event_id, "text", "fts_rank", rank)

        if embedding_model is not None:
            query_embedding = _embed_query(embedding_model, text)
            if query_embedding is not None:
                # vec0 KNN cannot join, so when a window is set we oversample and
                # let the post-filter below trim (best-effort recall).
                fetch_k = (
                    min(limit * _WINDOW_OVERSAMPLE, _VECTOR_MAX_K)
                    if windowed
                    else limit
                )
                vector_scores = _vector_event_scores(conn, query_embedding, fetch_k)
                vector_door_fired = bool(vector_scores)
                for event_id, score in vector_scores:
                    _record_hit(hits, event_id, "vector", "vector_score", score)

    if not hits:
        return []

    hydrated: list[EventHit] = []
    for event_id, data in hits.items():
        hit = _hydrate_event(conn, event_id, data["matched_by"], data["scores"])
        if hit is None:
            continue
        if start_bound is not None and hit.timestamp < start_bound:
            continue
        if end_bound is not None and hit.timestamp > end_bound:
            continue
        hydrated.append(hit)

    if vector_door_fired or text_door_fired:
        # Unified v1 ranking (fusion is the OPEN/tunable area): agreement across
        # doors wins first, then per-door relevance, then a deterministic
        # recency-ish tiebreak. Entity-only hits use a neutral 0.0 vector score
        # so they are not buried beneath every vector hit.
        hydrated.sort(key=_rank_key)
    else:
        hydrated.sort(key=lambda hit: (hit.timestamp, hit.event_id), reverse=True)

    return hydrated[:limit]


def _rank_key(hit: EventHit) -> tuple[int, float, float, int]:
    return (
        -len(hit.matched_by),
        -hit.scores.get("vector_score", 0.0),
        hit.scores.get("fts_rank", float("inf")),
        -hit.event_id,
    )


def get_thread_detail(conn: sqlite3.Connection, thread_id: int) -> Episode | None:
    """Return one full episode with events in chronological order."""

    thread = conn.execute(
        "SELECT id, title, summary, created_at, updated_at FROM threads WHERE id = ?",
        (thread_id,),
    ).fetchone()
    if thread is None:
        return None

    rows = conn.execute(
        "SELECT e.id, e.content, e.timestamp, e.source "
        "FROM events e "
        "JOIN event_thread_edges ete ON e.id = ete.event_id "
        "WHERE ete.thread_id = ? "
        "ORDER BY e.timestamp ASC, e.id ASC",
        (thread_id,),
    ).fetchall()
    events = [
        EventHit(
            event_id=int(row["id"]),
            thread_id=thread_id,
            content=row["content"],
            timestamp=row["timestamp"],
            source=row["source"],
            entities=_event_entities(conn, int(row["id"])),
            matched_by=[],
        )
        for row in rows
    ]
    return Episode(
        thread_id=int(thread["id"]),
        title=thread["title"],
        summary=thread["summary"],
        created_at=thread["created_at"],
        updated_at=thread["updated_at"],
        events=events,
    )


def recent(
    conn: sqlite3.Connection,
    start: str | None = None,
    end: str | None = None,
    limit: int = config.RETRIEVAL_DEFAULT_LIMIT,
) -> list[Episode]:
    """Return episodes active within a time window, newest first."""

    limit = max(int(limit), 0)
    if limit == 0:
        return []

    clauses: list[str] = []
    params: list[object] = []
    start_bound, end_bound = normalize_time_window(start, end)
    if start_bound:
        clauses.append("updated_at >= ?")
        params.append(start_bound)
    if end_bound:
        clauses.append("updated_at <= ?")
        params.append(end_bound)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.append(limit)

    rows = conn.execute(
        f"SELECT id FROM threads {where}ORDER BY updated_at DESC, id DESC LIMIT ?",
        params,
    ).fetchall()
    episodes: list[Episode] = []
    for row in rows:
        episode = get_thread_detail(conn, int(row["id"]))
        if episode is not None:
            episodes.append(episode)
    return episodes


def _cap_episode_events(events: list[EventHit], hit_ids: set[int]) -> list[EventHit]:
    """Trim an episode to MAX_EVENTS_PER_EPISODE, keeping matched events first.

    The events that matched the query carry the evidence, so they are never
    dropped; leftover slots go to the thread's other events in chronological
    order. Deterministic: same inputs always yield the same trimmed set.
    """

    cap = config.MAX_EVENTS_PER_EPISODE
    if cap <= 0 or len(events) <= cap:
        return events
    matched = [event for event in events if event.event_id in hit_ids]
    others = [event for event in events if event.event_id not in hit_ids]
    keep = matched[:cap]
    keep.extend(others[: max(0, cap - len(keep))])
    return sorted(keep, key=lambda event: (event.timestamp, event.event_id))


def group_into_episodes(conn: sqlite3.Connection, hits: list[EventHit]) -> list[Episode]:
    """Group matching events into full episodes for handoff.

    Each episode is capped at MAX_EVENTS_PER_EPISODE events so the context handed
    to the caller stays bounded without discarding the matched evidence.
    """

    hit_ids = {hit.event_id for hit in hits}
    grouped: dict[int, dict] = {}
    order: list[int] = []
    for hit in hits:
        if hit.thread_id is None:
            continue
        if hit.thread_id not in grouped:
            grouped[hit.thread_id] = {
                "matched_by": set(),
                "best_score": 0.0,
                "most_recent_match": hit.timestamp,
            }
            order.append(hit.thread_id)
        grouped[hit.thread_id]["matched_by"].update(hit.matched_by)
        grouped[hit.thread_id]["best_score"] = max(
            grouped[hit.thread_id]["best_score"],
            _best_hit_score(hit),
        )
        if hit.timestamp > grouped[hit.thread_id]["most_recent_match"]:
            grouped[hit.thread_id]["most_recent_match"] = hit.timestamp

    episodes: list[Episode] = []
    for thread_id in order:
        episode = get_thread_detail(conn, thread_id)
        if episode is None:
            continue
        episode.events = _cap_episode_events(episode.events, hit_ids)
        episode.matched_by = sorted(grouped[thread_id]["matched_by"])
        episode.best_score = grouped[thread_id]["best_score"]
        episodes.append(episode)
    return episodes


def _entity_event_ids(conn: sqlite3.Connection, raw_entity: str) -> set[int]:
    normalized = normalize(raw_entity)
    if not normalized:
        return set()

    entity_ids: set[int] = set()
    row = conn.execute(
        "SELECT id FROM entities WHERE normalized_form = ?",
        (normalized,),
    ).fetchone()
    if row is not None:
        entity_ids.add(int(row["id"]))

    alias_rows = conn.execute(
        "SELECT entity_id FROM entity_aliases WHERE normalized_form = ?",
        (normalized,),
    ).fetchall()
    entity_ids.update(int(row["entity_id"]) for row in alias_rows)

    if not entity_ids:
        return set()
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        f"SELECT event_id FROM event_entity_edges WHERE entity_id IN ({placeholders})",
        sorted(entity_ids),
    ).fetchall()
    return {int(row["event_id"]) for row in rows}


def _text_event_scores(
    conn: sqlite3.Connection,
    text: str,
    limit: int,
    start_bound: str | None = None,
    end_bound: str | None = None,
) -> list[tuple[int, float]]:
    match_expr = _sanitize_fts_query(text)
    if not match_expr:
        return []

    # Filter by the time window inside SQL so the top-`limit` are the top matches
    # WITHIN the window, not the top matches overall (which may all be outside).
    clauses = ["events_fts MATCH ?"]
    params: list[object] = [match_expr]
    if start_bound is not None:
        clauses.append("e.timestamp >= ?")
        params.append(start_bound)
    if end_bound is not None:
        clauses.append("e.timestamp <= ?")
        params.append(end_bound)
    params.append(limit)

    query = (
        "SELECT e.id AS event_id, events_fts.rank AS rank "
        "FROM events_fts "
        "JOIN events e ON e.id = events_fts.rowid "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY events_fts.rank LIMIT ?"
    )
    try:
        rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("FTS retrieval unavailable for query %r: %s", text, exc)
        return []
    return [(int(row["event_id"]), float(row["rank"])) for row in rows]


def _vector_event_scores(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int,
) -> list[tuple[int, float]]:
    query_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    try:
        rows = conn.execute(
            # vec0 permits ONLY a single `ORDER BY distance`. Final ordering is
            # made deterministic downstream by `_rank_key` (which tiebreaks on
            # event_id), so no secondary key is needed here.
            "SELECT event_id "
            "FROM event_embeddings "
            "WHERE embedding MATCH ? "
            "  AND k = ? "
            "ORDER BY distance",
            (query_blob, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not _is_expected_vector_unavailable(exc):
            raise
        logger.warning("Vector retrieval unavailable: %s", exc)
        return []

    scored: list[tuple[int, float]] = []
    for row in rows:
        event_id = int(row["event_id"])
        embedding = get_event_embedding(conn, event_id)
        if embedding is None:
            continue
        scored.append((event_id, cosine_similarity(query_embedding, embedding)))
    return scored


def _embed_query(
    embedding_model: EmbeddingInterface,
    text: str,
) -> list[float] | None:
    try:
        return embedding_model.embed(text)
    except ModelUnavailableError as exc:
        logger.warning("Embedding model unavailable during retrieval: %s", exc)
        return None


def _record_hit(
    hits: dict[int, dict],
    event_id: int,
    matched_by: str,
    score_name: str | None = None,
    score_value: float | None = None,
) -> None:
    data = hits.setdefault(event_id, {"matched_by": set(), "scores": {}})
    data["matched_by"].add(matched_by)
    if score_name is not None and score_value is not None:
        data["scores"][score_name] = score_value


def _hydrate_event(
    conn: sqlite3.Connection,
    event_id: int,
    matched_by: set[str],
    scores: dict[str, float],
) -> EventHit | None:
    row = conn.execute(
        "SELECT id, content, timestamp, source FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        return None
    edge = conn.execute(
        "SELECT thread_id FROM event_thread_edges WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    return EventHit(
        event_id=int(row["id"]),
        thread_id=int(edge["thread_id"]) if edge is not None else None,
        content=row["content"],
        timestamp=row["timestamp"],
        source=row["source"],
        entities=_event_entities(conn, event_id),
        matched_by=sorted(matched_by),
        scores=dict(scores),
    )


def _event_entities(conn: sqlite3.Connection, event_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT en.canonical_name "
        "FROM event_entity_edges ee "
        "JOIN entities en ON ee.entity_id = en.id "
        "WHERE ee.event_id = ? "
        "ORDER BY en.canonical_name",
        (event_id,),
    ).fetchall()
    return [row["canonical_name"] for row in rows]


def _best_hit_score(hit: EventHit) -> float:
    if "vector_score" in hit.scores:
        return hit.scores["vector_score"]
    if "fts_rank" in hit.scores:
        return -hit.scores["fts_rank"]
    return 0.0


def _is_expected_vector_unavailable(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return (
        "no such table: event_embeddings" in message
        or "no such module: vec0" in message
        or "no query solution" in message
        or "unable to use function match" in message
    )
