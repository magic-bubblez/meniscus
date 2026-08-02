"""Deterministic retrieval over distilled facts."""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from meniscus import config
from meniscus.embedding_interface import EmbeddingInterface
from meniscus.exceptions import ModelUnavailableError
from meniscus.time_bounds import normalize_time_window
from meniscus.tokens import estimate_tokens
from meniscus.vocab_reconciliation import normalize

logger = logging.getLogger(__name__)

_FTS_TOKEN_RE = re.compile(r"\w+")
_VECTOR_MAX_K = 500


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)

KIND_FACT: str = "fact"
KIND_EVENT: str = "event"


@dataclass
class Fact:
    """One fact packaged for handoff, with its source event for provenance."""

    fact_id: int
    event_id: int
    text: str
    timestamp: str
    source: str
    entities: list[str] = field(default_factory=list)
    matched: bool = False
    matched_by: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    kind: str = KIND_FACT


@dataclass
class EventUnit:
    """A raw source event standing in for its facts when they cost as much as it does."""

    event_id: int
    text: str
    timestamp: str
    source: str
    kind: str = KIND_EVENT


@dataclass
class EpisodeEvent:
    """One event within a reconstructed episode, carrying its distilled facts."""

    event_id: int
    timestamp: str
    source: str
    content: str
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Episode:
    """A contiguous session-segment of events around an anchor, chronological."""

    anchor_timestamp: str
    start: str
    end: str
    event_count: int
    events: list[EpisodeEvent] = field(default_factory=list)


def retrieve(
    conn: sqlite3.Connection,
    text: str | None = None,
    entity: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = config.RETRIEVAL_DEFAULT_LIMIT,
    embedding_model: EmbeddingInterface | None = None,
) -> list[Fact | EventUnit]:
    """Return matched facts plus their scored neighbours, chronological (source-collapsed)."""

    query_embedding = _embed_query(embedding_model, text) if text else None
    matched = search_facts(conn, text, entity, start, end, limit, query_embedding)
    if not matched:
        return []

    neighbours = _neighbours(conn, matched, query_embedding) if config.NEIGHBOUR_ENABLED else []

    merged: dict[int, Fact] = {}
    for fact in matched:
        fact.matched = True
        merged[fact.fact_id] = fact
    for fact in neighbours:
        merged.setdefault(fact.fact_id, fact)

    ordered = sorted(merged.values(), key=lambda f: (f.timestamp, f.fact_id))
    if not config.SOURCE_COLLAPSE:
        return ordered
    return _collapse_by_source(conn, ordered)


def _collapse_by_source(
    conn: sqlite3.Connection,
    facts: list[Fact],
) -> list[Fact | EventUnit]:
    """Per event, replace its selected facts with the raw event when they cost as much."""

    by_event: dict[int, list[Fact]] = {}
    for fact in facts:
        by_event.setdefault(fact.event_id, []).append(fact)

    output: list[Fact | EventUnit] = []
    for event_id, event_facts in by_event.items():
        fact_tokens = sum(estimate_tokens(f.text) for f in event_facts)
        row = conn.execute(
            "SELECT content, timestamp, source FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is not None and fact_tokens >= estimate_tokens(row["content"]):
            output.append(
                EventUnit(
                    event_id=event_id,
                    text=row["content"],
                    timestamp=row["timestamp"],
                    source=row["source"],
                )
            )
        else:
            output.extend(event_facts)

    output.sort(key=_unit_sort_key)
    return output


def _unit_sort_key(item: Fact | EventUnit) -> tuple[str, int, int]:
    if item.kind == KIND_EVENT:
        return (item.timestamp, item.event_id, -1)
    return (item.timestamp, item.event_id, item.fact_id)


def search_facts(
    conn: sqlite3.Connection,
    text: str | None = None,
    entity: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = config.RETRIEVAL_DEFAULT_LIMIT,
    query_embedding: list[float] | None = None,
) -> list[Fact]:
    """Find facts matching a topic, optionally within a time window."""

    limit = max(int(limit), 0)
    if limit == 0:
        return []

    start_bound, end_bound = normalize_time_window(start, end)
    fetch_k = min(max(limit, config.CANDIDATE_POOL), _VECTOR_MAX_K)

    hits: dict[int, dict] = {}

    if entity:
        for fact_id in _entity_fact_ids(conn, entity):
            _record_hit(hits, fact_id, "entity")

    if text:
        for position, (fact_id, rank) in enumerate(
            _text_fact_scores(conn, text, fetch_k, start_bound, end_bound)
        ):
            _record_hit(hits, fact_id, "text", fts_rank=rank, fts_pos=float(position))
        if query_embedding is not None:
            for position, (fact_id, score) in enumerate(
                _vector_fact_scores(conn, query_embedding, fetch_k)
            ):
                _record_hit(hits, fact_id, "vector", vector_score=score, vector_pos=float(position))

    if not hits:
        return []

    hydrated: list[Fact] = []
    for fact_id, data in hits.items():
        fact = _hydrate_fact(conn, fact_id, data["matched_by"], data["scores"])
        if fact is None:
            continue
        if start_bound is not None and fact.timestamp < start_bound:
            continue
        if end_bound is not None and fact.timestamp > end_bound:
            continue
        fact.scores["fused"] = _fused_score(fact.scores, fact.matched_by)
        hydrated.append(fact)

    hydrated.sort(key=lambda f: (f.scores["fused"], f.timestamp, f.fact_id), reverse=True)
    return _adaptive_cut(hydrated, _is_enumeration(text), limit)


def recent_facts(
    conn: sqlite3.Connection,
    start: str | None = None,
    end: str | None = None,
    limit: int = config.RETRIEVAL_DEFAULT_LIMIT,
) -> list[Fact]:
    """Return facts active within a time window, newest first."""

    limit = max(int(limit), 0)
    if limit == 0:
        return []

    clauses: list[str] = []
    params: list[object] = []
    start_bound, end_bound = normalize_time_window(start, end)
    if start_bound is not None:
        clauses.append("e.timestamp >= ?")
        params.append(start_bound)
    if end_bound is not None:
        clauses.append("e.timestamp <= ?")
        params.append(end_bound)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.append(limit)

    rows = conn.execute(
        "SELECT f.id FROM facts f JOIN events e ON e.id = f.event_id "
        f"{where}ORDER BY e.timestamp DESC, f.id DESC LIMIT ?",
        params,
    ).fetchall()
    facts = [_hydrate_fact(conn, int(r["id"]), set(), {}) for r in rows]
    return [f for f in facts if f is not None]


def get_event(conn: sqlite3.Connection, event_id: int) -> dict | None:
    """Return one raw source event with its facts (provenance handoff)."""

    event = conn.execute(
        "SELECT id, content, timestamp, source FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if event is None:
        return None
    fact_rows = conn.execute(
        "SELECT id FROM facts WHERE event_id = ? ORDER BY position ASC",
        (event_id,),
    ).fetchall()
    facts = [_hydrate_fact(conn, int(r["id"]), set(), {}) for r in fact_rows]
    return {
        "event_id": int(event["id"]),
        "content": event["content"],
        "timestamp": event["timestamp"],
        "source": event["source"],
        "facts": [f for f in facts if f is not None],
    }


def episode(
    conn: sqlite3.Connection,
    anchor_text: str | None = None,
    anchor_event_id: int | None = None,
    at: str | None = None,
    window_seconds: int = config.EPISODE_WINDOW_SECONDS,
    group: bool = True,
    max_events: int = config.EPISODE_MAX_EVENTS,
    embedding_model: EmbeddingInterface | None = None,
) -> list[Episode]:
    """Reconstruct the session-segments around an anchor, chronological, with facts."""

    anchor = _resolve_anchor(conn, anchor_text, anchor_event_id, at, embedding_model)
    if anchor is None:
        return []
    anchor_dt = datetime.fromisoformat(anchor)
    low = (anchor_dt - timedelta(seconds=window_seconds)).isoformat()
    high = (anchor_dt + timedelta(seconds=window_seconds)).isoformat()
    rows = conn.execute(
        "SELECT id, content, timestamp, source FROM events "
        "WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC, id ASC",
        (low, high),
    ).fetchall()
    if not rows:
        return []

    segments = _segment_by_gap(rows, config.SESSION_GAP_SECONDS)
    if group:
        segments = [_anchor_segment(segments, anchor_dt)]

    episodes: list[Episode] = []
    for segment in segments:
        kept = _cap_segment(segment, anchor_dt, max_events)
        events = [_episode_event(conn, row) for row in kept]
        episodes.append(
            Episode(
                anchor_timestamp=anchor,
                start=kept[0]["timestamp"],
                end=kept[-1]["timestamp"],
                event_count=len(events),
                events=events,
            )
        )
    return episodes


def _resolve_anchor(
    conn: sqlite3.Connection,
    anchor_text: str | None,
    anchor_event_id: int | None,
    at: str | None,
    embedding_model: EmbeddingInterface | None,
) -> str | None:
    if at is not None:
        start, _ = normalize_time_window(at, None)
        return start
    if anchor_event_id is not None:
        row = conn.execute(
            "SELECT timestamp FROM events WHERE id = ?", (anchor_event_id,)
        ).fetchone()
        return row["timestamp"] if row is not None else None
    if anchor_text:
        query_embedding = _embed_query(embedding_model, anchor_text)
        matched = search_facts(conn, text=anchor_text, limit=1, query_embedding=query_embedding)
        return matched[0].timestamp if matched else None
    return None


def _segment_by_gap(rows: list, gap_seconds: int) -> list[list]:
    segments: list[list] = []
    current: list = []
    previous: datetime | None = None
    for row in rows:
        moment = datetime.fromisoformat(row["timestamp"])
        if previous is not None and (moment - previous).total_seconds() > gap_seconds:
            segments.append(current)
            current = []
        current.append(row)
        previous = moment
    if current:
        segments.append(current)
    return segments


def _anchor_segment(segments: list[list], anchor_dt: datetime) -> list:
    anchor = anchor_dt.isoformat()
    for segment in segments:
        if segment[0]["timestamp"] <= anchor <= segment[-1]["timestamp"]:
            return segment
    return min(segments, key=lambda s: _segment_distance(s, anchor_dt))


def _segment_distance(segment: list, anchor_dt: datetime) -> float:
    first = abs((datetime.fromisoformat(segment[0]["timestamp"]) - anchor_dt).total_seconds())
    last = abs((datetime.fromisoformat(segment[-1]["timestamp"]) - anchor_dt).total_seconds())
    return min(first, last)


def _cap_segment(segment: list, anchor_dt: datetime, max_events: int) -> list:
    if len(segment) <= max_events:
        return segment
    nearest = sorted(
        segment,
        key=lambda r: abs((datetime.fromisoformat(r["timestamp"]) - anchor_dt).total_seconds()),
    )[:max_events]
    return sorted(nearest, key=lambda r: (r["timestamp"], r["id"]))


def _episode_event(conn: sqlite3.Connection, row) -> EpisodeEvent:
    fact_rows = conn.execute(
        "SELECT id FROM facts WHERE event_id = ? ORDER BY position ASC", (row["id"],)
    ).fetchall()
    facts = [_hydrate_fact(conn, int(r["id"]), set(), {}) for r in fact_rows]
    return EpisodeEvent(
        event_id=int(row["id"]),
        timestamp=row["timestamp"],
        source=row["source"],
        content=row["content"],
        facts=[f for f in facts if f is not None],
    )


def _neighbours(
    conn: sqlite3.Connection,
    matched: list[Fact],
    query_embedding: list[float] | None,
) -> list[Fact]:
    """Gather facts near the matched set (shared entity / topical / temporal), cut."""

    matched_ids = {f.fact_id for f in matched}
    matched_entity_ids = _entity_ids_for_facts(conn, matched_ids)

    candidates: set[int] = set()
    if matched_entity_ids:
        candidates |= _facts_for_entity_ids(conn, matched_entity_ids)
    if query_embedding is not None:
        for fact_id, _score in _vector_fact_scores(conn, query_embedding, config.VECTOR_CANDIDATE_K):
            candidates.add(fact_id)
    candidates -= matched_ids
    if not candidates:
        return []

    idf = _fact_idf(conn, matched_entity_ids)
    matched_idf_sum = sum(idf.get(eid, 0.0) for eid in matched_entity_ids)

    scored: list[Fact] = []
    for fact_id in candidates:
        fact = _hydrate_fact(conn, fact_id, set(), {})
        if fact is None:
            continue
        score = _neighbour_score(
            conn, fact, matched_entity_ids, matched_idf_sum, idf, query_embedding
        )
        if score <= 0.0:
            continue
        fact.scores["neighbour"] = score
        scored.append(fact)

    scored.sort(key=lambda f: (f.scores["neighbour"], f.timestamp, f.fact_id), reverse=True)
    return _apply_cut(scored)


def _neighbour_score(
    conn: sqlite3.Connection,
    fact: Fact,
    matched_entity_ids: set[int],
    matched_idf_sum: float,
    idf: dict[int, float],
    query_embedding: list[float] | None,
) -> float:
    """topical relevance: idf-containment blended with cosine (no temporal decay)."""

    fact_entity_ids = _fact_entity_ids(conn, fact.fact_id)
    shared = fact_entity_ids & matched_entity_ids
    idf_for = _fact_idf(conn, fact_entity_ids)
    fact_idf_sum = sum(idf_for.get(eid, 0.0) for eid in fact_entity_ids)
    shared_idf_sum = sum(idf.get(eid, idf_for.get(eid, 0.0)) for eid in shared)
    containment = shared_idf_sum / fact_idf_sum if fact_idf_sum > 0 else 0.0

    cosine_val: float | None = None
    if query_embedding is not None:
        fact_embedding = _fact_embedding(conn, fact.fact_id)
        if fact_embedding is not None:
            cosine_val = cosine_similarity(query_embedding, fact_embedding)

    if cosine_val is None:
        return containment
    return config.HYBRID_ALPHA * containment + (1 - config.HYBRID_ALPHA) * cosine_val


def _apply_cut(scored: list[Fact]) -> list[Fact]:
    """Keep the relevant neighbours by the configured cut rule, bounded."""

    if not scored:
        return []

    if config.NEIGHBOUR_CUT_RULE == "gap":
        kept = _gap_cut(scored)
    else:
        top = scored[0].scores["neighbour"]
        threshold = config.NEIGHBOUR_CUT_RATIO * top
        kept = [f for f in scored if f.scores["neighbour"] >= threshold]

    return kept[: config.NEIGHBOUR_MAX]


def _gap_cut(scored: list[Fact]) -> list[Fact]:
    values = [f.scores["neighbour"] for f in scored]
    cut = len(scored)
    biggest_drop = -1.0
    for i in range(len(values) - 1):
        drop = values[i] - values[i + 1]
        if drop > biggest_drop:
            biggest_drop = drop
            cut = i + 1
    return scored[:cut]


def _sanitize_fts_query(raw: str) -> str:
    return " OR ".join(f'"{token}"' for token in _FTS_TOKEN_RE.findall(raw))


def _entity_fact_ids(conn: sqlite3.Connection, raw_entity: str) -> set[int]:
    entity_ids = _resolve_entity_ids(conn, raw_entity)
    if not entity_ids:
        return set()
    return _facts_for_entity_ids(conn, entity_ids)


def _resolve_entity_ids(conn: sqlite3.Connection, raw_entity: str) -> set[int]:
    normalized = normalize(raw_entity)
    if not normalized:
        return set()
    entity_ids: set[int] = set()
    row = conn.execute(
        "SELECT id FROM entities WHERE normalized_form = ?", (normalized,)
    ).fetchone()
    if row is not None:
        entity_ids.add(int(row["id"]))
    for alias_row in conn.execute(
        "SELECT entity_id FROM entity_aliases WHERE normalized_form = ?", (normalized,)
    ).fetchall():
        entity_ids.add(int(alias_row["entity_id"]))
    return entity_ids


def _facts_for_entity_ids(conn: sqlite3.Connection, entity_ids: set[int]) -> set[int]:
    if not entity_ids:
        return set()
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        f"SELECT DISTINCT fact_id FROM fact_entity_edges WHERE entity_id IN ({placeholders})",
        sorted(entity_ids),
    ).fetchall()
    return {int(row["fact_id"]) for row in rows}


def _entity_ids_for_facts(conn: sqlite3.Connection, fact_ids: set[int]) -> set[int]:
    if not fact_ids:
        return set()
    placeholders = ",".join("?" for _ in fact_ids)
    rows = conn.execute(
        f"SELECT DISTINCT entity_id FROM fact_entity_edges WHERE fact_id IN ({placeholders})",
        sorted(fact_ids),
    ).fetchall()
    return {int(row["entity_id"]) for row in rows}


def _fact_entity_ids(conn: sqlite3.Connection, fact_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT entity_id FROM fact_entity_edges WHERE fact_id = ?", (fact_id,)
    ).fetchall()
    return {int(row["entity_id"]) for row in rows}


def _fact_idf(conn: sqlite3.Connection, entity_ids: set[int]) -> dict[int, float]:
    if not entity_ids:
        return {}
    n = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    weights: dict[int, float] = {}
    for entity_id in entity_ids:
        df = conn.execute(
            "SELECT COUNT(DISTINCT fact_id) FROM fact_entity_edges WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()[0]
        weights[entity_id] = math.log((n + 1) / (df + 1)) + 1
    return weights


def _text_fact_scores(
    conn: sqlite3.Connection,
    text: str,
    limit: int,
    start_bound: str | None,
    end_bound: str | None,
) -> list[tuple[int, float]]:
    match_expr = _sanitize_fts_query(text)
    if not match_expr:
        return []
    clauses = ["facts_fts MATCH ?"]
    params: list[object] = [match_expr]
    if start_bound is not None:
        clauses.append("e.timestamp >= ?")
        params.append(start_bound)
    if end_bound is not None:
        clauses.append("e.timestamp <= ?")
        params.append(end_bound)
    params.append(limit)
    query = (
        "SELECT f.id AS fact_id, facts_fts.rank AS rank "
        "FROM facts_fts "
        "JOIN facts f ON f.id = facts_fts.rowid "
        "JOIN events e ON e.id = f.event_id "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY facts_fts.rank LIMIT ?"
    )
    try:
        rows = conn.execute(query, params).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("FTS retrieval unavailable for query %r: %s", text, exc)
        return []
    return [(int(row["fact_id"]), float(row["rank"])) for row in rows]


def _vector_fact_scores(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int,
) -> list[tuple[int, float]]:
    query_blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    try:
        rows = conn.execute(
            "SELECT fact_id FROM fact_embeddings "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (query_blob, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not _is_expected_vector_unavailable(exc):
            raise
        logger.warning("Vector retrieval unavailable: %s", exc)
        return []
    scored: list[tuple[int, float]] = []
    for row in rows:
        fact_id = int(row["fact_id"])
        embedding = _fact_embedding(conn, fact_id)
        if embedding is None:
            continue
        scored.append((fact_id, cosine_similarity(query_embedding, embedding)))
    return scored


def _fact_embedding(conn: sqlite3.Connection, fact_id: int) -> list[float] | None:
    try:
        row = conn.execute(
            "SELECT embedding FROM fact_embeddings WHERE fact_id = ?", (fact_id,)
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if not _is_expected_vector_unavailable(exc):
            raise
        return None
    if row is None:
        return None
    blob = bytes(row[0])
    dim = len(blob) // 4
    return list(struct.unpack(f"{dim}f", blob))


def _embed_query(
    embedding_model: EmbeddingInterface | None,
    text: str | None,
) -> list[float] | None:
    if embedding_model is None or not text:
        return None
    try:
        return embedding_model.embed_query(text)
    except ModelUnavailableError as exc:
        logger.warning("Embedding model unavailable during retrieval: %s", exc)
        return None


def _record_hit(
    hits: dict[int, dict],
    fact_id: int,
    matched_by: str,
    **scores: float,
) -> None:
    data = hits.setdefault(fact_id, {"matched_by": set(), "scores": {}})
    data["matched_by"].add(matched_by)
    data["scores"].update(scores)


def _hydrate_fact(
    conn: sqlite3.Connection,
    fact_id: int,
    matched_by: set[str],
    scores: dict[str, float],
) -> Fact | None:
    row = conn.execute(
        "SELECT f.id, f.text, f.event_id, e.timestamp, e.source "
        "FROM facts f JOIN events e ON e.id = f.event_id WHERE f.id = ?",
        (fact_id,),
    ).fetchone()
    if row is None:
        return None
    entity_rows = conn.execute(
        "SELECT en.canonical_name FROM fact_entity_edges fe "
        "JOIN entities en ON en.id = fe.entity_id WHERE fe.fact_id = ? "
        "ORDER BY en.canonical_name",
        (fact_id,),
    ).fetchall()
    return Fact(
        fact_id=int(row["id"]),
        event_id=int(row["event_id"]),
        text=row["text"],
        timestamp=row["timestamp"],
        source=row["source"],
        entities=[r["canonical_name"] for r in entity_rows],
        matched_by=sorted(matched_by),
        scores=dict(scores),
    )


def _fused_score(scores: dict[str, float], matched_by: list[str]) -> float:
    """Magnitude noisy-OR: a strong single door can win on its own merit."""

    vector_term = min(max(scores.get("vector_score", 0.0), 0.0), 1.0)
    fts_term = config.RRF_K / (config.RRF_K + scores["fts_pos"]) if "fts_pos" in scores else 0.0
    entity_term = 1.0 if "entity" in matched_by else 0.0
    fused = 1.0 - (
        (1.0 - config.W_VECTOR * vector_term)
        * (1.0 - config.W_FTS * fts_term)
        * (1.0 - config.W_ENTITY * entity_term)
    )
    return fused + config.DOOR_PRIOR * len(matched_by)


def _is_enumeration(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(trigger in lowered for trigger in config.ENUMERATION_TRIGGERS)


def _adaptive_cut(sorted_facts: list[Fact], enumerate_intent: bool, cap: int) -> list[Fact]:
    """Cut the fused-score curve: whole plateau on enumeration, else ratio-of-top."""

    n = len(sorted_facts)
    if n == 0:
        return []
    if enumerate_intent:
        cut = n
    elif config.MATCHED_CUT_RULE == "gap":
        cut = _gap_cut_index([f.scores["fused"] for f in sorted_facts])
    else:
        threshold = config.MATCHED_CUT_RATIO * sorted_facts[0].scores["fused"]
        cut = sum(1 for f in sorted_facts if f.scores["fused"] >= threshold)
    cut = min(max(cut, config.MIN_RETURN), cap, n)
    return sorted_facts[:cut]


def _gap_cut_index(values: list[float]) -> int:
    cut = len(values)
    biggest_drop = -1.0
    for i in range(len(values) - 1):
        drop = values[i] - values[i + 1]
        if drop > biggest_drop:
            biggest_drop = drop
            cut = i + 1
    return cut


def _is_expected_vector_unavailable(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return (
        "no such table: fact_embeddings" in message
        or "no such module: vec0" in message
        or "no query solution" in message
        or "unable to use function match" in message
    )
