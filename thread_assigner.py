"""Deterministic event-to-thread assignment."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

from config import (
    ALGORITHM_VERSION,
    ASSIGNMENT_THRESHOLD,
    EMBEDDING_PROVIDER,
    HALF_LIFE_HOURS,
    HYBRID_ALPHA,
    TEMPORAL_FLOOR,
    TOPICAL_FLOOR,
    VECTOR_CANDIDATE_K,
)
from meniscus_types import DecisionType

logger = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """Scoring breakdown for one candidate thread."""

    thread_id: int
    containment: float
    cosine: float | None = None
    hybrid_topical: float = 0.0
    temporal: float = 0.0
    score: float = 0.0
    gated: str | None = None

    def to_log_dict(self) -> dict:
        return {
            "containment": round(self.containment, 6),
            "cosine": round(self.cosine, 6) if self.cosine is not None else None,
            "hybrid_topical": round(self.hybrid_topical, 6),
            "temporal": round(self.temporal, 6),
            "score": round(self.score, 6),
            "gated": self.gated,
        }


def compute_idf(conn: sqlite3.Connection, entity_ids: list[int]) -> dict[int, float]:
    """Compute smoothed IDF weights for event entities."""

    n = conn.execute(
        "SELECT COUNT(DISTINCT event_id) FROM event_entity_edges"
    ).fetchone()[0]
    weights: dict[int, float] = {}
    for entity_id in entity_ids:
        df = conn.execute(
            "SELECT COUNT(*) FROM event_entity_edges WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()[0]
        weights[entity_id] = math.log((n + 1) / (df + 1)) + 1
    return weights


def get_candidate_threads(conn: sqlite3.Connection, entity_ids: list[int]) -> list[int]:
    """Find threads sharing at least one entity with the event."""

    if not entity_ids:
        return []
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        "SELECT DISTINCT et.thread_id "
        "FROM event_entity_edges ee "
        "JOIN event_thread_edges et ON ee.event_id = et.event_id "
        f"WHERE ee.entity_id IN ({placeholders})",
        entity_ids,
    ).fetchall()
    return [int(row["thread_id"]) for row in rows]


def get_vector_candidates(
    conn: sqlite3.Connection,
    event_id: int,
    event_embedding: list[float] | None,
) -> set[int]:
    """Find threads containing nearest embedding-neighbor events."""

    if event_embedding is None:
        return set()

    query_blob = struct.pack(f"{len(event_embedding)}f", *event_embedding)
    try:
        rows = conn.execute(
            "SELECT DISTINCT et.thread_id "
            "FROM ( "
            "    SELECT event_id "
            "    FROM event_embeddings "
            "    WHERE embedding MATCH ? "
            "      AND k = ? "
            # vec0 permits ONLY a single `ORDER BY distance`; a secondary key is
            # rejected. The result feeds a set of thread ids (order-independent),
            # and exact float-distance ties are vanishingly rare.
            "    ORDER BY distance "
            ") nn "
            "JOIN event_thread_edges et ON nn.event_id = et.event_id "
            "WHERE nn.event_id != ?",
            (query_blob, VECTOR_CANDIDATE_K, event_id),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if not _is_expected_vec_unavailable(exc):
            raise
        logger.warning(
            "Vector candidate lookup unavailable for event %s: %s",
            event_id,
            exc,
        )
        return set()
    return {int(row["thread_id"]) for row in rows}


def get_thread_entity_ids(conn: sqlite3.Connection, thread_id: int) -> set[int]:
    """Get distinct entity IDs associated with a thread."""

    rows = conn.execute(
        "SELECT DISTINCT ee.entity_id "
        "FROM event_entity_edges ee "
        "JOIN event_thread_edges et ON ee.event_id = et.event_id "
        "WHERE et.thread_id = ?",
        (thread_id,),
    ).fetchall()
    return {int(row["entity_id"]) for row in rows}


def compute_hours_between(timestamp_a: str, timestamp_b: str) -> float:
    """Compute absolute hours between two ISO 8601 timestamps."""

    dt_a = _parse_timestamp(timestamp_a)
    dt_b = _parse_timestamp(timestamp_b)
    return abs((dt_b - dt_a).total_seconds()) / 3600.0


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def get_event_embedding(conn: sqlite3.Connection, event_id: int) -> list[float] | None:
    """Fetch and deserialize an event embedding."""

    try:
        row = conn.execute(
            "SELECT embedding FROM event_embeddings WHERE event_id = ?",
            (event_id,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if not _is_expected_vec_unavailable(exc):
            raise
        logger.warning("event_embeddings unavailable for event %s: %s", event_id, exc)
        return None

    if row is None:
        return None
    blob = bytes(row[0])
    dim = len(blob) // 4
    return list(struct.unpack(f"{dim}f", blob))


def get_thread_centroid(conn: sqlite3.Connection, thread_id: int) -> list[float] | None:
    """Compute the average embedding for all embedded events in a thread."""

    event_rows = conn.execute(
        "SELECT event_id FROM event_thread_edges WHERE thread_id = ?",
        (thread_id,),
    ).fetchall()
    if not event_rows:
        return None

    embeddings: list[list[float]] = []
    for row in event_rows:
        embedding = get_event_embedding(conn, int(row["event_id"]))
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return None

    dim = len(embeddings[0])
    return [
        sum(embedding[i] for embedding in embeddings) / len(embeddings)
        for i in range(dim)
    ]


def score_candidate(
    conn: sqlite3.Connection,
    event_entity_ids: set[int],
    event_embedding: list[float] | None,
    event_timestamp: str,
    thread_id: int,
    idf_weights: dict[int, float],
) -> CandidateScore:
    """Score one candidate thread against an event."""

    thread_entity_ids = get_thread_entity_ids(conn, thread_id)
    shared = event_entity_ids & thread_entity_ids

    shared_idf_sum = sum(idf_weights.get(entity_id, 0.0) for entity_id in shared)
    event_idf_sum = sum(idf_weights.get(entity_id, 0.0) for entity_id in event_entity_ids)
    containment = shared_idf_sum / event_idf_sum if event_idf_sum > 0 else 0.0

    cosine_val: float | None = None
    if event_embedding is not None:
        thread_centroid = get_thread_centroid(conn, thread_id)
        if thread_centroid is not None:
            cosine_val = cosine_similarity(event_embedding, thread_centroid)

    if cosine_val is None:
        hybrid_topical = containment
    else:
        hybrid_topical = HYBRID_ALPHA * containment + (1 - HYBRID_ALPHA) * cosine_val

    if hybrid_topical < TOPICAL_FLOOR:
        return CandidateScore(
            thread_id=thread_id,
            containment=containment,
            cosine=cosine_val,
            hybrid_topical=hybrid_topical,
            gated="topical",
        )

    thread_row = conn.execute(
        "SELECT updated_at FROM threads WHERE id = ?",
        (thread_id,),
    ).fetchone()
    thread_updated_at = thread_row["updated_at"]
    hours = compute_hours_between(thread_updated_at, event_timestamp)
    temporal = 1.0 / (1.0 + hours / HALF_LIFE_HOURS)

    if temporal < TEMPORAL_FLOOR:
        return CandidateScore(
            thread_id=thread_id,
            containment=containment,
            cosine=cosine_val,
            hybrid_topical=hybrid_topical,
            temporal=temporal,
            gated="temporal",
        )

    score = hybrid_topical * temporal
    return CandidateScore(
        thread_id=thread_id,
        containment=containment,
        cosine=cosine_val,
        hybrid_topical=hybrid_topical,
        temporal=temporal,
        score=score,
    )


def assign_thread(conn: sqlite3.Connection, event_id: int, entity_ids: list[int]) -> int:
    """Assign an event to a new or existing thread."""

    event_row = conn.execute(
        "SELECT timestamp FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if event_row is None:
        raise ValueError(f"Event {event_id} not found")
    event_timestamp = event_row["timestamp"]
    # In disabled-embeddings mode there is no vector table, so skip the lookup
    # entirely rather than querying a missing table (and logging noise) per event.
    event_embedding = (
        None if EMBEDDING_PROVIDER == "none" else get_event_embedding(conn, event_id)
    )
    event_entity_ids = set(entity_ids)

    entity_candidates = set(get_candidate_threads(conn, entity_ids))
    vector_candidates = get_vector_candidates(conn, event_id, event_embedding)
    candidate_thread_ids = entity_candidates | vector_candidates

    if not candidate_thread_ids:
        thread_id = _create_new_thread(conn, event_id, event_timestamp)
        _log_assignment(
            conn,
            event_id=event_id,
            thread_id=thread_id,
            candidate_scores={},
            entity_ids=entity_ids,
            decision_type=DecisionType.NEW_THREAD,
        )
        return thread_id

    idf_weights = compute_idf(conn, entity_ids)
    scored: dict[int, CandidateScore] = {}
    for thread_id in sorted(candidate_thread_ids):
        scored[thread_id] = score_candidate(
            conn,
            event_entity_ids,
            event_embedding,
            event_timestamp,
            thread_id,
            idf_weights,
        )

    best: CandidateScore | None = None
    # Ascending thread_id order + strict `>` means an exact score tie keeps the
    # first (lowest) thread_id — the documented tie-break, without a dead branch.
    for thread_id in sorted(scored):
        candidate = scored[thread_id]
        if candidate.gated is not None or candidate.score <= ASSIGNMENT_THRESHOLD:
            continue
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        thread_id = _create_new_thread(conn, event_id, event_timestamp)
        decision_type = DecisionType.NEW_THREAD
    else:
        thread_id = best.thread_id
        conn.execute(
            "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
            (event_id, thread_id),
        )
        conn.execute(
            "UPDATE threads SET updated_at = MAX(updated_at, ?) WHERE id = ?",
            (event_timestamp, thread_id),
        )
        decision_type = DecisionType.EXISTING_THREAD

    _log_assignment(
        conn,
        event_id=event_id,
        thread_id=thread_id,
        candidate_scores=scored,
        entity_ids=entity_ids,
        decision_type=decision_type,
    )
    return thread_id


def _create_new_thread(
    conn: sqlite3.Connection,
    event_id: int,
    event_timestamp: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO threads (title, summary, created_at, updated_at) "
        "VALUES ('', '', ?, ?)",
        (event_timestamp, event_timestamp),
    )
    thread_id = int(cursor.lastrowid)
    conn.execute(
        "INSERT INTO event_thread_edges (event_id, thread_id) VALUES (?, ?)",
        (event_id, thread_id),
    )
    return thread_id


def _log_assignment(
    conn: sqlite3.Connection,
    event_id: int,
    thread_id: int,
    candidate_scores: dict[int, CandidateScore],
    entity_ids: list[int],
    decision_type: DecisionType,
) -> None:
    scores_json = json.dumps(
        {str(tid): score.to_log_dict() for tid, score in candidate_scores.items()}
    )
    conn.execute(
        "INSERT INTO assignment_log "
        "(event_id, assigned_thread_id, candidate_scores, threshold, half_life, "
        "algorithm_version, entity_snapshot, decision_type, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            thread_id,
            scores_json,
            ASSIGNMENT_THRESHOLD,
            HALF_LIFE_HOURS,
            ALGORITHM_VERSION,
            json.dumps(entity_ids),
            decision_type,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_expected_vec_unavailable(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    expected = [
        "no such table: event_embeddings",
        "no such module: vec0",
        "unable to use function match",
    ]
    return any(fragment in message for fragment in expected)
