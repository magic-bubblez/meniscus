"""Pipeline orchestration across intake, semantic processing, and threading."""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import transactional
from embedding_interface import EmbeddingInterface
from entity_extractor import extract_entities
from event_intake import ingest_event
from exceptions import EmbeddingBackendUnavailableError, ModelUnavailableError
from model_interface import ModelInterface
from thread_assigner import assign_thread
from thread_summarizer import summarize_thread
from meniscus_types import ExtractionStatus
from vocab_reconciliation import reconcile_entities

logger = logging.getLogger(__name__)


def process_event(
    conn: sqlite3.Connection,
    event_id: int,
    model: ModelInterface,
    embedding_model: EmbeddingInterface | None,
) -> int:
    """Process a pending event through semantic processing, assignment, summary."""

    extraction_result = extract_entities(conn, event_id, model)
    embedding = _compute_embedding(conn, event_id, embedding_model)

    with transactional(conn) as txn:
        entity_ids = reconcile_entities(txn, event_id, extraction_result)
        if embedding is not None:
            _store_embedding(txn, event_id, embedding)
        txn.execute(
            "UPDATE events SET extraction_status = ? WHERE id = ?",
            (ExtractionStatus.COMPLETED, event_id),
        )

    with transactional(conn) as txn:
        thread_id = assign_thread(txn, event_id, entity_ids)

    summarize_thread(conn, thread_id, model)
    return thread_id


def _compute_embedding(
    conn: sqlite3.Connection,
    event_id: int,
    embedding_model: EmbeddingInterface | None,
) -> list[float] | None:
    if embedding_model is None:
        return None

    event_row = conn.execute(
        "SELECT content FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if event_row is None:
        raise ValueError(f"Event {event_id} not found")

    try:
        return embedding_model.embed(event_row["content"])
    except ModelUnavailableError:
        logger.warning(
            "Embedding unavailable for event %s; proceeding unembedded "
            "(pure containment). Event has no vector until re-embedded.",
            event_id,
        )
        return None


def _vec0_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'event_embeddings'"
    ).fetchone()
    return row is not None


def _store_embedding(
    conn: sqlite3.Connection,
    event_id: int,
    embedding: list[float],
) -> None:
    """Store an embedding; raise if vector backend is absent."""

    if not _vec0_available(conn):
        raise EmbeddingBackendUnavailableError(
            "Cannot store embedding because event_embeddings vec0 table is absent. "
            "This should only happen when embeddings are disabled, in which case "
            "_store_embedding must not be called."
        )

    blob = struct.pack(f"{len(embedding)}f", *embedding)
    conn.execute(
        "INSERT INTO event_embeddings (event_id, embedding) VALUES (?, ?)",
        (event_id, blob),
    )


def process_pending_events(
    conn: sqlite3.Connection,
    model: ModelInterface | None,
    embedding_model: EmbeddingInterface | None,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> list[int]:
    """Process all pending events using the parallel batch engine.

    ``on_progress(done, total)`` reports live progress.
    """

    if model is None:
        logger.warning("No model available; leaving pending events unprocessed.")
        return []

    rows = conn.execute(
        "SELECT id, source, content FROM events "
        "WHERE extraction_status = 'pending' ORDER BY timestamp ASC, id ASC"
    ).fetchall()
    pending = [(int(r["id"]), r["source"], r["content"]) for r in rows]
    return _process_events_batch(conn, pending, model, embedding_model, on_progress)


def ingest_and_process(
    conn: sqlite3.Connection,
    content: str,
    source: str,
    model: ModelInterface | None,
    embedding_model: EmbeddingInterface | None,
    timestamp: str | None = None,
    metadata: dict | None = None,
) -> list[int]:
    """Ingest content and immediately process created events.

    Intake (Stage 1) always commits, so the event is durable even when no model
    is available. If the model is None or becomes unavailable mid-run, the
    created events stay ``pending`` for a later ``men process``; this function
    never raises ModelUnavailableError to the caller.
    """

    event_ids = ingest_event(conn, content, source, timestamp, metadata)
    if model is None:
        if event_ids:
            logger.warning(
                "No model available; %d event(s) saved as pending.", len(event_ids)
            )
        return event_ids

    for event_id in event_ids:
        try:
            process_event(conn, event_id, model, embedding_model)
        except ModelUnavailableError as exc:
            logger.warning(
                "Model call failed (%s); event %s and any remaining stay pending.",
                exc,
                event_id,
            )
            break
    return event_ids


def import_and_process(
    conn: sqlite3.Connection,
    path: str,
    model: ModelInterface | None,
    embedding_model: EmbeddingInterface | None,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> list[int]:
    """Import a file/directory and process the created events in PARALLEL.

    Design (see below): the slow, independent work (entity extraction, embedding)
    runs concurrently; only the order-dependent thread assignment stays serial,
    preserving determinism. Memory is bounded by processing in windows.

      Phase A  extract (parallel) + embed (batched)   — I/O-bound, concurrent
      Phase B  reconcile + store + assign (serial, timestamp order) — fast, local
      Phase C  summarize each affected thread (parallel LLM, serial writes)

    Intake always commits first; if no model is available the events stay
    ``pending`` for a later ``men process`` rather than raising.
    """

    from pathlib import Path

    from event_intake import ingest_directory, ingest_file

    target = Path(path)
    if target.is_dir():
        event_ids = ingest_directory(conn, path)
    elif target.is_file():
        event_ids = ingest_file(conn, path)
    else:
        raise FileNotFoundError(path)

    if not event_ids:
        return []
    if model is None:
        logger.warning(
            "No model available; %d imported event(s) saved as pending.",
            len(event_ids),
        )
        return event_ids

    placeholders = ",".join("?" for _ in event_ids)
    rows = conn.execute(
        f"SELECT id, source, content FROM events "
        f"WHERE id IN ({placeholders}) AND extraction_status = 'pending' "
        f"ORDER BY timestamp ASC, id ASC",
        event_ids,
    ).fetchall()
    pending = [(int(r["id"]), r["source"], r["content"]) for r in rows]
    _process_events_batch(conn, pending, model, embedding_model, on_progress)
    return event_ids


def _process_events_batch(
    conn: sqlite3.Connection,
    pending: list[tuple[int, str, str]],
    model: ModelInterface,
    embedding_model: EmbeddingInterface | None,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> list[int]:
    """Parallel batch engine shared by `men import` and `men process`.

      Phase A  extract (parallel) + embed (batched)   — I/O-bound, concurrent
      Phase B  reconcile + store + assign (serial, timestamp order) — fast, local
      Phase C  summarize each affected thread (parallel LLM, serial writes)

    Assignment stays serial and in timestamp order to preserve determinism;
    memory is bounded by windows. Returns the processed event ids.
    """

    import config
    import thread_summarizer as summarizer

    total = len(pending)
    processed_ids: list[int] = []
    affected: set[int] = set()
    workers = max(1, config.IMPORT_CONCURRENCY)

    for start in range(0, total, config.IMPORT_WINDOW):
        window = pending[start : start + config.IMPORT_WINDOW]

        # --- Phase A1: parallel entity extraction (workers touch NO DB) ---
        extractions: dict[int, object] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_with_retry, src, content, model): eid
                for eid, src, content in window
            }
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    extractions[eid] = future.result()
                except ModelUnavailableError:
                    pass  # this event stays pending

        if not extractions:
            logger.warning(
                "Import halted: model unavailable for a whole window; "
                "remaining events stay pending (resume with `men process`)."
            )
            break

        # --- Phase A2: batched embeddings for the extracted events ---
        embeddings: dict[int, list[float] | None] = {}
        if embedding_model is not None:
            ok = [(eid, content) for eid, _src, content in window if eid in extractions]
            for i in range(0, len(ok), config.EMBED_BATCH_SIZE):
                chunk = ok[i : i + config.EMBED_BATCH_SIZE]
                try:
                    vectors = _embed_batch_with_retry(
                        embedding_model, [c for _eid, c in chunk]
                    )
                    for (eid, _c), vector in zip(chunk, vectors):
                        embeddings[eid] = vector
                except ModelUnavailableError as exc:
                    logger.warning(
                        "Embedding batch failed (%s); those events stay unembedded.",
                        exc,
                    )
                    for eid, _c in chunk:
                        embeddings[eid] = None

        # --- Phase B: serial reconcile + store + assign, in timestamp order ---
        for eid, _src, _content in window:
            if eid not in extractions:
                continue
            with transactional(conn) as txn:
                entity_ids = reconcile_entities(txn, eid, extractions[eid])
                vector = embeddings.get(eid)
                if vector is not None:
                    _store_embedding(txn, eid, vector)
                txn.execute(
                    "UPDATE events SET extraction_status = ? WHERE id = ?",
                    (ExtractionStatus.COMPLETED, eid),
                )
            with transactional(conn) as txn:
                affected.add(assign_thread(txn, eid, entity_ids))
            processed_ids.append(eid)
            if on_progress is not None:
                on_progress(len(processed_ids), total)

    # --- Phase C: parallel summarization (LLM in workers, writes on main) ---
    if affected:
        inputs: dict[int, tuple[str, str]] = {}
        for tid in sorted(affected):
            event_rows = conn.execute(
                "SELECT e.id, e.content, e.timestamp, e.source FROM events e "
                "JOIN event_thread_edges ete ON e.id = ete.event_id "
                "WHERE ete.thread_id = ? ORDER BY e.timestamp ASC",
                (tid,),
            ).fetchall()
            if event_rows:
                inputs[tid] = summarizer.render_events(event_rows)

        summaries: dict[int, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_summarize_with_retry, text, first, model): tid
                for tid, (text, first) in inputs.items()
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    summaries[tid] = future.result()
                except Exception:  # keep going; thread keeps its default title
                    pass

        for tid, (title, summary) in summaries.items():
            summarizer.write_thread_summary(conn, tid, title, summary)

    return processed_ids


# --------------------------------------------------------------------------- #
# Batch helpers: rate-limit-aware retry wrappers (run inside worker threads)   #
# --------------------------------------------------------------------------- #

def _is_rate_limit(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "rate" in message.lower()


def _ratelimit_retry(call):
    """Call `call()`, backing off and retrying only on a rate-limit (429).

    The providers fail fast on 429 for the interactive path; bulk import expects
    to brush the per-minute limit, so here we back off and retry.
    """

    from config import IMPORT_RATELIMIT_RETRIES, RETRY_BASE_DELAY_SECONDS

    last_error: Exception | None = None
    for attempt in range(IMPORT_RATELIMIT_RETRIES):
        try:
            return call()
        except ModelUnavailableError as exc:
            last_error = exc
            if _is_rate_limit(exc) and attempt < IMPORT_RATELIMIT_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))
                continue
            raise
    raise last_error  # pragma: no cover


def _extract_with_retry(source: str, content: str, model: ModelInterface):
    from entity_extractor import extract_from_content

    return _ratelimit_retry(lambda: extract_from_content(source, content, model))


def _embed_batch_with_retry(embedding_model: EmbeddingInterface, texts: list[str]):
    return _ratelimit_retry(lambda: embedding_model.embed_batch(texts))


def _summarize_with_retry(
    events_text: str, first_content: str, model: ModelInterface
) -> tuple[str, str]:
    import thread_summarizer as summarizer

    try:
        return _ratelimit_retry(lambda: summarizer.summarize_from_text(events_text, model))
    except ModelUnavailableError:
        return summarizer.deterministic_title(first_content), ""
