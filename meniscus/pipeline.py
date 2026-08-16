"""Pipeline orchestration across intake and fact extraction."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from meniscus.db import transactional
from meniscus.embedding_interface import EmbeddingInterface
from meniscus.entity_extractor import EXTRACTION_PROMPT_VERSION, extract_entities
from meniscus.event_intake import ingest_event
from meniscus.exceptions import ModelUnavailableError
from meniscus.models import ExtractionResult
from meniscus.model_interface import ModelInterface
from meniscus.meniscus_types import ExtractionStatus
from meniscus.vocab_reconciliation import reconcile_entities

logger = logging.getLogger(__name__)


def process_event(
    conn: sqlite3.Connection,
    event_id: int,
    model: ModelInterface,
    embedding_model: EmbeddingInterface | None,
) -> int:
    """Extract facts from a pending event, then store and embed them."""

    extraction_result = extract_entities(conn, event_id, model)
    if not extraction_result.facts:
        with transactional(conn) as txn:
            txn.execute(
                "UPDATE events SET extraction_status = ? WHERE id = ?",
                (ExtractionStatus.COMPLETED, event_id),
            )
        logger.info("Event %s had nothing to extract; marked completed.", event_id)
        return event_id

    with transactional(conn) as txn:
        entity_ids = reconcile_entities(txn, event_id, extraction_result)
        fact_rows = _store_facts(txn, event_id, extraction_result)
        _store_fact_entity_edges(txn, fact_rows, entity_ids)
        txn.execute(
            "UPDATE events SET extraction_status = ? WHERE id = ?",
            (ExtractionStatus.COMPLETED, event_id),
        )

    _embed_facts(conn, fact_rows, embedding_model)
    return event_id


def _extraction_provenance() -> tuple[str, str]:
    """Resolve (provider, model) for the active extractor from config/env."""

    from meniscus import config

    return config.DEFAULT_MODEL_PROVIDER, config.MODEL


def _store_facts(
    conn: sqlite3.Connection,
    event_id: int,
    extraction_result: ExtractionResult,
) -> list[tuple[int, str]]:
    """Append one extraction run and its facts (never updates); return (fact_id, text)."""

    provider, model_name = _extraction_provenance()
    extracted_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO extractions "
        "(event_id, provider, model, prompt_version, extracted_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, provider, model_name, EXTRACTION_PROMPT_VERSION, extracted_at),
    )
    extraction_id = int(cursor.lastrowid)
    fact_rows: list[tuple[int, str]] = []
    for position, fact in enumerate(extraction_result.facts):
        cur = conn.execute(
            "INSERT INTO facts (event_id, extraction_id, text, position, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_id, extraction_id, fact, position, extracted_at),
        )
        fact_rows.append((int(cur.lastrowid), fact))
    return fact_rows


_WORD_RE = re.compile(r"\w+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 1}


def _store_fact_entity_edges(
    conn: sqlite3.Connection,
    fact_rows: list[tuple[int, str]],
    entity_ids: list[int],
) -> None:
    """Link each fact to the event's entities whose surface form appears in it."""

    surfaces: dict[int, list[set[str]]] = {}
    for entity_id in entity_ids:
        forms: list[str] = []
        row = conn.execute(
            "SELECT canonical_name FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is not None:
            forms.append(row["canonical_name"])
        for alias_row in conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?", (entity_id,)
        ).fetchall():
            forms.append(alias_row["alias"])
        token_sets = [tokens for tokens in (_content_tokens(f) for f in forms) if tokens]
        surfaces[entity_id] = token_sets

    for fact_id, text in fact_rows:
        fact_tokens = _content_tokens(text)
        for entity_id, token_sets in surfaces.items():
            if any(form_tokens <= fact_tokens for form_tokens in token_sets):
                conn.execute(
                    "INSERT OR IGNORE INTO fact_entity_edges (fact_id, entity_id) "
                    "VALUES (?, ?)",
                    (fact_id, entity_id),
                )


def _fact_vec0_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fact_embeddings'"
    ).fetchone()
    return row is not None


def _embed_facts(
    conn: sqlite3.Connection,
    fact_rows: list[tuple[int, str]],
    embedding_model: EmbeddingInterface | None,
) -> None:
    """Embed fact texts and store them in fact_embeddings (own transaction)."""

    from meniscus import config

    if embedding_model is None or not fact_rows or not _fact_vec0_available(conn):
        return
    for i in range(0, len(fact_rows), config.EMBED_BATCH_SIZE):
        chunk = fact_rows[i : i + config.EMBED_BATCH_SIZE]
        try:
            vectors = _embed_batch_with_retry(embedding_model, [text for _fid, text in chunk])
        except ModelUnavailableError as exc:
            logger.warning("Fact embedding unavailable (%s); %d facts left unembedded.", exc, len(chunk))
            continue
        with transactional(conn) as txn:
            for (fact_id, _text), vector in zip(chunk, vectors):
                blob = struct.pack(f"{len(vector)}f", *vector)
                txn.execute(
                    "INSERT INTO fact_embeddings (fact_id, embedding) VALUES (?, ?)",
                    (fact_id, blob),
                )


def process_pending_events(
    conn: sqlite3.Connection,
    model: ModelInterface | None,
    embedding_model: EmbeddingInterface | None,
    on_progress: "Callable[[int, int], None] | None" = None,
    allow_large_run: bool = False,
) -> list[int]:
    """Process all pending events using the parallel batch engine.

    ``on_progress(done, total)`` reports live progress.
    """

    if model is None:
        logger.warning("No model available; leaving pending events unprocessed.")
        return []

    rows = conn.execute(
        "SELECT id, source, content, timestamp FROM events "
        "WHERE extraction_status = ? ORDER BY timestamp ASC, id ASC",
        (ExtractionStatus.PENDING,),
    ).fetchall()
    pending = [(int(r["id"]), r["source"], r["content"], r["timestamp"]) for r in rows]
    return _process_events_batch(
        conn, pending, model, embedding_model, on_progress, allow_large_run
    )


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
    allow_large_run: bool = False,
) -> list[int]:
    """Import a file/directory and batch-process created events (parallel extract)."""

    from pathlib import Path

    from meniscus.event_intake import ingest_directory, ingest_file

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
        f"SELECT id, source, content, timestamp FROM events "
        f"WHERE id IN ({placeholders}) AND extraction_status = ? "
        f"ORDER BY timestamp ASC, id ASC",
        [*event_ids, ExtractionStatus.PENDING],
    ).fetchall()
    pending = [(int(r["id"]), r["source"], r["content"], r["timestamp"]) for r in rows]
    _process_events_batch(
        conn, pending, model, embedding_model, on_progress, allow_large_run
    )
    return event_ids


def _process_events_batch(
    conn: sqlite3.Connection,
    pending: list[tuple[int, str, str, str]],
    model: ModelInterface,
    embedding_model: EmbeddingInterface | None,
    on_progress: "Callable[[int, int], None] | None" = None,
    allow_large_run: bool = False,
) -> list[int]:
    """Parallel batch engine shared by `men import` and `men process`."""

    from meniscus import config
    from meniscus.spend_gate import enforce_spend_ceiling

    _provider, model_slug = _extraction_provenance()
    enforce_spend_ceiling(pending, model_slug, allow_large_run)

    total = len(pending)
    processed_ids: list[int] = []
    workers = max(1, config.IMPORT_CONCURRENCY)

    for start in range(0, total, config.IMPORT_WINDOW):
        window = pending[start : start + config.IMPORT_WINDOW]

        # --- Phase A: parallel entity + fact extraction (workers touch NO DB) ---
        extractions: dict[int, object] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_with_retry, src, content, ts, model): eid
                for eid, src, content, ts in window
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

        # --- Phase B: serial reconcile + store facts + edges, in timestamp order ---
        window_fact_rows: list[tuple[int, str]] = []
        for eid, _src, _content, _ts in window:
            if eid not in extractions:
                continue
            result = extractions[eid]
            if not result.facts:
                with transactional(conn) as txn:
                    txn.execute(
                        "UPDATE events SET extraction_status = ? WHERE id = ?",
                        (ExtractionStatus.COMPLETED, eid),
                    )
                processed_ids.append(eid)
                if on_progress is not None:
                    on_progress(len(processed_ids), total)
                logger.info("Event %s had nothing to extract; marked completed.", eid)
                continue
            with transactional(conn) as txn:
                entity_ids = reconcile_entities(txn, eid, result)
                fact_rows = _store_facts(txn, eid, result)
                _store_fact_entity_edges(txn, fact_rows, entity_ids)
                txn.execute(
                    "UPDATE events SET extraction_status = ? WHERE id = ?",
                    (ExtractionStatus.COMPLETED, eid),
                )
            window_fact_rows.extend(fact_rows)
            processed_ids.append(eid)
            if on_progress is not None:
                on_progress(len(processed_ids), total)

        # --- Phase C: batched fact embeddings ---
        _embed_facts(conn, window_fact_rows, embedding_model)

    return processed_ids


def _is_rate_limit(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "rate" in message.lower()


def _ratelimit_retry(call):
    """Call `call()`, backing off and retrying only on a rate-limit (429).

    The providers fail fast on 429 for the interactive path; bulk import expects
    to brush the per-minute limit, so here we back off and retry.
    """

    from meniscus.config import IMPORT_RATELIMIT_RETRIES, RETRY_BASE_DELAY_SECONDS

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


def _extract_with_retry(source: str, content: str, timestamp: str, model: ModelInterface):
    from meniscus.entity_extractor import extract_from_content

    return _ratelimit_retry(lambda: extract_from_content(source, content, model, timestamp))


def _embed_batch_with_retry(embedding_model: EmbeddingInterface, texts: list[str]):
    return _ratelimit_retry(lambda: embedding_model.embed_batch(texts))
