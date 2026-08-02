"""Stage 1 event intake."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from meniscus.chunker import chunk_content
from meniscus.db import transactional
from meniscus.meniscus_types import ExtractionStatus

_SUPPORTED_EXTENSIONS = {".txt", ".md"}


def compute_content_hash(content: str, source: str) -> str:
    """Compute the dedup hash for content from a source."""

    return hashlib.sha256((content + source).encode("utf-8")).hexdigest()


def _normalize_timestamp(raw: str | None) -> str:
    """Normalize timestamp to canonical UTC ISO-8601 with +00:00 offset."""

    if raw is None:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def ingest_event(
    conn: sqlite3.Connection,
    content: str,
    source: str,
    timestamp: str | None = None,
    metadata: dict | None = None,
    source_id: str | None = None,
) -> list[int]:
    """Persist raw content as one or more pending events."""

    normalized_timestamp = _normalize_timestamp(timestamp)
    metadata = {} if metadata is None else metadata
    chunks = chunk_content(content)
    if len(chunks) > 1 and source_id is None:
        source_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    event_ids: list[int] = []
    with transactional(conn) as txn:
        for chunk_index, chunk in enumerate(chunks):
            content_hash = compute_content_hash(chunk, source)
            existing = txn.execute(
                "SELECT 1 FROM events WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing is not None:
                continue

            chunk_metadata = dict(metadata)
            if len(chunks) > 1:
                chunk_metadata["chunk_index"] = chunk_index

            cursor = txn.execute(
                "INSERT INTO events "
                "(source, content, timestamp, metadata, extraction_status, "
                "content_hash, source_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    source,
                    chunk,
                    normalized_timestamp,
                    json.dumps(chunk_metadata),
                    ExtractionStatus.PENDING,
                    content_hash,
                    source_id,
                ),
            )
            event_ids.append(int(cursor.lastrowid))
    return event_ids


def ingest_file(
    conn: sqlite3.Connection,
    path: str,
    root: str | Path | None = None,
) -> list[int]:
    """Read and ingest a supported text/markdown file."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")
    if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    content = file_path.read_text()
    timestamp = datetime.fromtimestamp(
        file_path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()
    relative = _relative_source_path(file_path, root)
    source = f"file:{relative}"
    source_id = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return ingest_event(
        conn,
        content=content,
        source=source,
        timestamp=timestamp,
        metadata={"path": str(file_path)},
        source_id=source_id,
    )


def ingest_directory(conn: sqlite3.Connection, path: str) -> list[int]:
    """Recursively ingest .txt and .md files from a directory."""

    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(path)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {path}")

    event_ids: list[int] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            event_ids.extend(ingest_file(conn, str(file_path), root=directory))
    return event_ids


def _relative_source_path(path: Path, root: str | Path | None) -> str:
    if root is None:
        return path.name
    try:
        return path.relative_to(Path(root)).as_posix()
    except ValueError:
        return path.name
