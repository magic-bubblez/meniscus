"""SQLite connection and schema management."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config import DB_PATH, EMBEDDING_DIMENSIONS, EMBEDDING_PROVIDER
from exceptions import (
    DatabaseError,
    EmbeddingBackendUnavailableError,
    EmbeddingDimensionMismatchError,
)

_VECTOR_MARKER = "-- VECTOR SEARCH"


def get_db_path() -> Path:
    """Resolve the database path, honoring MENISCUS_DB_PATH."""

    value = os.environ.get("MENISCUS_DB_PATH") or DB_PATH
    path = Path(value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Create and configure a SQLite connection."""

    path = db_path if db_path is not None else get_db_path()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Wait up to 5s for the single writer lock instead of erroring immediately —
    # lets a fast log write and the background processor coexist (WAL = 1 writer).
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _find_schema() -> Path:
    """Locate schema.sql in installed-package or source-tree layout."""

    candidates = [
        Path(__file__).parent / "schema.sql",
        Path(__file__).parent.parent / "schema.sql",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise DatabaseError("schema.sql not found")


def _split_schema(schema_sql: str) -> tuple[str, str]:
    """Split schema into non-vector and vector sections."""

    if _VECTOR_MARKER not in schema_sql:
        raise DatabaseError("schema.sql missing VECTOR SEARCH marker")
    main_schema, vector_tail = schema_sql.split(_VECTOR_MARKER, 1)
    return main_schema, vector_tail


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize schema and validate embedding backend/dimension invariants."""

    _validate_known_provider_dimension_config()

    schema_path = _find_schema()
    try:
        schema_sql = schema_path.read_text()
    except OSError as exc:
        raise DatabaseError(f"Could not read schema.sql: {exc}") from exc

    schema_sql = schema_sql.replace(
        "{EMBEDDING_DIMENSIONS}", str(EMBEDDING_DIMENSIONS)
    )
    main_schema, vector_schema = _split_schema(schema_sql)

    try:
        conn.executescript(main_schema)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Failed to initialize schema: {exc}") from exc

    if EMBEDDING_PROVIDER != "none":
        try:
            import sqlite_vec

            # Loadable extensions are disabled on a fresh connection; enable them
            # before loading sqlite-vec, then re-disable so nothing else can load
            # arbitrary extensions. (AttributeError here means this Python's
            # sqlite3 was built without extension support — also a real failure.)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            raise EmbeddingBackendUnavailableError(
                f"Embedding provider {EMBEDDING_PROVIDER!r} is configured but "
                "the sqlite-vec extension could not be loaded. Install "
                'sqlite-vec, or set EMBEDDING_PROVIDER="none" to run in the '
                "deliberate disabled mode."
            ) from exc

        try:
            conn.executescript(_VECTOR_MARKER + vector_schema)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to initialize vector schema: {exc}") from exc

    _validate_or_record_embedding_dimensions(conn)


def _validate_or_record_embedding_dimensions(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'embedding_dimensions'"
    ).fetchone()
    if row is None:
        with transactional(conn) as txn:
            txn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("embedding_dimensions", str(EMBEDDING_DIMENSIONS)),
            )
        return

    stored = int(row["value"])
    if stored != EMBEDDING_DIMENSIONS:
        raise EmbeddingDimensionMismatchError(
            "Embedding dimension mismatch: config "
            f"EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS} but database was "
            f"created with {stored}. Recreate the database."
        )


def _validate_known_provider_dimension_config() -> None:
    """Fail before schema creation for built-in provider/config mismatches."""

    if EMBEDDING_PROVIDER == "local" and EMBEDDING_DIMENSIONS != 768:
        raise EmbeddingDimensionMismatchError(
            "Embedding provider 'local' (bge-base) produces 768-dimensional vectors, "
            f"but EMBEDDING_DIMENSIONS is {EMBEDDING_DIMENSIONS}. Set "
            "EMBEDDING_DIMENSIONS=768 before initializing the database."
        )


@contextmanager
def transactional(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Execute a block in an explicit transaction."""

    conn.execute("BEGIN")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
