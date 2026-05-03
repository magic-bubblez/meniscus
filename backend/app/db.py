from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "backend" / "data"
DB_PATH = DATA_DIR / "meniscus.db"
SEED_PATH = DATA_DIR / "seed_events.json"


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    connection = get_connection()
    try:
        cursor = connection.cursor()
        yield cursor
        connection.commit()
    finally:
        connection.close()


def initialize_db() -> None:
    with db_cursor() as cursor:
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_entity_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                UNIQUE(event_id, entity_id),
                FOREIGN KEY(event_id) REFERENCES events(id),
                FOREIGN KEY(entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS event_thread_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL UNIQUE,
                thread_id INTEGER NOT NULL,
                FOREIGN KEY(event_id) REFERENCES events(id),
                FOREIGN KEY(thread_id) REFERENCES threads(id)
            );
            """
        )


def event_count() -> int:
    with db_cursor() as cursor:
        row = cursor.execute("SELECT COUNT(*) AS count FROM events").fetchone()
    return int(row["count"])


def serialize_metadata(metadata: dict) -> str:
    return _json_dumps(metadata)


def deserialize_metadata(raw_metadata: str) -> dict:
    return json.loads(raw_metadata)

