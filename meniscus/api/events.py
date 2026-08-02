from __future__ import annotations

import sqlite3
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

router = APIRouter()


def get_db() -> Iterator[sqlite3.Connection]:
    from meniscus.db import get_connection, init_db

    conn = get_connection()
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


class EventCreate(BaseModel):
    content: str
    source: str = "api"
    timestamp: str | None = None
    metadata: dict | None = None


class EventResponse(BaseModel):
    event_ids: list[int]
    message: str


@router.post("/events", response_model=EventResponse)
def create_event(
    payload: EventCreate,
    request: Request,
    conn: sqlite3.Connection = Depends(get_db),
) -> EventResponse:
    from meniscus.pipeline import ingest_and_process

    event_ids = ingest_and_process(
        conn,
        content=payload.content,
        source=payload.source,
        timestamp=payload.timestamp,
        metadata=payload.metadata,
        model=request.app.state.model,
        embedding_model=request.app.state.embedding_model,
    )
    return EventResponse(event_ids=event_ids, message="created" if event_ids else "duplicate")


@router.get("/events")
def list_events(
    limit: int = 20,
    conn: sqlite3.Connection = Depends(get_db),
):
    rows = conn.execute(
        "SELECT id, source, content, timestamp, extraction_status "
        "FROM events ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {"events": [dict(row) for row in rows]}
