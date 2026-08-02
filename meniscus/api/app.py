from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from meniscus.db import get_connection, init_db
    from meniscus.pipeline import process_pending_events
    from meniscus.providers import get_embedding_model, get_model
    from meniscus.startup import announce_embedding_state

    model = get_model()
    embedding_model = get_embedding_model()
    conn = get_connection()
    try:
        init_db(conn)
        announce_embedding_state()
        process_pending_events(conn, model, embedding_model)
    finally:
        conn.close()

    app.state.model = model
    app.state.embedding_model = embedding_model
    yield


app = FastAPI(
    title="Meniscus",
    description="Local structured memory for AI agents",
    lifespan=lifespan,
)

from meniscus.api.events import router as events_router

app.include_router(events_router)
