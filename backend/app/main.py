from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env from backend/.env at import time so GEMINI_API_KEY (and any future
# secrets) are available before the pipeline runs its first synthesize_answer.
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

from .db import SEED_PATH, event_count, initialize_db
from .pipeline import (
    answer_query,
    build_overview,
    build_thread_graph,
    get_graph_payload,
    get_thread_subgraph,
    ingest_event,
    list_threads,
    prune_orphan_entities,
    prune_stopword_entities,
    refresh_all_threads,
    search_threads,
)


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"


class IngestRequest(BaseModel):
    source: str = Field(..., examples=["youtube"])
    content: str = Field(..., examples=["Watched JWT auth flow breakdown"])
    timestamp: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., examples=["What have I been doing about auth?"])


app = FastAPI(title="Meniscus MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    initialize_db()
    if event_count() == 0:
        seed_demo_data()
    prune_stopword_entities()
    prune_orphan_entities()
    refresh_all_threads()


def seed_demo_data() -> dict:
    with SEED_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = [ingest_event(item) for item in payload["events"]]
    return {"ingested": len(results), "results": results}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/overview")
def overview() -> dict:
    return build_overview()


@app.post("/api/ingest")
def ingest(request: IngestRequest) -> dict:
    # MVP uses simulated ingestion via API or seed data.
    # In production:
    # - browser extensions capture ChatGPT/YouTube
    # - GitHub API provides commit events
    return ingest_event(request.model_dump())


@app.post("/api/seed")
def seed() -> dict:
    return seed_demo_data()


@app.get("/api/search")
def search(q: str = "") -> dict:
    return {"query": q, "threads": search_threads(q)}


@app.get("/threads")
def threads(q: str = "") -> list[dict]:
    return list_threads(q)


@app.get("/threads/{thread_id}")
def threads_detail(thread_id: int) -> dict:
    subgraph = get_thread_subgraph(thread_id)
    if subgraph is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return subgraph


@app.get("/threads/{thread_id}/graph")
def thread_graph(thread_id: int) -> dict:
    payload = build_thread_graph(thread_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return payload


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return answer_query(request.question)


@app.get("/api/threads/{thread_id}")
def thread_detail(thread_id: int) -> dict:
    subgraph = get_thread_subgraph(thread_id)
    if subgraph is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    # Any context can be retrieved as a connected subgraph
    # of events + entities + thread
    return subgraph


@app.get("/api/graph")
def graph() -> dict:
    return get_graph_payload()
