# Meniscus

> **A context layer for AI agents.** Memory shouldn't live inside one tool — it should live outside, so every AI you use can read what you're working on.

**Demo video:** [Link](https://drive.google.com/file/d/1HIE1Kz1njfJRazMOHIFh5263E-ZAD4b4/view?usp=sharing)
**Live:** [Link](https://meniscus.onrender.com)

---

## What it does

Meniscus captures your activity across tools (ChatGPT, YouTube, GitHub, Notion…), structures it into a knowledge graph of **events → entities → threads**, and lets any AI agent retrieve a relevant **subgraph** instead of being handed raw history.

```
activity → event → entity → thread → subgraph → agent
```

- **Event** — atomic, immutable record of one user action.
- **Entity** — meaningful concept extracted from an event (`jwt`, `react`, `pitch`).
- **Thread** — cluster of related events. The retrieval unit.
- **Subgraph** — thread + its events + its entities. What an agent receives.

The full pipeline is documented in [`REDESIGN_BRIEF.md`](./REDESIGN_BRIEF.md).

---

## Quick start

```bash
# 1. clone and enter
git clone <your-fork-url> meniscus
cd meniscus

# 2. set up the backend
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt

# 3. add your Gemini key
echo "GEMINI_API_KEY=your-key-here" > backend/.env

# 4. run
backend/.venv/bin/uvicorn --app-dir backend app.main:app --reload
```

Open **http://127.0.0.1:8000** — you'll land on the marketing page; click *Open workspace* to enter the app.

> Without a Gemini key, `/query` falls back to deterministic synthesis so the demo still works — but the conversational answers come from Gemini.

---

## Architecture

```
┌────────────────┐   POST /ingest    ┌─────────────────────────────┐
│  Sources       │ ────────────────▶ │  Ingestion pipeline         │
│  (simulated)   │                   │  normalize → extract signals│
│  YouTube       │                   │  → upsert entities          │
│  ChatGPT       │                   │  → cluster into threads     │
│  GitHub        │                   │  → write edges              │
│  Notion        │                   └────────────┬────────────────┘
└────────────────┘                                │
                                                  ▼
                                       ┌──────────────────────┐
                                       │   SQLite             │
                                       │   events             │
                                       │   entities           │
                                       │   threads            │
                                       │   event_entity_edges │
                                       │   event_thread_edges │
                                       └──────────┬───────────┘
                                                  │
                       ┌──────────────────────────┼─────────────────────────┐
                       ▼                          ▼                         ▼
                ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
                │ POST /query  │         │ GET          │          │ GET          │
                │              │         │ /threads/:id │          │ /api/graph   │
                │ Gemini +     │         │              │          │              │
                │ subgraph ─▶  │         │ subgraph     │          │ all nodes +  │
                │ grounded     │         │ retrieval    │          │ all edges    │
                │ answer       │         │              │          │              │
                └──────────────┘         └──────────────┘          └──────────────┘
```

**Stack:** FastAPI · SQLite · vanilla HTML/CSS/JS · Google `gemini-2.5-flash`.

---

## How `/query` actually works

The endpoint routes through three modes based on intent:

| Mode | Trigger | What it does |
|---|---|---|
| **Retrieve** | Query mentions entities that overlap with a thread | Grounds Gemini's answer in that thread's full subgraph |
| **Overview** | Query asks about user's own work (`"what have I been doing?"`) | Pulls all recent threads, weaves a cross-thread summary |
| **General** | Greeting or unrelated topic | Conversational reply, no retrieval, doesn't fabricate context |

This is what makes Meniscus answers different from a wrapped chatbot: every retrieved answer is **citation-backed by a thread subgraph**, and general chat doesn't pretend to be grounded.

---

## API

All endpoints are at the same origin as the frontend.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Add an event: `{source, content, timestamp?, metadata?}` |
| `POST` | `/query` | Ask Meniscus: `{question}` → `{answer, threads, context, mode}` |
| `GET` | `/threads` | List/search threads. Optional `?q=keyword`. |
| `GET` | `/threads/{id}` | Subgraph for one thread (events + entities) |
| `GET` | `/api/graph` | All nodes + edges (for the Memory Graph view) |
| `GET` | `/api/overview` | Counts + ranked threads (sidebar / dashboard) |
| `POST` | `/api/seed` | Re-load demo data |

---

## Project layout

```
meniscus/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI app + routes
│   │   ├── pipeline.py    # ingestion + retrieval + LLM call
│   │   └── db.py          # SQLite setup
│   ├── data/
│   │   ├── seed_events.json
│   │   └── meniscus.db    # auto-created on first run
│   └── requirements.txt
└── frontend/
    ├── index.html         # landing + workspace shell
    ├── styles.css         # light theme
    └── app.js             # views, fetch, force-directed graph
```

---

## Deploy

The simplest path is **Render** (free tier, auto-builds from GitHub):

1. Push this repo to GitHub.
2. New Web Service on render.com, connect the repo.
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `uvicorn --app-dir backend app.main:app --host 0.0.0.0 --port $PORT`
5. Add env var `GEMINI_API_KEY`.

---


