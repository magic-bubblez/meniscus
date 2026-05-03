from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from .db import db_cursor, deserialize_metadata, serialize_metadata


# Generic English noise + tool/activity verbs that are sources of capture, not
# concepts the user is actually working on. Domain terms (token, refresh, design,
# graph, react, etc.) intentionally remain extractable as entities.
STOPWORDS = {
    # Articles, conjunctions, prepositions
    "the", "and", "for", "from", "with", "into", "onto", "than",
    "about", "after", "before", "again", "around", "over", "under",
    "between", "through", "across", "more", "less", "just",
    # Pronouns and possessives
    "this", "that", "these", "those", "they", "them", "their",
    "your", "his", "her", "its", "our", "you", "we", "i",
    # Common verbs (be / aux)
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had",
    "does", "did", "doing", "done",
    "will", "would", "should", "could", "can", "may", "might", "must", "shall",
    # Question words and connectives
    "what", "when", "where", "why", "how", "who", "whom", "which",
    "not", "but", "any", "all", "some", "such", "very", "much", "even",
    # Tool / source names — these label the source, not the concept
    "chatgpt", "youtube", "github", "claude", "gemini", "notion",
    # Activity verbs around captured events
    "watched", "watches", "asked", "asks", "queried",
    "commit", "committed", "video", "query",
}


@dataclass
class NormalizedEvent:
    source: str
    content: str
    timestamp: str
    metadata: dict


def normalize_event(payload: dict) -> NormalizedEvent:
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    metadata = payload.get("metadata") or {}
    return NormalizedEvent(
        source=str(payload["source"]).strip(),
        content=str(payload["content"]).strip(),
        timestamp=timestamp,
        metadata=metadata,
    )


def extract_signals(text: str) -> list[str]:
    # This is a simplified approximation of semantic extraction.
    # In production, embeddings or LLMs would be used.
    cleaned = []
    token = []
    for character in text.lower():
        if character.isalnum():
            token.append(character)
            continue
        if token:
            cleaned.append("".join(token))
            token = []
    if token:
        cleaned.append("".join(token))

    signals = []
    for item in cleaned:
        if len(item) < 3:
            continue
        if item in STOPWORDS:
            continue
        signals.append(item)
    return list(dict.fromkeys(signals))


def upsert_entity(name: str) -> int:
    with db_cursor() as cursor:
        cursor.execute("INSERT OR IGNORE INTO entities(name) VALUES (?)", (name,))
        row = cursor.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    return int(row["id"])


def get_thread_entities(thread_id: int) -> set[str]:
    with db_cursor() as cursor:
        rows = cursor.execute(
            """
            SELECT DISTINCT entities.name
            FROM entities
            JOIN event_entity_edges ON event_entity_edges.entity_id = entities.id
            JOIN event_thread_edges ON event_thread_edges.event_id = event_entity_edges.event_id
            WHERE event_thread_edges.thread_id = ?
            """,
            (thread_id,),
        ).fetchall()
    return {row["name"] for row in rows}


def get_last_active_thread(timestamp: str) -> Optional[dict]:
    with db_cursor() as cursor:
        row = cursor.execute(
            """
            SELECT id, title, summary, created_at, updated_at
            FROM threads
            WHERE updated_at <= ?
            ORDER BY updated_at DESC
            LIMIT 1
            """
            ,
            (timestamp,),
        ).fetchone()
        if row is None:
            row = cursor.execute(
                """
                SELECT id, title, summary, created_at, updated_at
                FROM threads
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
    return dict(row) if row else None


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def choose_thread(signals: Iterable[str], timestamp: str) -> int:
    signal_set = set(signals)
    last_thread = get_last_active_thread(timestamp)
    if last_thread is None:
        return create_thread(signal_set, timestamp)

    thread_entities = get_thread_entities(int(last_thread["id"]))
    shared_entities = signal_set & thread_entities
    total_entities = signal_set | thread_entities
    overlap = len(shared_entities) / len(total_entities) if total_entities else 0.0

    event_time = parse_timestamp(timestamp)
    last_time = parse_timestamp(last_thread["updated_at"])
    minutes_since_last = abs((event_time - last_time).total_seconds()) / 60

    # This is simple rule-based clustering.
    # In production:
    # - embeddings
    # - semantic similarity
    # - temporal modeling
    # will replace this.
    if overlap > 0.5 or minutes_since_last <= 30:
        return int(last_thread["id"])
    return create_thread(signal_set, timestamp)


def create_thread(signals: set[str], timestamp: str) -> int:
    title = build_thread_title(signals)
    summary = "No events yet."
    with db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO threads(title, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (title, summary, timestamp, timestamp),
        )
        thread_id = cursor.lastrowid
    return int(thread_id)


def build_thread_title(signals: set[str]) -> str:
    if not signals:
        return "Untitled thread"
    ordered = sorted(signals)
    return " / ".join(word.title() for word in ordered[:3])


def store_event(normalized_event: NormalizedEvent) -> int:
    with db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO events(source, content, timestamp, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                normalized_event.source,
                normalized_event.content,
                normalized_event.timestamp,
                serialize_metadata(normalized_event.metadata),
            ),
        )
        event_id = cursor.lastrowid
    return int(event_id)


def create_event_entity_edges(event_id: int, entity_ids: Iterable[int]) -> None:
    entity_pairs = [(event_id, entity_id) for entity_id in entity_ids]
    with db_cursor() as cursor:
        cursor.executemany(
            "INSERT OR IGNORE INTO event_entity_edges(event_id, entity_id) VALUES (?, ?)",
            entity_pairs,
        )


def create_event_thread_edge(event_id: int, thread_id: int) -> None:
    with db_cursor() as cursor:
        cursor.execute(
            "INSERT OR REPLACE INTO event_thread_edges(event_id, thread_id) VALUES (?, ?)",
            (event_id, thread_id),
        )


def update_thread_page(thread_id: int) -> None:
    # Thread page acts as compressed context.
    # Avoids scanning all events during retrieval.
    with db_cursor() as cursor:
        rows = cursor.execute(
            """
            SELECT events.content, events.timestamp
            FROM events
            JOIN event_thread_edges ON event_thread_edges.event_id = events.id
            WHERE event_thread_edges.thread_id = ?
            ORDER BY events.timestamp DESC
            LIMIT 5
            """,
            (thread_id,),
        ).fetchall()
        if not rows:
            return
        ordered_rows = list(reversed(rows))
        summary = summarize_thread_page(ordered_rows)
        entity_rows = cursor.execute(
            """
            SELECT entities.name
            FROM entities
            JOIN event_entity_edges ON event_entity_edges.entity_id = entities.id
            JOIN event_thread_edges ON event_thread_edges.event_id = event_entity_edges.event_id
            WHERE event_thread_edges.thread_id = ?
            """,
            (thread_id,),
        ).fetchall()
        title = summarize_thread_title([row["name"] for row in entity_rows])
        cursor.execute(
            """
            UPDATE threads
            SET title = ?, summary = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, summary, ordered_rows[-1]["timestamp"], thread_id),
        )


def summarize_thread_page(event_rows: list) -> str:
    if not event_rows:
        return "No summary available."
    snippets = [row["content"].strip().rstrip(".") for row in event_rows]
    first = f"This thread follows {snippets[0].lower()}."
    if len(snippets) == 1:
        return first[0].upper() + first[1:]
    second = f"It then moves through {snippets[1].lower()}."
    if len(snippets) == 2:
        return f"{first} {second}"
    third = f"Most recently it includes {snippets[-1].lower()}."
    return f"{first} {second} {third}"


def summarize_thread_title(entity_names: list[str]) -> str:
    if not entity_names:
        return "Untitled thread"
    ranked = [
        name
        for name, _count in Counter(entity_names).most_common()
        if name not in STOPWORDS
    ]
    if not ranked:
        ranked = entity_names
    return " / ".join(word.title() for word in ranked[:3])


def ingest_event(payload: dict) -> dict:
    normalized = normalize_event(payload)
    signals = extract_signals(normalized.content)

    # Entities are approximate signals, not perfect representations.
    # They exist to enable clustering and retrieval.
    entity_ids = [upsert_entity(signal) for signal in signals]
    thread_id = choose_thread(signals, normalized.timestamp)
    event_id = store_event(normalized)
    create_event_entity_edges(event_id, entity_ids)
    create_event_thread_edge(event_id, thread_id)
    update_thread_page(thread_id)
    return {
        "event_id": event_id,
        "thread_id": thread_id,
        "entities": signals,
    }


def search_threads(query: str) -> list[dict]:
    query_signals = extract_signals(query)
    now = datetime.now(timezone.utc)
    with db_cursor() as cursor:
        thread_rows = cursor.execute(
            """
            SELECT id, title, summary, created_at, updated_at
            FROM threads
            ORDER BY updated_at DESC
            """
        ).fetchall()

    ranked = []
    for row in thread_rows:
        thread_entities = get_thread_entities(int(row["id"]))
        overlap = len(set(query_signals) & thread_entities)
        title_text = row["title"].lower()
        title_hits = sum(1 for signal in query_signals if signal in title_text)
        updated_at = parse_timestamp(row["updated_at"])
        recency_hours = max((now - updated_at).total_seconds() / 3600, 0.0)
        recency_score = 1 / (1 + recency_hours)
        relevance = overlap * 3 + title_hits * 2 + recency_score
        if query_signals and relevance == recency_score:
            continue
        ranked.append(
            {
                "id": int(row["id"]),
                "title": row["title"],
                "summary": row["summary"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "entity_overlap": overlap,
                "score": round(relevance, 4),
            }
        )
    ranked.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
    return ranked


def get_thread_subgraph(thread_id: int) -> Optional[dict]:
    with db_cursor() as cursor:
        thread_row = cursor.execute(
            "SELECT id, title, summary, created_at, updated_at FROM threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
        if thread_row is None:
            return None

        event_rows = cursor.execute(
            """
            SELECT events.id, events.source, events.content, events.timestamp, events.metadata
            FROM events
            JOIN event_thread_edges ON event_thread_edges.event_id = events.id
            WHERE event_thread_edges.thread_id = ?
            ORDER BY events.timestamp DESC
            """,
            (thread_id,),
        ).fetchall()

        entity_rows = cursor.execute(
            """
            SELECT DISTINCT entities.id, entities.name
            FROM entities
            JOIN event_entity_edges ON event_entity_edges.entity_id = entities.id
            JOIN event_thread_edges ON event_thread_edges.event_id = event_entity_edges.event_id
            WHERE event_thread_edges.thread_id = ?
            ORDER BY entities.name ASC
            """,
            (thread_id,),
        ).fetchall()

    events = [
        {
            "id": int(row["id"]),
            "source": row["source"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "metadata": deserialize_metadata(row["metadata"]),
        }
        for row in event_rows
    ]
    entities = [
        {"id": int(row["id"]), "name": row["name"]}
        for row in entity_rows
        if row["name"] not in STOPWORDS
    ]
    return {
        "thread": {
            "id": int(thread_row["id"]),
            "title": thread_row["title"],
            "summary": thread_row["summary"],
            "created_at": thread_row["created_at"],
            "updated_at": thread_row["updated_at"],
        },
        "events": events,
        "entities": entities,
    }


def get_graph_payload() -> dict:
    with db_cursor() as cursor:
        thread_rows = cursor.execute(
            "SELECT id, title, summary, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
        entity_rows = cursor.execute("SELECT id, name FROM entities ORDER BY name ASC").fetchall()
        event_rows = cursor.execute(
            "SELECT id, source, content, timestamp FROM events ORDER BY timestamp DESC"
        ).fetchall()
        event_entity_rows = cursor.execute(
            "SELECT event_id, entity_id FROM event_entity_edges"
        ).fetchall()
        event_thread_rows = cursor.execute(
            "SELECT event_id, thread_id FROM event_thread_edges"
        ).fetchall()

    nodes = []
    links = []
    for row in thread_rows:
        nodes.append(
            {
                "id": f"thread-{row['id']}",
                "kind": "thread",
                "label": row["title"],
                "size": 26,
                "thread_id": int(row["id"]),
                "summary": row["summary"],
            }
        )
    for row in entity_rows:
        nodes.append(
            {
                "id": f"entity-{row['id']}",
                "kind": "entity",
                "label": row["name"],
                "size": 16,
                "entity_id": int(row["id"]),
            }
        )
    for row in event_rows:
        nodes.append(
            {
                "id": f"event-{row['id']}",
                "kind": "event",
                "label": row["content"],
                "size": 10,
                "event_id": int(row["id"]),
                "source": row["source"],
                "timestamp": row["timestamp"],
            }
        )
    for row in event_entity_rows:
        links.append(
            {
                "source": f"event-{row['event_id']}",
                "target": f"entity-{row['entity_id']}",
                "kind": "event_entity",
            }
        )
    for row in event_thread_rows:
        links.append(
            {
                "source": f"event-{row['event_id']}",
                "target": f"thread-{row['thread_id']}",
                "kind": "event_thread",
            }
        )
    return {"nodes": nodes, "links": links}


def build_thread_graph(thread_id: int) -> Optional[dict]:
    subgraph = get_thread_subgraph(thread_id)
    if subgraph is None:
        return None

    thread = subgraph["thread"]
    events = subgraph["events"]
    entities = subgraph["entities"]

    nodes = [
        {
            "id": f"thread-{thread['id']}",
            "kind": "thread",
            "label": thread["title"],
            "size": 30,
            "thread_id": thread["id"],
        }
    ]
    links = []

    for entity in entities:
        nodes.append(
            {
                "id": f"entity-{entity['id']}",
                "kind": "entity",
                "label": entity["name"],
                "size": 18,
            }
        )

    entity_ids = {entity["id"] for entity in entities}
    with db_cursor() as cursor:
        edge_rows = cursor.execute(
            """
            SELECT event_id, entity_id
            FROM event_entity_edges
            WHERE event_id IN (
                SELECT event_id
                FROM event_thread_edges
                WHERE thread_id = ?
            )
            """,
            (thread_id,),
        ).fetchall()

    for event in events:
        nodes.append(
            {
                "id": f"event-{event['id']}",
                "kind": "event",
                "label": event["content"],
                "size": 10,
                "event_id": event["id"],
            }
        )
        links.append(
            {
                "source": f"event-{event['id']}",
                "target": f"thread-{thread['id']}",
                "kind": "event_thread",
            }
        )

    for row in edge_rows:
        if int(row["entity_id"]) not in entity_ids:
            continue
        links.append(
            {
                "source": f"event-{row['event_id']}",
                "target": f"entity-{row['entity_id']}",
                "kind": "event_entity",
            }
        )

    return {"nodes": nodes, "links": links}


def list_threads(query: str = "") -> list[dict]:
    return search_threads(query)


_SELF_OVERVIEW_TRIGGERS = (
    "what have i", "what was i", "what am i", "what i've",
    "tell me about my", "tell me what i", "summarize my", "summary of my",
    "show me my", "remind me what", "remind me of",
    "my work", "my threads", "my events", "my context", "my activity",
    "what i've been", "what i have been",
)


def _is_self_overview(query: str) -> bool:
    q = query.lower().strip()
    return any(trigger in q for trigger in _SELF_OVERVIEW_TRIGGERS)


def answer_query(query: str) -> dict:
    """Route the query through one of three modes:

    1. RETRIEVE — query mentions entities that overlap with a thread →
       ground the LLM response in that thread's subgraph.
    2. SELF-OVERVIEW — query asks about the user's own work in general
       ("what have I been doing?") → ground in the top recent threads.
    3. GENERAL — greeting or unrelated topic → respond conversationally
       with no retrieval. Gemini is told the thread titles so it can
       offer to surface one if relevant, but doesn't fabricate links.
    """
    ranked = search_threads(query)

    # Mode 1: any thread with actual entity overlap → retrieve.
    relevant = [t for t in ranked if t.get("entity_overlap", 0) > 0]
    if relevant:
        top_matches = relevant[:2]
        contexts = []
        for thread in top_matches:
            subgraph = get_thread_subgraph(thread["id"])
            if subgraph is not None:
                contexts.append(subgraph)
        return {
            "query": query,
            "answer": synthesize_answer(query, contexts),
            "threads": top_matches,
            "context": contexts,
            "mode": "retrieve",
        }

    # Mode 2: explicit "what have I been doing" style → overview retrieval.
    # Use an unfiltered thread list (search_threads with a non-matching query
    # would drop everything; for an overview we want all recent threads).
    if _is_self_overview(query):
        all_threads = search_threads("")
        if all_threads:
            top_matches = all_threads[:3]
            contexts = []
            for thread in top_matches:
                subgraph = get_thread_subgraph(thread["id"])
                if subgraph is not None:
                    contexts.append(subgraph)
            return {
                "query": query,
                "answer": synthesize_answer(query, contexts),
                "threads": top_matches,
                "context": contexts,
                "mode": "overview",
            }

    # Mode 3: general assistant — no retrieval, no fabricated grounding.
    return {
        "query": query,
        "answer": synthesize_general_answer(query),
        "threads": [],
        "context": [],
        "mode": "general",
    }


def synthesize_general_answer(query: str) -> str:
    """Conversational LLM response with NO subgraph context.

    Used when the query doesn't match any thread — greetings, unrelated
    questions, generic chat. Gemini is given the thread titles so it can
    point the user toward existing context if relevant, but it must not
    pretend to have retrieved anything.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _general_fallback(query)
    try:
        return _gemini_general_answer(query, api_key)
    except Exception as exc:
        print(f"[meniscus] general Gemini call failed, falling back: {exc}")
        return _general_fallback(query)


def _gemini_general_answer(query: str, api_key: str) -> str:
    from google import genai

    overview = _list_all_threads_overview()

    system_prompt = (
        "You are Meniscus, a context layer for AI agents. The user is talking to you "
        "directly — their message does NOT match any specific thread of their captured "
        "work, so you have NOT retrieved a subgraph for this turn.\n\n"
        "Respond as a friendly, concise general assistant (1-3 sentences). "
        "If their message is a greeting or small talk, greet them back warmly and, "
        "in one short line, mention that you can surface what they've been working on "
        "(reference 1-2 of their thread titles if it fits naturally).\n"
        "If their message is a general question unrelated to their work, answer it briefly "
        "as a normal assistant would — do NOT pretend to have retrieved their context.\n"
        "Never invent events, dates, or details about the user's activity."
    )

    user_prompt = (
        f"User: {query}\n\n"
        f"For your awareness only — the user has these active threads, but they did NOT "
        f"ask about them:\n{overview}"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config={"system_instruction": system_prompt},
    )
    return (response.text or "").strip() or _general_fallback(query)


def _general_fallback(query: str) -> str:
    q = query.lower().strip()
    greetings = ("hi", "hello", "hey", "yo", "thanks", "thank you", "ok", "cool")
    if any(q.startswith(g) for g in greetings) or len(q) < 4:
        return "Hey — ask me about anything you've been working on, or open the Memory Graph to see what I've captured."
    return (
        "I'm Meniscus, your context layer. I don't see a thread that matches that yet — "
        "try asking about something you've been working on."
    )


def synthesize_answer(query: str, contexts: list[dict]) -> str:
    """Generate an answer grounded in the retrieved thread subgraphs.

    When GEMINI_API_KEY is set we route the prompt to Gemini; otherwise we fall
    back to a deterministic string built from the subgraph so the demo is never
    silently broken.
    """
    if not contexts:
        return "Meniscus could not find a relevant thread for that question yet."

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            return _gemini_answer(query, contexts, api_key)
        except Exception as exc:
            # Log to stderr but always degrade gracefully — the demo must keep working.
            print(f"[meniscus] Gemini call failed, falling back: {exc}")

    return _deterministic_answer(query, contexts)


def _format_subgraph_blocks(contexts: list[dict]) -> str:
    blocks = []
    for ctx in contexts:
        thread = ctx["thread"]
        events_text = "\n".join(
            f"  - [{event['source']} · {event['timestamp']}] {event['content']}"
            for event in ctx["events"]
        )
        entity_names = [entity["name"] for entity in ctx["entities"]][:12]
        entities_text = ", ".join(entity_names) if entity_names else "(none)"
        blocks.append(
            f"### Thread: {thread['title']}\n"
            f"Summary: {thread['summary']}\n"
            f"Events ({len(ctx['events'])}):\n{events_text}\n"
            f"Connected entities: {entities_text}"
        )
    return "\n\n".join(blocks)


def _list_all_threads_overview() -> str:
    """All thread titles + entity counts, so the LLM understands the broader landscape."""
    with db_cursor() as cursor:
        rows = cursor.execute(
            """
            SELECT t.id, t.title,
                   (SELECT COUNT(*) FROM event_thread_edges WHERE thread_id = t.id) AS events_count
            FROM threads t
            ORDER BY t.updated_at DESC
            """
        ).fetchall()
    if not rows:
        return "(no other threads)"
    return "\n".join(
        f"- {row['title']} ({row['events_count']} events)" for row in rows
    )


def _gemini_answer(query: str, contexts: list[dict], api_key: str) -> str:
    # Imported lazily so the server can boot even without the SDK installed.
    from google import genai

    system_prompt = (
        "You are Meniscus, a context layer that holds the user's working state "
        "across tools. You will be given (1) an overview of every thread you "
        "currently know about and (2) the full subgraph for the most relevant "
        "thread(s) to the user's question.\n\n"
        "Answer using ONLY this retrieved context — treat it as factual evidence "
        "of what the user has been doing. Your goal is to feel like a colleague "
        "who has actually read their work, not a search engine.\n\n"
        "Rules:\n"
        "- Reference concrete events when relevant (\"you watched...\", \"you asked...\", \"you committed...\").\n"
        "- Connect events within a thread when they form a pattern (e.g. video → question → fix).\n"
        "- If the question naturally spans multiple threads in the overview, weave them together.\n"
        "- Be conversational and specific, not a bullet list. 3-5 sentences.\n"
        "- Never invent details, dates, or events not in the context.\n"
        "- If the retrieved context doesn't actually answer the question, say so plainly."
    )

    user_prompt = (
        f"User question: {query}\n\n"
        f"## All active threads\n{_list_all_threads_overview()}\n\n"
        f"## Retrieved subgraph (most relevant)\n\n"
        f"{_format_subgraph_blocks(contexts)}"
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config={"system_instruction": system_prompt},
    )
    return (response.text or "").strip() or _deterministic_answer(query, contexts)


def _deterministic_answer(query: str, contexts: list[dict]) -> str:
    # MVP fallback when no LLM is configured. In production every answer would
    # be LLM-generated; this keeps the demo coherent if the key is missing.
    lead = contexts[0]
    thread = lead["thread"]
    events = list(reversed(lead["events"]))
    event_lines = "; ".join(event["content"] for event in events[:3])
    clean_entities = [entity["name"] for entity in lead["entities"] if entity["name"] not in STOPWORDS]
    entity_line = ", ".join(clean_entities[:6]) if clean_entities else "no stable entities yet"
    return (
        f"For '{query}', the strongest context is '{thread['title']}'. "
        f"{thread['summary']} "
        f"Key events: {event_lines}. "
        f"Connected entities include {entity_line}."
    )


def build_overview() -> dict:
    with db_cursor() as cursor:
        row = cursor.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM events) AS events_count,
                (SELECT COUNT(*) FROM entities) AS entities_count,
                (SELECT COUNT(*) FROM threads) AS threads_count
            """
        ).fetchone()

    thread_rows = search_threads("")
    return {
        "counts": {
            "events": int(row["events_count"]),
            "entities": int(row["entities_count"]),
            "threads": int(row["threads_count"]),
        },
        "threads": thread_rows,
    }


def refresh_all_threads() -> None:
    with db_cursor() as cursor:
        rows = cursor.execute("SELECT id FROM threads").fetchall()
    for row in rows:
        update_thread_page(int(row["id"]))


def prune_orphan_entities() -> None:
    with db_cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM entities
            WHERE id NOT IN (
                SELECT DISTINCT entity_id
                FROM event_entity_edges
            )
            """
        )


def prune_stopword_entities() -> None:
    placeholders = ", ".join("?" for _ in STOPWORDS)
    with db_cursor() as cursor:
        cursor.execute(
            f"""
            DELETE FROM event_entity_edges
            WHERE entity_id IN (
                SELECT id
                FROM entities
                WHERE name IN ({placeholders})
            )
            """,
            tuple(STOPWORDS),
        )
        cursor.execute(
            f"DELETE FROM entities WHERE name IN ({placeholders})",
            tuple(STOPWORDS),
        )
