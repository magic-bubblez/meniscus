# Meniscus — Design

A local, structured, long-term memory for AI agents and people. Meniscus receives
unstructured text (chat turns, notes, imported files), distills it into **atomic facts**,
stores everything **append-only** in one SQLite file, and makes it retrievable through a
**deterministic hybrid search** — no LLM in the read path. Any MCP-capable agent reads and
writes the same memory.

---

## 1. Principles

- **Append-only.** Raw events are never overwritten or deleted. Facts are only ever added.
  Every fact traces back to the exact source event it came from, so nothing is lost and
  everything is recoverable.
- **Deterministic retrieval.** Reading memory is a fixed formula over indexes — no model
  call. The same query against the same data always returns the same result, and every
  ranking decision is inspectable with plain SQL.
- **Local and single-file.** One SQLite database on the user's machine. No server, no
  cloud, no account. Back it up by copying a file.
- **Small models at the edges.** An LLM is used only at *write* time (to distill facts) and
  optionally at *read* time (to phrase an answer). The retrieval itself never calls a model,
  so a small/cheap model is enough and nothing heavy runs locally.

## 2. Data model

All state lives in one SQLite file:

- **`events`** — raw ingested content, verbatim, timestamped, with a `source`. The source of
  truth. `extraction_status` tracks whether facts have been distilled yet.
- **`facts`** — atomic statements distilled from an event by the LLM. Each fact links to its
  `event_id` (provenance) and carries its text and position.
- **`entities`** — canonical named things (people, projects, tools) with a normalized form
  for matching; `entity_aliases` maps surface variants to a canonical entity.
- **`fact_entity_edges`** — which entities each fact mentions (the entity retrieval door).
- **`extractions`** — one row per distillation run (provider, model, prompt version), so the
  provenance of every fact is auditable.
- **`fact_embeddings`** — a `sqlite-vec` (vec0) virtual table holding each fact's vector.
- **`facts_fts`** — an FTS5 virtual table over fact text (the keyword door), kept in sync.

Facts and their embeddings/edges are only appended. Correcting the record means adding a new
event, never mutating an old one.

## 3. Ingestion pipeline

1. **Intake** — content is stored as one or more `events` (large inputs are chunked). Intake
   always commits first, so an event is durable even if distillation later fails. Duplicate
   content is deduplicated by a content hash of `(source, content)`.
2. **Distillation** — a small LLM (via whichever provider is configured — see §7) reads each
   pending event and extracts atomic facts. Filler turns with nothing worth keeping produce
   zero facts and are marked complete (not retried forever).
3. **Entity reconciliation** — entities named by the facts are resolved to canonical rows
   (via normalization + aliases) and linked through `fact_entity_edges`.
4. **Embedding** — each fact is embedded locally with `bge-base-en-v1.5` (768-d) and written
   to `fact_embeddings`; `facts_fts` indexes the text.

Processing runs as a parallel batch and is resumable: if the model becomes unavailable,
processed events stay done and the rest remain pending for the next run.

## 4. Retrieval — deterministic hybrid fusion

Retrieval scores candidate facts through three independent **doors**, then fuses them. No
model is involved.

- **Vector door** — cosine similarity between the query embedding and each fact embedding.
  Queries are embedded with an instruction prefix; facts are embedded plain. A wide candidate
  pool is fetched (far more than the return limit) so fusion sees a broad set.
- **Keyword door** — FTS5 (BM25) over fact text. The query is tokenized and OR-matched, so a
  fact matching *any* term is a candidate; the fact's *rank position* feeds fusion.
- **Entity door** — facts linked to an explicitly named entity (used when a caller passes an
  entity argument).

**Fusion (magnitude noisy-OR).** Each door contributes independently; a single strong door
can win on its own merit:

```
fused = 1 − (1 − W_VECTOR·vector) · (1 − W_FTS·fts) · (1 − W_ENTITY·entity)
```

where `vector` is the raw cosine, `fts` is a reciprocal-rank term `RRF_K/(RRF_K + position)`,
and `entity` is 1 if the entity door matched. Vector is the senior partner; keyword is a
booster/tiebreaker.

**Adaptive cut.** Results are sorted by fused score and cut on the *shape* of the curve, not
a fixed `k`: keep every fact scoring at least a ratio of the top score, bounded to
`[MIN_RETURN, limit]`. A sharp, specific query returns a tight set; a broad query returns
more. Counting/enumeration questions (detected by trigger phrases like "how many", "every")
bypass the ratio cut and return the whole plateau.

**Source collapse.** If the facts selected from one event cost as many tokens as the raw
event, the raw event is returned instead — full fidelity when distillation didn't save
anything.

## 5. Episodic reconstruction

Meniscus does **not** store episodes or threads. Instead, "what happened around this moment"
is reconstructed *at read time*: given an anchor (a matched fact, an event id, or a
timestamp), events in a time window are segmented into sessions by gaps longer than
`SESSION_GAP_SECONDS`, and the relevant segment is returned with its facts. Episodes are a
read-time view over the append-only events, never materialized state.

## 6. Interfaces

- **CLI (`men`)** — `init` (interactive setup: provider, key, capture, MCP wiring), `add`,
  `import`, `process`, `ask`, `watch` (silent transcript capture), `sql`, `tables`, `status`,
  `doctor`, `help`, and event inspection (`list events`, `show event`). `men ask` plans a
  query, retrieves deterministically, and phrases an answer.
- **MCP** — exactly **two** tools, so an agent's surface stays minimal:
  - `meniscus_recall` — the single read entry point. One call; the argument selects the
    operation (topic search, a time window, a reconstructed session, or the raw source
    behind a fact). Returns structured facts with timestamps and source ids — not a finished
    answer.
  - `meniscus_log` — an invisible background write; the agent logs what's worth remembering
    without announcing it.

## 7. Providers

- **LLM** — bring your own key. Any OpenAI-compatible backend works (OpenRouter, OpenAI,
  Gemini, Groq, xAI, Hugging Face, a local Ollama server, or any other OpenAI-compatible
  endpoint via `provider = "custom"`); Anthropic is wired separately against Claude's native
  API.
- **Embeddings** — `bge-base-en-v1.5` run **locally** (768-d), no API. Query embeddings use a
  retrieval instruction prefix.

## 8. What makes it inspectable

Because it is one SQLite file and retrieval is a fixed formula, nothing is hidden. `men sql`
and `men tables` expose the raw events, facts, entities, edges, and scores — the exact data
an agent sees. Every fact links to its source event; every extraction records the model that
produced it. Memory you can audit is memory you can trust.
