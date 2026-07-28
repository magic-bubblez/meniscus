-- =============================================================
-- SOURCE OF TRUTH
-- =============================================================

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    content_hash TEXT NOT NULL,
    source_id TEXT,
    UNIQUE(content_hash)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    normalized_form TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    alias TEXT NOT NULL,
    normalized_form TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS event_entity_edges (
    event_id INTEGER NOT NULL REFERENCES events(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (event_id, entity_id)
);

-- Provenance ledger: one row per extraction run over an event (append-only).
CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    extracted_at TEXT NOT NULL
);

-- Distilled atomic facts: the retrieval unit; time inherited from events(timestamp).
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    extraction_id INTEGER NOT NULL REFERENCES extractions(id),
    text TEXT NOT NULL,
    position INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

-- Claim-level entity links (which entities each fact involves), built deterministically.
CREATE TABLE IF NOT EXISTS fact_entity_edges (
    fact_id INTEGER NOT NULL REFERENCES facts(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (fact_id, entity_id)
);

-- =============================================================
-- DERIVED STATE
-- =============================================================

CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_thread_edges (
    event_id INTEGER NOT NULL REFERENCES events(id),
    thread_id INTEGER NOT NULL REFERENCES threads(id),
    PRIMARY KEY (event_id)
);

CREATE TABLE IF NOT EXISTS assignment_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    assigned_thread_id INTEGER NOT NULL REFERENCES threads(id),
    candidate_scores TEXT NOT NULL,
    threshold REAL NOT NULL,
    half_life REAL NOT NULL,
    algorithm_version TEXT NOT NULL,
    entity_snapshot TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- =============================================================
-- INDEXES
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_events_extraction_status ON events(extraction_status);
CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(content_hash);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source_id ON events(source_id);
CREATE INDEX IF NOT EXISTS idx_entities_normalized_form ON entities(normalized_form);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_normalized_form ON entity_aliases(normalized_form);
CREATE INDEX IF NOT EXISTS idx_event_entity_edges_entity_id ON event_entity_edges(entity_id);
CREATE INDEX IF NOT EXISTS idx_event_thread_edges_thread_id ON event_thread_edges(thread_id);
CREATE INDEX IF NOT EXISTS idx_facts_event_id ON facts(event_id);
CREATE INDEX IF NOT EXISTS idx_facts_extraction_id ON facts(extraction_id);
CREATE INDEX IF NOT EXISTS idx_extractions_event_id ON extractions(event_id);
CREATE INDEX IF NOT EXISTS idx_fact_entity_edges_entity_id ON fact_entity_edges(entity_id);

-- =============================================================
-- FULL-TEXT SEARCH
-- =============================================================

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    content,
    content='events',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS events_fts_insert AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS events_fts_delete AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, content)
        VALUES('delete', old.id, old.content);
END;

-- Text door over facts (auto-populated by triggers).
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text,
    content='facts',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS facts_fts_insert AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_delete AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text)
        VALUES('delete', old.id, old.text);
END;

-- =============================================================
-- VECTOR SEARCH
-- =============================================================

-- distance_metric=cosine: KNN must rank by direction, not magnitude. Reduced-
-- dimension embeddings (e.g. gemini-embedding-001 at 768) are NOT normalized,
-- so the default L2 metric would rank by magnitude and pick the wrong
-- neighbors. Cosine here matches the cosine_similarity used in scoring.
CREATE VIRTUAL TABLE IF NOT EXISTS event_embeddings USING vec0(
    event_id INTEGER PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSIONS}] distance_metric=cosine
);

-- Semantic door over facts (populated when fact-embedding is wired in).
CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(
    fact_id INTEGER PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSIONS}] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
