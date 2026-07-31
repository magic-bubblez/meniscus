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
-- INDEXES
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_events_extraction_status ON events(extraction_status);
CREATE INDEX IF NOT EXISTS idx_events_content_hash ON events(content_hash);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source_id ON events(source_id);
CREATE INDEX IF NOT EXISTS idx_entities_normalized_form ON entities(normalized_form);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_normalized_form ON entity_aliases(normalized_form);
CREATE INDEX IF NOT EXISTS idx_facts_event_id ON facts(event_id);
CREATE INDEX IF NOT EXISTS idx_facts_extraction_id ON facts(extraction_id);
CREATE INDEX IF NOT EXISTS idx_extractions_event_id ON extractions(event_id);
CREATE INDEX IF NOT EXISTS idx_fact_entity_edges_entity_id ON fact_entity_edges(entity_id);

-- =============================================================
-- FULL-TEXT SEARCH
-- =============================================================

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

-- Semantic door over facts (populated when fact-embedding is wired in).
-- distance_metric=cosine: KNN must rank by direction, not magnitude.
CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(
    fact_id INTEGER PRIMARY KEY,
    embedding float[{EMBEDDING_DIMENSIONS}] distance_metric=cosine
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
