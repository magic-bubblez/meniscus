"""Meniscus configuration. Every tunable setting lives here; secrets live in .env."""

# LLM -- OpenRouter is the only provider; models are OpenRouter slugs
DEFAULT_MODEL_PROVIDER: str = "openrouter"
OPENROUTER_MODEL: str = "google/gemini-3.5-flash-lite"
SYNTHESIS_MODEL: str = "deepseek/deepseek-v4-flash"
MAX_OUTPUT_TOKENS: int = 4096
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 1.0

# Embeddings -- run locally, no API
EMBEDDING_PROVIDER: str = "local"
EMBEDDING_DIMENSIONS: int = 768

# Storage
DB_PATH: str = "~/.meniscus/meniscus.db"

# Retrieval -- fusion across the three doors (vector, keyword, entity)
RETRIEVAL_DEFAULT_LIMIT: int = 30
CANDIDATE_POOL: int = 200
RRF_K: int = 60
W_VECTOR: float = 1.0
W_FTS: float = 0.5
W_ENTITY: float = 0.5
DOOR_PRIOR: float = 0.0

# Retrieval -- adaptive cut on the fused-score curve
MATCHED_CUT_RULE: str = "ratio"
MATCHED_CUT_RATIO: float = 0.5
MIN_RETURN: int = 10
ENUMERATION_TRIGGERS: frozenset[str] = frozenset(
    {"how many", "how much", "list all", "every", "all of", "total number"}
)
SOURCE_COLLAPSE: bool = True

# Retrieval -- read-time neighbour gathering (opt-in)
NEIGHBOUR_ENABLED: bool = True
NEIGHBOUR_CUT_RULE: str = "ratio"
NEIGHBOUR_CUT_RATIO: float = 0.75
NEIGHBOUR_MAX: int = 20
VECTOR_CANDIDATE_K: int = 15
HYBRID_ALPHA: float = 0.7

# Episodic reconstruction -- read-time session segmentation
SESSION_GAP_SECONDS: int = 1800
EPISODE_WINDOW_SECONDS: int = 6 * 3600
EPISODE_MAX_EVENTS: int = 50

# Ingestion
CHUNK_SIZE_WORDS: int = 5000
ENTITY_CAP: int = 50

# Bulk import (parallel batch path)
IMPORT_CONCURRENCY: int = 20
EMBED_BATCH_SIZE: int = 100
IMPORT_WINDOW: int = 500
IMPORT_RATELIMIT_RETRIES: int = 4

# Spend guard
MAX_RUN_COST_USD: float = 31.00
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
}
DEFAULT_PRICE_USD_PER_MTOK: tuple[float, float] = (1.0, 5.0)
