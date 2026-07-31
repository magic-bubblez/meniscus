"""Meniscus configuration constants."""

ENTITY_CAP: int = 50

CHUNK_SIZE_WORDS: int = 5000

# Thread assignment (dormant; retune via assignment_log).
HALF_LIFE_HOURS: float = 4.0
ASSIGNMENT_THRESHOLD: float = 0.15
TOPICAL_FLOOR: float = 0.20
TEMPORAL_FLOOR: float = 0.05
VECTOR_CANDIDATE_K: int = 15
ALGORITHM_VERSION: str = "v1"

EMBEDDING_PROVIDER: str = "local"
EMBEDDING_DIMENSIONS: int = 768
HYBRID_ALPHA: float = 0.7

# Retrieval
RETRIEVAL_DEFAULT_LIMIT: int = 60
CANDIDATE_POOL: int = 200
RRF_K: int = 60
W_VECTOR: float = 1.0
W_FTS: float = 0.5
W_ENTITY: float = 0.5
DOOR_PRIOR: float = 0.0
MATCHED_CUT_RULE: str = "ratio"
MATCHED_CUT_RATIO: float = 0.5
MIN_RETURN: int = 10
ENUMERATION_TRIGGERS: frozenset[str] = frozenset(
    {"how many", "how much", "list all", "every", "all of", "total number"}
)
NEIGHBOUR_CUT_RULE: str = "ratio"
NEIGHBOUR_CUT_RATIO: float = 0.75
NEIGHBOUR_MAX: int = 20
NEIGHBOUR_ENABLED: bool = False
MAX_EVENTS_PER_EPISODE: int = 20
SOURCE_COLLAPSE: bool = True

# Episodic reconstruction
SESSION_GAP_SECONDS: int = 1800
EPISODE_WINDOW_SECONDS: int = 6 * 3600
EPISODE_MAX_EVENTS: int = 50

# Bulk import (parallel batch path)
IMPORT_CONCURRENCY: int = 20
EMBED_BATCH_SIZE: int = 100
IMPORT_WINDOW: int = 500
IMPORT_RATELIMIT_RETRIES: int = 4
MAX_RUN_COST_USD: float = 31.00
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
}
DEFAULT_PRICE_USD_PER_MTOK: tuple[float, float] = (1.0, 5.0)

# LLM
DEFAULT_MODEL_PROVIDER: str = "openrouter"
GEMINI_MODEL: str = "gemini-2.5-flash"
OPENROUTER_MODEL: str = "google/gemini-3.5-flash-lite"
ANTHROPIC_MODEL: str = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS: int = 4096
SYNTHESIS_MODEL: str = "anthropic/claude-sonnet-5"
OLLAMA_MODEL: str = "qwen3:4b-instruct"
OLLAMA_URL: str = "http://localhost:11434/v1/chat/completions"
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 1.0

DB_PATH: str = "~/.meniscus/meniscus.db"
