"""Meniscus configuration. Every tunable setting lives here; secrets live in .env.

User overrides go in ~/.meniscus/config.toml — any key set there wins over the defaults below.
"""

import tomllib
from dataclasses import dataclass
from importlib.resources import files

from meniscus.home import CONFIG_FILENAME, MENISCUS_HOME


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    base_url: str | None
    api_key_env: str | None
    auth_url: str | None = None
    max_tokens_field: str = "max_tokens"


DEFAULT_MODEL_PROVIDER: str | None = None
CUSTOM_PROVIDER = "custom"
with files("meniscus").joinpath("provider_defaults.toml").open("rb") as _f:
    PROVIDER_DEFAULT_MODELS: dict[str, str] = tomllib.load(_f)["models"]

PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1/key",
    ),
    "openai": ProviderPreset(
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        max_tokens_field="max_completion_tokens",
    ),
    "anthropic": ProviderPreset(None, "ANTHROPIC_API_KEY"),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
    ),
    "groq": ProviderPreset("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "xai": ProviderPreset("https://api.x.ai/v1", "XAI_API_KEY"),
    "huggingface": ProviderPreset("https://router.huggingface.co/v1", "HF_TOKEN"),
    "ollama": ProviderPreset("http://localhost:11434/v1", None),
}
CUSTOM_BASE_URL: str | None = None
CUSTOM_API_KEY_ENV: str | None = None

MODEL: str | None = None
MAX_OUTPUT_TOKENS: int = 4096
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 1.0
MODEL_VALIDATION_TIMEOUT_SECONDS: float = 15.0
MODEL_REQUEST_TIMEOUT_SECONDS: float = 120.0

# Embeddings -- run locally, no API
EMBEDDING_DISABLED = "none"  # deliberate keyword-only mode, not a failure
EMBEDDING_PROVIDER: str = "local"
EMBEDDING_DIMENSIONS: int = 768

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
MAX_RUN_COST_USD: float = 20.00
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {}
DEFAULT_PRICE_USD_PER_MTOK: tuple[float, float] = (1.0, 5.0)

# ── user config override (config.toml wins over every default above) ──────────
_USER_CONFIG_PATH = MENISCUS_HOME / CONFIG_FILENAME

if _USER_CONFIG_PATH.exists():
    with open(_USER_CONFIG_PATH, "rb") as _f:
        _cfg = tomllib.load(_f)

    _model = _cfg.get("model", {})
    if "provider" in _model:
        DEFAULT_MODEL_PROVIDER = _model["provider"]
    if "base_url" in _model:
        CUSTOM_BASE_URL = _model["base_url"]
    if "api_key_env" in _model:
        CUSTOM_API_KEY_ENV = _model["api_key_env"]
    if "model" in _model:
        MODEL = _model["model"]
    elif "extraction_model" in _model:
        MODEL = _model["extraction_model"]
    elif "synthesis_model" in _model:
        MODEL = _model["synthesis_model"]

    _spend = _cfg.get("spend", {})
    if "max_run_cost_usd" in _spend:
        MAX_RUN_COST_USD = float(_spend["max_run_cost_usd"])

    _retrieval = _cfg.get("retrieval", {})
    if "default_limit" in _retrieval:
        RETRIEVAL_DEFAULT_LIMIT = int(_retrieval["default_limit"])
    if "neighbour_max" in _retrieval:
        NEIGHBOUR_MAX = int(_retrieval["neighbour_max"])

    _ingestion = _cfg.get("ingestion", {})
    if "concurrency" in _ingestion:
        IMPORT_CONCURRENCY = int(_ingestion["concurrency"])
