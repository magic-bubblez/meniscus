"""Meniscus configuration constants."""

# Entity extraction
ENTITY_CAP: int = 50

# Chunking
CHUNK_SIZE_WORDS: int = 5000

# Thread assignment
# NOTE: these are calibration knobs, tuned from real assignment_log scores.
# TOPICAL_FLOOR=0.20 means a semantic-ONLY merge (no shared entities) needs
# cosine >= ~0.67 — enough to keep genuinely-related notes while rejecting the
# high baseline similarity (~0.55) of unrelated short notes. Retune with a real
# corpus by reviewing assignment_log.
HALF_LIFE_HOURS: float = 4.0
ASSIGNMENT_THRESHOLD: float = 0.15
TOPICAL_FLOOR: float = 0.20
TEMPORAL_FLOOR: float = 0.05
VECTOR_CANDIDATE_K: int = 15
ALGORITHM_VERSION: str = "v1"

# Embedding
# "gemini" — embeddings use a SEPARATE Gemini quota from generate_content, so
# this keeps working even when the LLM's daily generate cap is exhausted (and it
# runs fine alongside an OpenRouter LLM). This powers the semantic/vector door.
# "local" runs bge-small on CPU (no API); "none" disables vector search.
EMBEDDING_PROVIDER: str = "gemini"
EMBEDDING_DIMENSIONS: int = 768
HYBRID_ALPHA: float = 0.7

# Retrieval
RETRIEVAL_DEFAULT_LIMIT: int = 20
# Max events returned per episode. Bounds the context handed to the caller: the
# events that actually matched the query are ALWAYS kept, and remaining slots are
# filled with the thread's other events in chronological order. Lower = leaner
# context, higher = more surrounding detail. 0 disables the cap.
MAX_EVENTS_PER_EPISODE: int = 20

# Bulk import (parallel batch path)
# How many extract/summarize calls run at once. TUNE THIS PER MODEL/BACKEND — the
# ceiling is the server's serving capacity, NOT the task. Too high on a small
# backend = 429s/stalls; too low on a big one = needlessly slow.
#   local Ollama (one GPU) .............. 1-2
#   budget OpenRouter open model ........ ~5
#   high-capacity API (gemini-flash) .... 15-20+
IMPORT_CONCURRENCY: int = 5
EMBED_BATCH_SIZE: int = 100      # texts per embedding API call
IMPORT_WINDOW: int = 500         # events processed per memory window (bounds RAM)
IMPORT_RATELIMIT_RETRIES: int = 4  # backoff-retries on a 429 during import

# LLM
# "ollama" (local, on-device, no API key), "openrouter" (pay-per-token), or
# "gemini" (direct API).
DEFAULT_MODEL_PROVIDER: str = "openrouter"
# Direct-Gemini model name (used when DEFAULT_MODEL_PROVIDER = "gemini").
GEMINI_MODEL: str = "gemini-2.5-flash"
# OpenRouter model slug (used when DEFAULT_MODEL_PROVIDER = "openrouter").
# Cheap + reliable structured output. Alternatives: google/gemini-2.5-flash-lite
# (cheaper), google/gemini-3.5-flash (stronger), x-ai/grok-... etc.
OPENROUTER_MODEL: str = "google/gemini-2.5-flash"
# Synthesis (final-answer) model for the benchmark. Represents the strong CLIENT
# that answers from retrieved episodes — the product never synthesizes locally.
# Served via OpenRouter (uses OPENROUTER_API_KEY).
SYNTHESIS_MODEL: str = "anthropic/claude-sonnet-5"
# Local Ollama (used when DEFAULT_MODEL_PROVIDER = "ollama"). No API key; runs
# fully on-device. Pull the model first: `ollama pull qwen3:4b-instruct`.
OLLAMA_MODEL: str = "qwen3:4b-instruct"
OLLAMA_URL: str = "http://localhost:11434/v1/chat/completions"
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 1.0

# Database
DB_PATH: str = "~/.meniscus/meniscus.db"
