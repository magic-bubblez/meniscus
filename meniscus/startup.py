"""Startup helpers shared by CLI, API, and MCP entry points."""

from __future__ import annotations

import sys

from meniscus import config

_EMBEDDING_NOTICE = (
    "Embeddings disabled -- vector search is unavailable; "
    "retrieval falls back to keyword and entity matching."
)
_embedding_notice_shown = False


def announce_embedding_state() -> None:
    """Print the embedding-disabled notice once per process."""

    global _embedding_notice_shown
    if _embedding_notice_shown:
        return
    if config.EMBEDDING_PROVIDER == "none":
        print(_EMBEDDING_NOTICE, file=sys.stderr)
    _embedding_notice_shown = True
