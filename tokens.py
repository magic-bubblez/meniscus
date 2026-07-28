
from __future__ import annotations

TOKEN_ESTIMATE_DIVISOR: int = 4


def estimate_tokens(text: str) -> int:
    return len(text) // TOKEN_ESTIMATE_DIVISOR
