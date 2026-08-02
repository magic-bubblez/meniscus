"""Deterministic pre-flight cost gate for bulk extraction runs (no LLM calls)."""

from __future__ import annotations

import os

from meniscus import config
from meniscus.entity_extractor import build_extraction_prompt
from meniscus.exceptions import SpendCeilingError
from meniscus.tokens import estimate_tokens

ALLOW_LARGE_RUN_ENV: str = "MENISCUS_ALLOW_LARGE_RUN"
_FALSEY = {"", "0", "false", "no", "off"}


def _price(model: str) -> tuple[float, float]:
    return config.MODEL_PRICES_USD_PER_MTOK.get(model, config.DEFAULT_PRICE_USD_PER_MTOK)


def estimate_cost_usd(items: list[tuple[str, str, str]], model: str) -> float:
    """Estimate USD to extract (source, content, timestamp) items on `model` (facts ≈ content)."""

    in_price, out_price = _price(model)
    input_tokens = 0
    output_tokens = 0
    for source, content, timestamp in items:
        input_tokens += estimate_tokens(build_extraction_prompt(source, content, timestamp))
        output_tokens += estimate_tokens(content)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _override_active(allow_large_run: bool) -> bool:
    if allow_large_run:
        return True
    value = os.environ.get(ALLOW_LARGE_RUN_ENV)
    return value is not None and value.strip().lower() not in _FALSEY


def enforce_spend_ceiling(
    pending: list[tuple[int, str, str, str]],
    model: str,
    allow_large_run: bool = False,
) -> float:
    """Abort before spending if the run's estimated cost exceeds config.MAX_RUN_COST_USD; return the estimate."""

    items = [(source, content, timestamp) for _event_id, source, content, timestamp in pending]
    estimate = estimate_cost_usd(items, model)
    ceiling = config.MAX_RUN_COST_USD
    if ceiling > 0 and estimate > ceiling and not _override_active(allow_large_run):
        raise SpendCeilingError(
            f"Estimated ${estimate:.2f} to extract {len(pending)} pending event(s) on {model} "
            f"exceeds MAX_RUN_COST_USD=${ceiling:.2f}. Aborted before any model call. To proceed: "
            f"set {ALLOW_LARGE_RUN_ENV}=1, pass allow_large_run=True, or raise config.MAX_RUN_COST_USD."
        )
    return estimate
