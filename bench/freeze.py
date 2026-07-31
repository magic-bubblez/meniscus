from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(str(ROOT / ".env"))

# Local embeddings (bge-base, 768d) so ingestion never hits an embedding cap.
# Patch BEFORE importing modules that bind these names.
import config as _cfg

_cfg.EMBEDDING_PROVIDER = "local"
_cfg.EMBEDDING_DIMENSIONS = 768

from db import get_connection, init_db  # noqa: E402
from event_intake import ingest_event  # noqa: E402
from exceptions import SpendCeilingError  # noqa: E402
from pipeline import process_pending_events  # noqa: E402
from spend_gate import estimate_cost_usd  # noqa: E402
from providers import get_embedding_model, get_model  # noqa: E402

import db as _db  # noqa: E402
import providers as _p  # noqa: E402
import providers.local_embedding as _le  # noqa: E402

_db.EMBEDDING_PROVIDER = "local"
_db.EMBEDDING_DIMENSIONS = 768
_p.EMBEDDING_PROVIDER = "local"
_p.EMBEDDING_DIMENSIONS = 768
_le.EMBEDDING_DIMENSIONS = 768

MEM_DIR = ROOT / "bench" / "memories"


def parse_ts(raw: str) -> datetime:
    """LongMemEval dates look like '2023/04/10 (Mon) 17:50'."""

    cleaned = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", raw).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def stratified_indices(data: list, count: int) -> list[int]:
    """Pick `count` instances spread as evenly as possible across question types.

    Round-robins across the types so a partial run still covers every type --
    important because the easy types (single-session) flatter the system and the
    hard ones (multi-session, temporal) are where threads must prove themselves.
    """

    by_type: dict[str, list[int]] = {}
    for index, instance in enumerate(data):
        by_type.setdefault(instance["question_type"], []).append(index)

    types = sorted(by_type)
    picked: list[int] = []
    depth = 0
    while len(picked) < count:
        added = False
        for question_type in types:
            bucket = by_type[question_type]
            if depth < len(bucket) and len(picked) < count:
                picked.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1
    return picked


def freeze_instance(instance: dict, db_path: Path, model, embedding_model, provider: str) -> dict:
    """Build one instance's memory into db_path; return its sidecar metadata."""

    conn = get_connection(db_path)
    init_db(conn)
    gold: set[int] = set()
    turn_count = 0
    try:
        for session, date in zip(instance["haystack_sessions"], instance["haystack_dates"]):
            base = parse_ts(date)
            for offset, turn in enumerate(session):
                timestamp = (base + timedelta(seconds=offset)).isoformat()
                event_ids = ingest_event(
                    conn, turn["content"], turn["role"], timestamp=timestamp
                )
                turn_count += 1
                if turn.get("has_answer"):
                    gold.update(event_ids)
        process_pending_events(conn, model, embedding_model)
        fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        embedded = conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
    finally:
        conn.close()

    return {
        "question_id": instance["question_id"],
        "question": instance["question"],
        "answer": str(instance["answer"]),
        "question_type": instance["question_type"],
        # Which LLM built this memory (extraction is non-deterministic across models).
        "provider": provider,
        "gold_event_ids": sorted(gold),
        "n_turns": turn_count,
        "n_facts": fact_count,
        "n_embedded": embedded,
        "full_haystack_chars": sum(
            len(turn["content"])
            for session in instance["haystack_sessions"]
            for turn in session
        ),
    }


def freeze(dataset_path: str, count: int, shard_index: int = 0, shard_count: int = 1) -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {dataset_path} ...", flush=True)
    data = json.load(open(dataset_path))
    indices = stratified_indices(data, count)

    if shard_count > 1:
        # Shard within each question type so every shard stays stratified.
        seen: dict[str, int] = {}
        sharded: list[int] = []
        for index in indices:
            question_type = data[index]["question_type"]
            position_in_type = seen.get(question_type, 0)
            seen[question_type] = position_in_type + 1
            if position_in_type % shard_count == shard_index:
                sharded.append(index)
        indices = sharded
        print(f"shard {shard_index} of {shard_count}: {len(indices)} instances", flush=True)

    types = {}
    for index in indices:
        types[data[index]["question_type"]] = types.get(data[index]["question_type"], 0) + 1
    print(f"freezing {len(indices)} instances into {MEM_DIR}", flush=True)
    print(f"  stratified across types: {types}", flush=True)

    _override = os.environ.get("MENISCUS_FREEZE_MODEL")
    if _override:
        from providers.openrouter import OpenRouterProvider
        model, model_slug = OpenRouterProvider(model=_override), _override
    else:
        model, model_slug = get_model(), _cfg.OPENROUTER_MODEL
    embedding_model = get_embedding_model()
    provider_label = _cfg.DEFAULT_MODEL_PROVIDER
    budget = _cfg.MAX_RUN_COST_USD
    spent = 0.0
    print(f"llm provider: {provider_label} | model {model_slug} | budget ${budget:.2f}", flush=True)
    started = time.time()
    done = 0

    for position, index in enumerate(indices, 1):
        instance = data[index]
        question_id = instance["question_id"]
        db_path = MEM_DIR / f"{question_id}.db"
        sidecar_path = MEM_DIR / f"{question_id}.json"

        if db_path.exists() and sidecar_path.exists():
            print(f"[{position}/{len(indices)}] {question_id} already frozen - skip", flush=True)
            continue

        items = [
            (turn["role"], turn["content"], (parse_ts(date) + timedelta(seconds=offset)).isoformat())
            for session, date in zip(instance["haystack_sessions"], instance["haystack_dates"])
            for offset, turn in enumerate(session)
        ]
        est = estimate_cost_usd(items, model_slug)
        if budget > 0 and spent + est > budget:
            print(
                f"budget ${budget:.2f} reached: next instance ~${est:.2f} would total "
                f"${spent + est:.2f}. Stopping after {done} frozen (~${spent:.2f} spent).",
                flush=True,
            )
            break

        # Claim the instance so parallel shards can never build the same one.
        lock_path = MEM_DIR / f"{question_id}.building"
        try:
            lock_path.touch(exist_ok=False)
        except FileExistsError:
            print(f"[{position}/{len(indices)}] {question_id} claimed by another process - skip", flush=True)
            continue

        instance_started = time.time()
        try:
            meta = freeze_instance(instance, db_path, model, embedding_model, provider_label)
        except SpendCeilingError:  # loud, whole-run abort — never a silent retry loop
            db_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # keep going; a failed instance retries next run
            print(f"[{position}/{len(indices)}] {question_id} FAILED: {exc!r}", flush=True)
            db_path.unlink(missing_ok=True)
            continue
        finally:
            lock_path.unlink(missing_ok=True)

        # No facts/embeddings means the extractor was unavailable; discard to retry.
        if meta["n_facts"] == 0 or meta["n_embedded"] == 0:
            print(
                f"[{position}/{len(indices)}] {question_id} EMPTY "
                f"(facts={meta['n_facts']}, embedded={meta['n_embedded']}) -- "
                "extractor unavailable; discarding so it retries",
                flush=True,
            )
            db_path.unlink(missing_ok=True)
            continue

        sidecar_path.write_text(json.dumps(meta, indent=2))
        done += 1
        spent += est
        elapsed = time.time() - instance_started
        total = time.time() - started
        print(
            f"[{position}/{len(indices)}] {question_id} ({meta['question_type']}) "
            f"{meta['n_turns']} turns -> {meta['n_facts']} facts, "
            f"gold={len(meta['gold_event_ids'])}, embedded={meta['n_embedded']} "
            f"| ~${est:.2f} (~${spent:.2f}/${budget:.2f}) "
            f"| {elapsed:.0f}s (avg {total/done:.0f}s, total {total/60:.1f}m)",
            flush=True,
        )

    print(f"\nfrozen memories in {MEM_DIR} - reusable indefinitely, no rebuild needed.", flush=True)


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "bench/data/longmemeval_s.json")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    shard_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    shard_count = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    freeze(dataset, n, shard_index, shard_count)
