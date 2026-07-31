"""End-to-end answer accuracy + context reduction over the frozen memories, per type."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(str(ROOT / ".env"))

import config as _cfg

_cfg.EMBEDDING_PROVIDER = "local"
_cfg.EMBEDDING_DIMENSIONS = 768

from db import get_connection, init_db  # noqa: E402
from providers import get_embedding_model  # noqa: E402
from providers.openrouter import OpenRouterProvider  # noqa: E402
import fact_retrieval as fr  # noqa: E402
from tokens import estimate_tokens  # noqa: E402

import db as _db  # noqa: E402
import providers as _p  # noqa: E402
import providers.local_embedding as _le  # noqa: E402

_db.EMBEDDING_PROVIDER = "local"
_db.EMBEDDING_DIMENSIONS = 768
_p.EMBEDDING_PROVIDER = "local"
_p.EMBEDDING_DIMENSIONS = 768
_le.EMBEDDING_DIMENSIONS = 768

MEM_DIR = ROOT / "bench" / "memories"
READER_MODEL = os.environ.get("EVAL_READER", "google/gemini-3.5-flash-lite")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE", "google/gemini-3.5-flash-lite")
PRICES_USD_PER_MTOK = {
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "openai/gpt-5": (1.25, 10.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "google/gemini-3-flash-preview": (0.50, 3.00),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
}
MAX_EVAL_COST_USD = 2.00

READER_PROMPT = """You are this person's memory. Answer the question using ONLY the facts below. Be concise and direct. If the facts do not contain the answer, reply exactly "I don't know".

FACTS:
{facts}

QUESTION: {question}
ANSWER:"""

JUDGE_PROMPT = """Grade whether the model answer is correct given the gold answer. They match if they convey the same key information, even if worded differently or with extra detail.

QUESTION: {question}
GOLD ANSWER: {gold}
MODEL ANSWER: {answer}

Return correct=true only if the model answer conveys the gold answer's meaning."""


class ReaderAnswer(BaseModel):
    answer: str


class JudgeVerdict(BaseModel):
    correct: bool


def _cost(prompt: str, out_tokens: int, model: str) -> float:
    price_in, price_out = PRICES_USD_PER_MTOK.get(model, (1.0, 5.0))
    return (estimate_tokens(prompt) * price_in + out_tokens * price_out) / 1_000_000


def main(limit: int | None = None) -> None:
    embedding_model = get_embedding_model()
    reader = OpenRouterProvider(model=READER_MODEL)
    judge = OpenRouterProvider(model=JUDGE_MODEL)

    per_type: dict[str, list[float]] = {}
    reductions: list[float] = []
    spent = 0.0

    sidecars = [s for s in sorted(MEM_DIR.glob("*.json")) if json.loads(s.read_text()).get("gold_event_ids")]
    if limit:
        sidecars = sidecars[:limit]

    for i, sidecar in enumerate(sidecars, 1):
        meta = json.loads(sidecar.read_text())
        conn = get_connection(sidecar.with_suffix(".db"))
        init_db(conn)
        query_embedding = fr._embed_query(embedding_model, meta["question"])
        facts = fr.search_facts(
            conn,
            text=meta["question"],
            limit=_cfg.RETRIEVAL_DEFAULT_LIMIT,
            query_embedding=query_embedding,
        )
        conn.close()

        facts_block = "\n".join(
            f"- {(getattr(f, 'timestamp', '') or '')[:10]} {f.text}".strip() for f in facts
        )
        reader_prompt = READER_PROMPT.format(facts=facts_block, question=meta["question"])
        answer = reader.generate_structured(reader_prompt, ReaderAnswer).answer
        spent += _cost(reader_prompt, estimate_tokens(answer), READER_MODEL)

        judge_prompt = JUDGE_PROMPT.format(question=meta["question"], gold=meta["answer"], answer=answer)
        correct = judge.generate_structured(judge_prompt, JudgeVerdict).correct
        spent += _cost(judge_prompt, 5, JUDGE_MODEL)

        per_type.setdefault(meta["question_type"], []).append(1.0 if correct else 0.0)
        injected = sum(len(f.text) for f in facts) // 4
        haystack = max(1, meta["full_haystack_chars"] // 4)
        reductions.append(1.0 - injected / haystack)

        print(f"  [{i}/{len(sidecars)}] {meta['question_type']:<26} {'OK ' if correct else 'MISS'}  (~${spent:.3f})", flush=True)
        if spent > MAX_EVAL_COST_USD:
            print(f"STOP: hit ${MAX_EVAL_COST_USD:.2f} cap at instance {i}.", flush=True)
            break

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    all_scores = [s for scores in per_type.values() for s in scores]
    print(f"\nn={len(all_scores)} | reader={READER_MODEL} | judge={JUDGE_MODEL}\n")
    print(f"{'type':<28}{'answer acc':>12}{'n':>5}")
    print("-" * 45)
    for qtype in sorted(per_type):
        print(f"{qtype:<28}{mean(per_type[qtype]):>12.0%}{len(per_type[qtype]):>5}")
    print("-" * 45)
    print(f"{'OVERALL':<28}{mean(all_scores):>12.0%}{len(all_scores):>5}")
    print(f"\navg context reduction: {mean(reductions):.1%}")
    print(f"estimated cost: ${spent:.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
