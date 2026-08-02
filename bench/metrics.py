"""Recall and set-completeness (all@k) over the frozen memories, per type."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(str(ROOT / ".env"))

import meniscus.config as _cfg

_cfg.EMBEDDING_PROVIDER = "local"
_cfg.EMBEDDING_DIMENSIONS = 768

from meniscus.db import get_connection, init_db  # noqa: E402
from meniscus.providers import get_embedding_model  # noqa: E402
from meniscus import fact_retrieval as fr  # noqa: E402

import meniscus.db as _db  # noqa: E402
import meniscus.providers as _p  # noqa: E402
import meniscus.providers.local_embedding as _le  # noqa: E402

_db.EMBEDDING_PROVIDER = "local"
_db.EMBEDDING_DIMENSIONS = 768
_p.EMBEDDING_PROVIDER = "local"
_p.EMBEDDING_DIMENSIONS = 768
_le.EMBEDDING_DIMENSIONS = 768

MEM_DIR = ROOT / "bench" / "memories"


def main() -> None:
    embedding_model = get_embedding_model()
    per_type: dict[str, list[tuple[float, float]]] = {}
    tokens: list[int] = []

    for sidecar in sorted(MEM_DIR.glob("*.json")):
        meta = json.loads(sidecar.read_text())
        gold = set(meta["gold_event_ids"])
        if not gold:
            continue
        conn = get_connection(sidecar.with_suffix(".db"))
        init_db(conn)
        facts = fr.retrieve(
            conn,
            text=meta["question"],
            limit=_cfg.RETRIEVAL_DEFAULT_LIMIT,
            embedding_model=embedding_model,
        )
        returned = {f.event_id for f in facts}
        recall = len(gold & returned) / len(gold)
        complete = 1.0 if gold <= returned else 0.0
        per_type.setdefault(meta["question_type"], []).append((recall, complete))
        tokens.append(sum(len(f.text) for f in facts) // 4)
        conn.close()

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    rows = [(r, c) for pairs in per_type.values() for (r, c) in pairs]
    print(f"n={len(rows)} frozen memories | limit={_cfg.RETRIEVAL_DEFAULT_LIMIT}\n")
    print(f"{'type':<28}{'avg recall':>11}{'set-complete':>14}{'n':>4}")
    print("-" * 57)
    for qtype in sorted(per_type):
        pairs = per_type[qtype]
        print(
            f"{qtype:<28}{mean([r for r, _ in pairs]):>11.2f}"
            f"{mean([c for _, c in pairs]):>13.0%}{len(pairs):>4}"
        )
    print("-" * 57)
    print(
        f"{'OVERALL':<28}{mean([r for r, _ in rows]):>11.2f}"
        f"{mean([c for _, c in rows]):>13.0%}{len(rows):>4}"
    )
    print(f"\navg injected tokens: {mean([float(t) for t in tokens]):.0f}")


if __name__ == "__main__":
    main()
