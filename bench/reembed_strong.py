
from __future__ import annotations

import shutil
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(str(ROOT / ".env"))

from sentence_transformers import SentenceTransformer

from meniscus.db import get_connection

MODEL_NAME = "BAAI/bge-base-en-v1.5"
SRC = ROOT / "bench" / "memories"
DST = ROOT / "bench" / "memories_g768"
BATCH = 64


def _load_vec0(conn) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def reembed(db_path: Path, model) -> int:
    conn = get_connection(db_path)
    _load_vec0(conn)
    conn.execute("UPDATE meta SET value = '768' WHERE key = 'embedding_dimensions'")
    conn.execute("DROP TABLE IF EXISTS fact_embeddings")
    conn.execute(
        "CREATE VIRTUAL TABLE fact_embeddings USING vec0("
        "fact_id INTEGER PRIMARY KEY, embedding float[768] distance_metric=cosine)"
    )
    rows = conn.execute("SELECT id, text FROM facts ORDER BY id").fetchall()
    vectors = model.encode(
        [r[1] for r in rows], batch_size=BATCH, normalize_embeddings=True
    )
    for row, vector in zip(rows, vectors):
        blob = struct.pack(f"{len(vector)}f", *vector.tolist())
        conn.execute(
            "INSERT INTO fact_embeddings (fact_id, embedding) VALUES (?, ?)",
            (int(row[0]), blob),
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
    conn.close()
    return count


def main() -> None:
    DST.mkdir(exist_ok=True)
    sidecars = sorted(SRC.glob("*.json"))
    print(f"loading {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    print(f"re-embedding {len(sidecars)} memories on bge-base-768d -> {DST}", flush=True)
    started = time.time()
    for position, sidecar in enumerate(sidecars, 1):
        stem = sidecar.stem
        dst_db = DST / f"{stem}.db"
        dst_json = DST / f"{stem}.json"
        if dst_db.exists() and dst_json.exists():
            print(f"[{position}/{len(sidecars)}] {stem} exists - skip", flush=True)
            continue
        shutil.copy(SRC / f"{stem}.db", dst_db)
        shutil.copy(sidecar, dst_json)
        count = reembed(dst_db, model)
        print(
            f"[{position}/{len(sidecars)}] {stem} -> {count} facts (768d) "
            f"| {time.time() - started:.0f}s",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
