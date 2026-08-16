"""LongMemEval harness for Meniscus.

For each instance: build a FRESH Meniscus memory from the haystack (each chat
turn -> one event), then retrieve + synthesize an answer to the question.
Emits a hypotheses JSONL (for LongMemEval's evaluate_qa.py) and a retrieval
recall report (did the gold `has_answer` turns come back?).
"""
from __future__ import annotations
import json, re, sys, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(str(ROOT / ".env"))


import meniscus.config as _cfg
_cfg.EMBEDDING_PROVIDER = "local"
_cfg.EMBEDDING_DIMENSIONS = 768

from meniscus.db import get_connection, init_db
from meniscus.event_intake import ingest_event
from meniscus.pipeline import process_pending_events
from meniscus.providers import get_model, get_embedding_model
from meniscus import fact_retrieval as fr
from meniscus.cli.main import SYNTHESIS_PROMPT_TEMPLATE, AskAnswer

# Belt-and-suspenders: re-patch the names each module bound at import time.
import meniscus.db as _db; _db.EMBEDDING_PROVIDER = "local"; _db.EMBEDDING_DIMENSIONS = 768
import meniscus.providers as _p; _p.EMBEDDING_PROVIDER = "local"; _p.EMBEDDING_DIMENSIONS = 768
import meniscus.providers.local_embedding as _le; _le.EMBEDDING_DIMENSIONS = 768


def parse_ts(s: str) -> datetime:
    s2 = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", s).strip()   # drop "(Mon)"
    return datetime.strptime(s2, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def build_memory(conn, instance, model, emb):
    gold: set[int] = set()
    for sess, date in zip(instance["haystack_sessions"], instance["haystack_dates"]):
        base = parse_ts(date)
        for i, turn in enumerate(sess):
            ts = (base + timedelta(seconds=i)).isoformat()
            ids = ingest_event(conn, turn["content"], turn["role"], timestamp=ts)
            if turn.get("has_answer"):
                gold.update(ids)
    process_pending_events(conn, model, emb)
    return gold


def answer_question(conn, question, emb, synth_model):
    facts = fr.retrieve(conn, text=question, limit=30, embedding_model=emb)
    retrieved = {f.event_id for f in facts}
    # Lean payload: only what the synthesizer reads — fact text, time, source.
    payload = [{"timestamp": f.timestamp, "source": f.source, "text": f.text} for f in facts]
    injected = json.dumps(payload)
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(question=question, facts=injected)
    ans = synth_model.generate_structured(prompt, AskAnswer)
    return ans.answer, retrieved, len(facts), len(injected)


def run(dataset_path, selector, out_path):
    full = json.load(open(dataset_path))
    data = [full[i] for i in selector] if isinstance(selector, list) else full[:selector]
    model, emb = get_model(), get_embedding_model()

    from meniscus.providers.openrouter import OpenRouterProvider
    synth = OpenRouterProvider(model=_cfg.MODEL)
    rows = []
    with open(out_path, "w") as fout:
        for k, inst in enumerate(data, 1):
            db = tempfile.mktemp(suffix=".db")
            conn = get_connection(Path(db)); init_db(conn)
            try:
                gold = build_memory(conn, inst, model, emb)
                hyp, retrieved, n_facts, inj_chars = answer_question(conn, inst["question"], emb, synth)
            finally:
                conn.close()
            recall = (len(gold & retrieved) / len(gold)) if gold else None

            full_chars = sum(len(t["content"]) for s in inst["haystack_sessions"] for t in s)
            reduction = (1 - inj_chars / full_chars) if full_chars else 0.0
            fout.write(json.dumps({"question_id": inst["question_id"], "hypothesis": hyp,
                                   "recall": recall, "reduction": reduction,
                                   "injected_tokens": inj_chars // 4, "full_tokens": full_chars // 4}) + "\n")
            rows.append((inst["question_type"], recall, reduction, inj_chars // 4))
            print(f"[{k}/{len(data)}] {inst['question_type']:<24} recall={recall} "
                  f"reduction={reduction:.1%} inj~{inj_chars//4}tok / full~{full_chars//4}tok facts={n_facts}", flush=True)
            print(f"     Q: {inst['question'][:80]}", flush=True)
            print(f"     gold: {str(inst['answer'])[:70]!r}", flush=True)
            print(f"     hyp : {hyp[:70]!r}\n", flush=True)
    recalls = [r for _, r, _, _ in rows if r is not None]
    reductions = [red for _, _, red, _ in rows]
    injs = [it for _, _, _, it in rows]
    if recalls:
        print(f"mean retrieval recall:  {sum(recalls)/len(recalls):.2f}  (n={len(recalls)})")
    if reductions:
        print(f"mean context reduction: {sum(reductions)/len(reductions):.1%}  (n={len(reductions)})")
        print(f"mean injected tokens:   {sum(injs)/len(injs):.0f}  (vs full haystack)")
    print(f"hypotheses -> {out_path}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "bench/data/longmemeval_oracle.json"
    arg = sys.argv[2] if len(sys.argv) > 2 else "3"
    selector = [int(x) for x in arg.split(",")] if "," in arg else int(arg)
    out = sys.argv[3] if len(sys.argv) > 3 else "bench/out/hypotheses.jsonl"
    run(ds, selector, out)
