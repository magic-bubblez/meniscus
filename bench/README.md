# Benchmarks

Meniscus is measured on [LongMemEval-S](https://arxiv.org/abs/2410.10813), a long-term-memory
benchmark of long chat histories with labelled evidence turns. The headline numbers in the
top-level README come from the scripts here, on a 100-instance stratified sample.

## Reproducing the numbers

### 1. Get the dataset

Download LongMemEval-S (`longmemeval_s.json`) from the official release linked in the
[paper](https://arxiv.org/abs/2410.10813) and place it at:

```
bench/data/longmemeval_s.json
```

### 2. Build the frozen memories (one-time; costs LLM calls)

Building a memory runs one extraction call per chat turn, so we build once and persist each
instance to `bench/memories/<id>.db` (plus a `.json` sidecar). Every later retrieval experiment
runs against these frozen memories with **zero** LLM calls, so any measured difference is purely
the retrieval config.

```bash
export OPENROUTER_API_KEY="sk-..."
# The published numbers used deepseek-v4-flash as the extraction model:
MENISCUS_FREEZE_MODEL=deepseek/deepseek-v4-flash python bench/freeze.py bench/data/longmemeval_s.json 100
```

The run is spend-guarded (`MAX_RUN_COST_USD` in `config.py`) and resumable — rerun to continue.
Frozen memories are gitignored (large); you rebuild them locally.

### 3. Retrieval metrics (free — no LLM)

```bash
python bench/metrics.py       # recall + set-complete (all@k) per question type   -> 0.89 recall
python bench/reduction.py     # injected vs full-history tokens                    -> ~99.1% reduction
```

### 4. End-to-end answer accuracy (small; spend-capped)

A reader model answers from the retrieved facts; a judge model grades it against the gold answer.
Both are set via env vars, and the run is capped at `$2` (`MAX_EVAL_COST_USD`).

```bash
EVAL_READER=openai/gpt-5 python bench/answer_accuracy.py   # strong reader        -> 72%
python bench/answer_accuracy.py                            # default small reader -> ~54%
```

## Which script substantiates which number

| Claim | Script | Notes |
|---|---|---|
| Retrieval recall 0.89 | `bench/metrics.py` | fraction of gold evidence in the returned set |
| Context reduction ~99.1% | `bench/reduction.py` | 1 − injected/full-history tokens |
| Answer accuracy 72% / ~54% | `bench/answer_accuracy.py` | 72% with a GPT-5 reader, ~54% with a small reader; LLM-judged |

`metrics.py` and `reduction.py` call `search_facts` directly and make no LLM calls, so they are
fully deterministic and reproduce exactly on identical frozen memories.
