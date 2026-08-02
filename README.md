<div align="center">

# Meniscus

### Give your AI a memory you own.

*Local, structured, long-term memory for AI agents — in one file on your machine.*

</div>

---

**Meniscus is a local, long-term memory for your AI tools.** It captures what you do across them, distills it into timestamped facts, and hands any connected AI the relevant ones on demand — from a single SQLite file on your machine that you own and can read.

Two ways in: your agents reach it **automatically over MCP**, or you drive it yourself with the **`men` CLI**. Both talk to the same memory.

## Quickstart

Requires Python 3.11+. Embeddings run locally (no API); distilling facts uses a small LLM via [OpenRouter](https://openrouter.ai). The local embedding model downloads once, then loads fully offline on every run after — no network at startup.

```bash
git clone https://github.com/magic-bubblez/meniscus && cd meniscus
pip install -e ".[all]"                # local embeddings, MCP server, and vector store
export OPENROUTER_API_KEY="sk-..."     # for distillation; embeddings are local
men doctor                             # check what's ready
```

### Connect your AI tools — MCP (the everyday way)

Meniscus runs as a local MCP server (`men-mcp`) and gives every connected agent two tools: **`meniscus_recall`** (read the relevant memory before answering) and **`meniscus_log`** (silently save what's worth remembering). Your tools share one memory and use it without you typing anything.

Add it with one command:

```bash
claude mcp add meniscus -- men-mcp     # Claude Code
codex  mcp add meniscus -- men-mcp     # Codex
```

Or drop it into any MCP client's config (Antigravity, Cursor, …):

```json
{
  "mcpServers": {
    "meniscus": {
      "command": "men-mcp"
    }
  }
}
```

**Claude Code plugin.** A ready-to-install plugin also lives in [`meniscus-plugin/`](meniscus-plugin/) — the same two tools, packaged as a one-click extension.

Restart the tool and ask it something only your own history knows.

### Drive it yourself — CLI (`men`)

```bash
men add   "went with SQLite for Fernwind — zero ops, one file to back up"
men ask   "what did I pick for storage and why?"
men watch --catch-up      # silently capture your Claude Code / Codex sessions into memory
men sql   "SELECT text FROM facts WHERE ..."   # it's just SQLite — look inside
```

## Benchmarks

Measured on [LongMemEval-S](https://arxiv.org/abs/2410.10813), a standard long-term-memory benchmark, on a 100-instance stratified sample.

| | Meniscus |
|---|---|
| Retrieval recall | **0.89** |
| Context reduction | **99.1%** (feeds ~1% of the full history) |
| End-to-end answer accuracy¹ | **72%** |

¹ With a strong reader (GPT-5); a small local reader scores ~54%. The oracle ceiling (perfect retrieval, GPT-4o reader) is ~82%, so retrieval costs ~10 points — most of it on multi-session aggregation, the known weak spot. We'd rather show you the honest number than a cherry-picked one.

**Run them yourself** — every number above is reproducible from frozen memories with no LLM calls: see **[`bench/README.md`](bench/README.md)** for the exact commands.


## What you get

- **It knows you — and compounds.** Every session builds on the last. The more you use it, the more your AI knows about your work, without you repeating yourself.
- **One memory, every AI.** The same memory works across Claude Code, Codex, Antigravity, and any MCP client. Teach it once; recall anywhere.
- **Yours, on your machine.** One local SQLite file. No cloud, no account, no vendor that can revoke it, silently train on it, or shut it down.
- **Nothing is a black box.** Every fact links back to the exact moment it came from, and you can query the whole thing with plain SQL.

## How it works

Meniscus distills what flows through your tools into **atomic facts** (via a small LLM), stores them **append-only** — nothing is ever overwritten or deleted, and every fact is recoverable to its source event — and retrieves them with a **deterministic hybrid search** (semantic vectors + keywords, plus an entity door for lookups by a named person, project, or tool). There is no LLM in the read path, so retrieval is reproducible and inspectable.

Build on it in Python — namespaced under `meniscus`, so nothing generic like `config` or `db` leaks onto your import path:

```python
import meniscus
from meniscus.fact_retrieval import retrieve
```

## Inspect everything

Because it's one SQLite file and retrieval is a fixed formula, nothing is hidden. `men sql` and `men tables` give you the raw facts, their sources, and the entities they link — the same data your AI sees. Memory you can audit is memory you can trust. For more, run `men --help`

