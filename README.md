<div align="center">

# Meniscus

### Give every AI you use one memory you own.

Local, structured memory for AI agents—in one SQLite file on your machine.

[![PyPI](https://img.shields.io/pypi/v/meniscus?label=PyPI&color=2446c7)](https://pypi.org/project/meniscus/)
[![License: MIT](https://img.shields.io/badge/license-MIT-d88ca6)](LICENSE)

</div>

Meniscus gives AI tools a shared local memory. It turns useful context into compact facts surfaces the minimum amount of memory sufficient for a query. Every connected agent reads and writes the same SQLite file.

```text
AI agents ───┐
Code editors ┼── MCP ── Meniscus ── ~/.meniscus/meniscus.db
Local tools ─┘
```

## Install

Meniscus requires Python 3.11 or newer and an API key for any supported language-model provider. Embeddings run locally; there is no embedding service or embedding key to configure.

```console
uv tool install "meniscus[all]"
men init
```

`men init` does the rest:

1. asks which model provider you want to use;
2. recommends a model or lets you enter another model ID;
3. verifies the key and model before saving them;
4. initializes the local database and embedding model;
5. offers background capture; and
6. connects detected AI tools over MCP.

Nothing is uploaded to a Meniscus server. Configuration and credentials stay under `~/.meniscus/`; the API key is sent only to the provider you select.

## Try it

Add something directly:

```console
men add "Chose SQLite for Fernwind because it needs zero operations and one-file backups."
```

Ask for it later:

```console
men ask "What storage did I choose for Fernwind, and why?"
```

Or ask from a connected agent. It receives two MCP tools:

- `meniscus_recall` retrieves relevant memory.
- `meniscus_log` stores something worth remembering.

## MCP setup

Meniscus works with any client that can run a local `stdio` MCP server, including Claude Code, Codex, Antigravity, Cursor, and Windsurf. `men init` automatically configures clients whose command-line tools it detects; other clients only need the `men-mcp` executable path.

Check an existing connection with:

```console
claude mcp list
codex mcp list
```

If Claude says `MCP server meniscus already exists in local config`, Meniscus is already registered for that project. Do not add it again. If the listed server is healthy, restart Claude Code and continue.

If that existing entry points to an old or missing executable, replace it:

```console
claude mcp remove --scope local meniscus
claude mcp add --scope user meniscus -- "$(command -v men-mcp)"
```

If you skipped connection during initialization, add it manually:

```console
claude mcp add --scope user meniscus -- "$(command -v men-mcp)"
codex mcp add meniscus -- "$(command -v men-mcp)"
```

The absolute executable path matters for GUI applications that do not inherit your shell `PATH`.

<details>
<summary>Antigravity, Cursor, Windsurf, and other MCP clients</summary>

First locate the server:

```console
command -v men-mcp
```

Then use the returned absolute path in the client's MCP settings:

```json
{
  "mcpServers": {
    "meniscus": {
      "command": "/absolute/path/to/men-mcp"
    }
  }
}
```

Open the client's MCP settings and add the object above. The settings screen and filename differ between clients, but the values do not: Meniscus uses local `stdio` transport and needs no MCP URL or MCP token.

</details>

## Commands

### Setup and health

| Command | Purpose |
|---|---|
| `men init` | Complete interactive setup |
| `men config set` | Change the provider, API key, or model |
| `men config show` | Show the active provider and model without exposing the key |
| `men doctor` | Check the database, model, embeddings, and environment |
| `men help` | Show the complete terminal guide |

### Capture and recall

| Command | Purpose |
|---|---|
| `men add "memory"` | Store one note or observation |
| `men ask "question"` | Retrieve relevant facts and answer a question |
| `men ask --json "question"` | Return the retrieved facts without synthesis |
| `men import PATH` | Import and process a text file or directory |
| `men process` | Retry every raw event still pending processing |

### Session capture

| Command | Purpose |
|---|---|
| `men watch --start` | Start capturing in the background |
| `men watch --stop` | Stop capturing in the background |
| `men watch --dry-run` | Show discoverable transcript turns without writing |
| `men watch --catch-up --distill` | Ignore old history, then capture and process new turns |
| `men watch --distill` | Capture and process continuously in the foreground |
| `men watch --once --distill` | Capture and process once, then exit |

Without `--distill`, `men watch` safely stores raw events as pending. Run `men process` later to turn them into searchable facts.

`--start` and `--stop` control [background capture](#background-capture). The other flags run capture in your terminal for as long as the command lasts.

Passive transcript adapters currently recognize Claude Code and Codex session files. Every MCP-compatible client can still recall and log memory through `meniscus_recall` and `meniscus_log`.

### Inspect your memory

| Command | Purpose |
|---|---|
| `men status` | Show event, fact, entity, embedding, and pending counts |
| `men list events` | Browse recent source events |
| `men list events --since 2026-08-01 --source codex` | Filter events by date or source |
| `men show event 42` | Read one complete source event |
| `men tables` | List database tables and row counts |
| `men sql "SELECT text FROM facts LIMIT 10"` | Run a read-only SQL query |

Run `men [COMMAND] --help` for every option accepted by a command.

## Where data lives

| Path | Contents |
|---|---|
| `~/.meniscus/meniscus.db` | Events, facts, entities, embeddings, and ingestion cursors |
| `~/.meniscus/config.toml` | Active provider, model, and runtime settings |
| `~/.meniscus/.env` | Provider API keys, stored with private file permissions |

## Supported model providers

`men init` supports OpenRouter, OpenAI, Anthropic, Google Gemini, Groq, xAI, Hugging Face Inference Providers, Ollama, and custom OpenAI-compatible endpoints. Meniscus uses the same selected model for memory processing and `men ask` synthesis.

If processing becomes unavailable, captured events remain pending instead of being discarded. Fix the provider with `men config set`, verify it with `men doctor`, then resume with `men process`.

## Background capture

Background capture keeps `men watch` running, so Claude Code and Codex sessions are recorded without you starting anything. `men init` offers to install it, and it can be switched on or off at any time:

```console
men watch --start    # start capturing in the background
men watch --stop     # stop capturing
```

Both commands report what actually happened, and capture stays off after `--stop` until you start it again.

If memory seems to have stopped growing, `men status` shows how long ago the last event landed and `men doctor` checks that the database still accepts new events.

## How retrieval works

```text
session or imported text
        ↓
append-only raw event
        ↓
compact facts + local embeddings + entity anchors
        ↓
deterministic fusion of semantic, keyword, and entity retrieval
        ↓
minimum relevant memory returned to the agent
```

Meniscus uses an LLM only to interpret and compact language and, optionally, to synthesize the final `men ask` answer. Storage, provenance, indexing, candidate fusion, and the final retrieval cut are ordinary code.

## Benchmarks

Measured on a 100-instance stratified sample of [LongMemEval-S](https://arxiv.org/abs/2410.10813):

| Metric | Result |
|---|---:|
| Retrieval recall | **0.89** |
| Context reduction | **99.1%** |
| End-to-end answer accuracy | **72%** |

The answer score uses a strong reader. A small local reader scored approximately 54%; the published oracle ceiling using perfect retrieval was approximately 82%. The remaining weakness is multi-session aggregation.

The benchmark inputs, frozen memories, and commands are documented in [`bench/README.md`](bench/README.md).

## Development

```console
git clone https://github.com/magic-bubblez/meniscus.git
cd meniscus
uv sync --all-extras
uv run pytest -q
```

Meniscus is alpha software. Before upgrading, copy `~/.meniscus/meniscus.db` if the memory matters to you.

## License

[MIT](LICENSE)
