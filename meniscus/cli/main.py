"""Command line interface for Meniscus."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date

import click
from pydantic import BaseModel

from meniscus.config import CUSTOM_PROVIDER
from meniscus.home import load_env
from meniscus.meniscus_types import ExtractionStatus

load_env()


class RetrievalParams(BaseModel):
    """Structured query parameters extracted from a natural-language question."""

    text: str | None = None
    start: str | None = None
    end: str | None = None


class AskAnswer(BaseModel):
    """Final answer for the CLI ask command."""

    answered: bool
    answer: str


def _test_llm(model) -> tuple[bool, str]:
    """Confirm the key authenticates via a free GET /models call — no generation cost."""

    try:
        return model.validate_key()
    except Exception as exc:
        return False, str(exc)


RETRIEVAL_PARAM_PROMPT_TEMPLATE = """\
# Your role
You convert a person's question about their own past into search parameters for
their memory. You do not answer the question -- you only decide what to look for.
Something else takes your parameters, retrieves the matching material, and writes
the answer.

# What you are given
The person's message, and today's date ({today}).

# What you produce
- text: the topic to search for -- the substance of what they're asking about, in
  a few plain keywords, stripped of filler like "what did I" or "tell me about".
  Null only when the question names no topic at all and is purely about a span of
  time.
- start and end: an explicit date range in ISO-8601 (YYYY-MM-DD), when the
  question refers to a period of time. Null when it refers to no particular time.

# How to read the topic
Pull out what the question is actually about and reduce it to the words that would
find it -- concepts, names, activities -- not a full sentence and not the framing
around it. If a question is purely temporal ("what did I do last week", "what have
I been up to"), there is no topic to extract and text is null; the time range
carries the search on its own. For instance, "what did I figure out about auth
last week" has the topic "auth" -- the rest is framing and time.

# How to read the time
Only produce a range when the person actually refers to a time; otherwise leave
start and end null so the search spans their whole history. When they do refer to
a time, convert it into explicit dates using today's date -- the search has no
notion of "last week" on its own, only real dates. Interpret relative references
the way a person naturally would, and when a phrase is loose, err a little wider
rather than narrower: missing the day they meant is worse than including a
neighboring one. As rough guidance: "yesterday" is that single day; "last week" is
the seven days before today; "the past month" is roughly the last thirty days; "in
March" is that month of its most recent occurrence; open-ended words like
"recently" or "lately" are still a short recent range, not null.

# Stay within the question
Do not invent a topic the person didn't raise, and do not impose a time window
they didn't imply. If they named no time, the range is null; if they named no
subject, the topic is null. You transcribe their intent into parameters -- you do
not add to it.

# What you return
A structured result with text, start, and end, each either a value or null.

Question: {question}
"""


SYNTHESIS_PROMPT_TEMPLATE = """\
# Who you are
You are the person's own memory, speaking back to them. They keep a record --
timestamped notes of what they did, learned, decided, and noticed -- and you have
complete, honest access to it. When they ask about their own past, you answer from
that record, in your own natural voice, the way a perceptive friend would if that
friend happened to remember everything accurately. You are not a search engine
returning rows, and not a generic assistant. You are close to this person's
history, and you speak to them plainly, warmly, and truthfully.

# What you are working from
You're given the person's message and a set of facts pulled from their memory.
Each fact is a single self-contained statement, stamped with when it happened, in
time order. The facts are what actually happened. The most relevant come first.

# The one line you never cross
Everything you say about what the person actually did, learned, or decided must
come from these records. You never invent an event, a date, an outcome, or a detail
-- not to fill a gap, not to make an answer neater, not to make it kinder. Your
entire worth is that they can trust every factual thing you tell them about their
past; the moment you make something up, you are worse than useless. If the record
is thin, you say so honestly instead of embroidering it. That is the hard boundary
-- and within it, you have real room.

# How to actually respond
Two things matter here, and both sit on top of the boundary above.

First, read the state of mind behind the message, not only its literal content. A
question is rarely just a request for data; it carries a mood -- doubt, overwhelm,
pride, dread of what's coming -- and the surface question is often a stand-in for
the real one. Identify what the person is actually asking and answer that, in a
register that fits how they arrived. For instance, someone who says they wasted the
past month with something stressful ahead is, underneath, asking whether it's true
that they wasted it -- so that is the question to answer, not the literal "what did
I do."

Second, you may interpret and offer perspective, not only recite. You can reframe,
surface patterns the events support, reflect progress back, and give grounded,
level-headed encouragement or a read on where things stand -- provided all of it
rests on the records and stays honest about what they show. The test: perspective
built on the record is welcome; perspective that needs the record to say more than
it does is fabrication in friendly clothing. A grounded reframe is often the most
useful thing you can offer -- for example, if the person believes they wasted a
month but the events show steady work, showing them that work, with dates, corrects
a distorted memory rather than flattering them. The same honesty runs the other
way: if the record genuinely shows little, be kind but don't invent a better past.

Throughout: when facts carry dates, use them and keep the sequence intact -- a
memory that scrambles the order disorients rather than helps -- and when several
facts bear on the question, weave them into one account instead of listing them
one by one.

# When their memory has changed
A person's understanding of their own past shifts -- what they believed at one
point is often revised later -- and those revisions are frequently the most
valuable thing the record holds. So when the facts disagree across time, never
quietly serve only the latest version or split the difference into a false middle.
Present the shift as a shift, anchored to its dates, so the person can see how their
own thinking moved. For instance: "at first you thought the refresh token was the
cause; by the next day you'd traced it to the TTL mismatch."

# When the record doesn't hold the answer
If the memory simply doesn't contain what they're reaching for, say so -- gently,
briefly, without stretching loosely related material into a false answer or
inventing one. You can still meet them warmly; you just don't manufacture a past to
do it. Set answered to false and let answer be an honest, human sentence that you
don't have anything on it. Don't guess where it might be, and don't list commands or
next steps -- the system handles that.

# Voice
Talk to them directly -- "you", not "the user". Lead with the substance; skip
preambles and don't restate their question. Never expose the machinery -- words like
"fact", "record", "retrieved", "context" don't belong in what they read; they
should feel they're hearing their own memory, not a database report. Be as warm or
as matter-of-fact as the moment calls for, and no longer than it needs to be.

# What you return
A structured result:
- answered: true if the memory held material relevant to what they asked; false if
  it held nothing to go on.
- answer: your response to them when true; an honest, human "I don't have anything
  on that" when false.

Question: {question}

Facts:
{facts}
"""


_BANNER = [
    " __  __   ______   _   _   _____    _____    _____   _    _    _____ ",
    "|  \\/  | |  ____| | \\ | | |_   _|  / ____|  / ____| | |  | |  / ____|",
    "| \\  / | | |__    |  \\| |   | |   | (___   | |      | |  | | | (___  ",
    "| |\\/| | |  __|   | . ` |   | |    \\___ \\  | |      | |  | |  \\___ \\ ",
    "| |  | | | |____  | |\\  |  _| |_   ____) | | |____  | |__| |  ____) |",
    "|_|  |_| |______| |_| \\_| |_____| |_____/   \\_____|  \\____/  |_____/ ",
]

_BANNER_COLORS = [24, 25, 68, 110, 175, 174]

# Written by `men doctor`'s write probe, then rolled back — it must never persist.
_WRITE_PROBE_MARKER = "meniscus-doctor-write-probe"

_BACKGROUND_SERVICE_LABEL = "ai.meniscus.watch"
_WATCH_ARGS = ["-m", "meniscus.cli.main", "watch"]
# Both supervisors run the same command; the seed pass primes the cursor first.
_WATCH_SUPERVISED_ARGS = [*_WATCH_ARGS, "--distill"]
_WATCH_SEED_ARGS = [*_WATCH_ARGS, "--catch-up", "--once"]
_SERVICE_START_GRACE_SECONDS = 1.5
# A busy `men watch` can be mid-poll or mid-LLM-call when it is signalled, so
# stopping is confirmed by polling rather than by one immediate look.
_SERVICE_STOP_TIMEOUT_SECONDS = 10.0
_SERVICE_STOP_POLL_SECONDS = 0.25

_MACOS = "Darwin"
_LINUX = "Linux"
_LAUNCHCTL = "launchctl"
_SYSTEMCTL = "systemctl"
_LOGINCTL = "loginctl"
_LINGER_DISABLED_OUTPUT = "Linger=no"
# The one line of `launchctl print` / `systemctl is-active` output we depend on.
_LAUNCHD_RUNNING_MARKER = "state = running"
_SYSTEMD_ACTIVE_STATE = "active"
_CAPTURE_OUT_LOG = "capture.out.log"
_CAPTURE_ERR_LOG = "capture.err.log"
_NOT_INSTALLED_NOTICE = "  • Background capture is not installed — nothing to stop."

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 60 * _SECONDS_PER_MINUTE
_SECONDS_PER_DAY = 24 * _SECONDS_PER_HOUR

# `men status` reports the last event's age in the largest unit that stays readable.
_AGE_JUST_NOW_SECONDS = 90
_AGE_MINUTES_SECONDS = 90 * _SECONDS_PER_MINUTE
_AGE_HOURS_SECONDS = 2 * _SECONDS_PER_DAY
_STALE_CAPTURE_SECONDS = _SECONDS_PER_DAY


def _print_banner() -> None:
    import sys
    use_color = sys.stdout.isatty()
    for line, code in zip(_BANNER, _BANNER_COLORS):
        if use_color:
            click.echo(f"\033[38;5;{code}m  {line}\033[0m", color=True)
        else:
            click.echo(f"  {line}")
    click.echo()


class _FriendlyGroup(click.Group):
    """Turn known setup errors into a clean one-line message, not a traceback."""

    def invoke(self, ctx: click.Context):
        from meniscus.exceptions import MeniscusError

        try:
            return super().invoke(ctx)
        except MeniscusError as exc:
            click.echo(f"Error: {exc}", err=True)
            click.echo("Run `men doctor` to see what's missing.", err=True)
            raise SystemExit(1)


@click.group(cls=_FriendlyGroup, invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Meniscus: local structured memory for AI agents."""
    if ctx.invoked_subcommand is None:
        from meniscus.home import ENV_FILENAME, MENISCUS_HOME

        env_exists = (MENISCUS_HOME / ENV_FILENAME).exists()
        if not env_exists:
            click.echo("")
            _print_banner()
            click.echo("  Get started:")
            click.echo("    men init      set up your key, embeddings, and AI tool connections")
            click.echo("    men doctor    check what's ready")
            click.echo("    men --help    see all commands")
            click.echo("")
        else:
            click.echo(ctx.get_help())


@cli.command()
def doctor() -> None:
    """Check the environment and report what (if anything) is missing."""

    from meniscus import config
    from meniscus.db import get_connection, get_db_path, init_db
    from meniscus.exceptions import MeniscusError, ModelUnavailableError

    def line(name: str, ok: bool | None, detail: str) -> None:
        mark = "✓" if ok else ("•" if ok is None else "✗")
        click.echo(f"  {mark} {name:<14} {detail}")

    click.echo("Meniscus setup check\n")

    try:
        db_path = get_db_path()
        line("database", True, str(db_path))
    except Exception as exc:  # pragma: no cover - defensive
        line("database", False, f"could not resolve path: {exc}")

    provider = config.EMBEDDING_PROVIDER
    if provider == config.EMBEDDING_DISABLED:
        line("sqlite-vec", None, "not required (EMBEDDING_PROVIDER=\"none\")")
    else:
        try:
            import sqlite3 as _sqlite3

            import sqlite_vec

            probe = _sqlite3.connect(":memory:")
            try:
                probe.enable_load_extension(True)
                sqlite_vec.load(probe)
            finally:
                probe.close()
            line("sqlite-vec", True, "installed and loadable")
        except ImportError:
            line(
                "sqlite-vec",
                False,
                'not installed  →  uv tool install "meniscus[all]"  '
                f'(or set EMBEDDING_PROVIDER="{config.EMBEDDING_DISABLED}")',
            )
        except Exception as exc:
            line("sqlite-vec", False, f"installed but cannot load: {exc}")

    line(
        "embeddings",
        None,
        f'provider="{provider}", {config.EMBEDDING_DIMENSIONS} dims',
    )
    from meniscus.providers import get_model

    llm = config.DEFAULT_MODEL_PROVIDER
    llm_ok = False
    try:
        model = get_model()
        ok, detail = _test_llm(model)
        llm_ok = ok
        line("LLM provider", ok, f'provider="{llm}" — {detail}')
    except ModelUnavailableError as exc:
        line(
            "LLM provider",
            False,
            f"{exc}  (required to process new memories)",
        )
    except ImportError:
        line(
            "LLM provider",
            False,
            "provider not resolvable — check [model] provider in ~/.meniscus/config.toml "
            "(required to process new memories)",
        )
    except Exception as exc:  # pragma: no cover - defensive
        line("LLM provider", False, str(exc))

    conn = get_connection()
    try:
        init_db(conn)
        line("startup", True, "database initializes cleanly")
        write_ok, write_name, write_detail = _probe_write_path(conn)
        line(write_name, write_ok, write_detail)
        startup_ok = write_ok
    except MeniscusError as exc:
        line("startup", False, str(exc))
        startup_ok = False
    finally:
        conn.close()

    click.echo("")
    if startup_ok and llm_ok:
        click.echo(
            "Ready. Meniscus can capture, process, and retrieve memory."
        )
    elif startup_ok:
        click.echo(
            "Storage is ready, but new memories will remain pending until the "
            "provider issue above is fixed."
        )
    else:
        click.echo("Fix the ✗ items above, then run `men doctor` again.")


_PROVIDER_MENU = [
    ("openrouter", "OpenRouter — one key, routes to hundreds of models"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("gemini", "Google Gemini"),
    ("groq", "Groq — fast inference on open-weight models"),
    ("xai", "xAI"),
    ("huggingface", "Hugging Face — open-weight models via Inference Providers"),
    ("ollama", "Ollama — free, runs models locally (no key needed)"),
    (CUSTOM_PROVIDER, "Custom — any other OpenAI-compatible endpoint you specify"),
]


def _write_model_config(
    cfg_path,
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> None:
    from meniscus.home import write_private_text

    lines = [f"provider = {json.dumps(provider)}"]
    if base_url:
        lines.append(f"base_url = {json.dumps(base_url)}")
    if api_key_env:
        lines.append(f"api_key_env = {json.dumps(api_key_env)}")
    lines.append(f"model = {json.dumps(model)}")
    block = "[model]\n" + "\n".join(lines) + "\n"

    existing = cfg_path.read_text() if cfg_path.exists() else ""
    kept: list[str] = []
    skipping = False
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped == "[model]":
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped != "[model]":
            skipping = False
        if not skipping:
            kept.append(line)
    rest = "\n".join(kept).strip()

    content = (block + ("\n\n" + rest if rest else "")).strip() + "\n"
    write_private_text(cfg_path, content)


def _prompt_provider() -> str:
    click.echo("  Choose your LLM provider")
    click.echo("  Meniscus uses the same model to process memories and answer `men ask`.")
    click.echo("")
    for i, (_, description) in enumerate(_PROVIDER_MENU, start=1):
        click.echo(f"    [{i}] {description}")
    click.echo("")

    choice = click.prompt(
        "  Choice",
        type=click.Choice([str(i) for i in range(1, len(_PROVIDER_MENU) + 1)]),
        show_choices=False,
    )
    return _PROVIDER_MENU[int(choice) - 1][0]


def _prompt_provider_connection(provider: str) -> tuple[str | None, str | None]:
    import re
    from urllib.parse import urlparse

    from meniscus import config

    click.echo("")
    if provider == CUSTOM_PROVIDER:
        base_url = click.prompt(
            "  Base URL (OpenAI-compatible, e.g. https://vendor.example/v1)",
            default=config.CUSTOM_BASE_URL,
        ).strip()
        api_key_env = click.prompt(
            "  Name for the key in .env (e.g. MY_VENDOR_API_KEY)",
            default=config.CUSTOM_API_KEY_ENV,
        ).strip()
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise click.UsageError("Base URL must be an absolute HTTP(S) URL.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise click.UsageError("API key variable must be a valid environment name.")
        return base_url, api_key_env

    preset = config.PROVIDER_PRESETS[provider]
    if preset.api_key_env is None:
        click.echo("  No API key needed — Meniscus will call your local Ollama server.")
    return None, preset.api_key_env


def _prompt_api_key(api_key_env: str | None) -> tuple[str | None, bool]:
    import os

    if api_key_env is None:
        return None, False

    existing_key = os.environ.get(api_key_env, "")
    if existing_key:
        click.echo(f"  {api_key_env} already loaded (…{existing_key[-4:]})")
        if not click.confirm("  Update it?", default=False):
            return existing_key, False

    key = click.prompt(f"  Paste your {api_key_env}", hide_input=True).strip()
    if not key:
        raise click.UsageError("API key cannot be empty.")
    return key, True


def _persist_api_key(api_key_env: str | None, key: str | None, env_path) -> None:
    import os

    from meniscus.home import write_private_text

    if api_key_env is None or key is None:
        return

    retained_lines = []
    if env_path.exists():
        retained_lines = [
            line
            for line in env_path.read_text().splitlines()
            if not line.startswith(f"{api_key_env}=")
        ]
    retained_lines.append(f"{api_key_env}={json.dumps(key)}")
    write_private_text(env_path, "\n".join(retained_lines) + "\n")
    os.environ[api_key_env] = key
    click.echo(f"  ✓ {api_key_env} saved to {env_path}")


def _prompt_model(provider: str) -> str:
    from meniscus import config

    click.echo("")
    if provider != CUSTOM_PROVIDER:
        recommended = config.PROVIDER_DEFAULT_MODELS[provider]
        click.echo(f"  Recommended model: {recommended}")
        if click.confirm("  Use this model?", default=True):
            return recommended

    model = click.prompt("  Model ID supported by this provider").strip()
    if not model:
        raise click.UsageError("Model ID cannot be empty.")
    return model


def _validate_provider_selection(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> tuple[bool, str]:
    from meniscus.providers import build_model

    return _test_llm(build_model(provider, model, api_key, base_url))


def _configure_model_interactively() -> None:
    """Ask for one BYOK provider, credential, and model; persist the choice."""
    import importlib

    from meniscus import config
    from meniscus.home import CONFIG_FILENAME, ENV_FILENAME, MENISCUS_HOME, ensure_home

    ensure_home()
    provider = _prompt_provider()
    base_url, api_key_env = _prompt_provider_connection(provider)
    api_key, should_persist_key = _prompt_api_key(api_key_env)
    model = _prompt_model(provider)

    click.echo("")
    click.echo("  Checking API access and model availability…")
    ok, detail = _validate_provider_selection(provider, model, api_key, base_url)
    if not ok:
        click.echo(f"  ✗ {detail}")
        raise click.ClickException("Provider setup failed. Nothing was saved.")
    click.echo(f"  ✓ {detail}")

    if should_persist_key:
        _persist_api_key(api_key_env, api_key, MENISCUS_HOME / ENV_FILENAME)

    _write_model_config(
        MENISCUS_HOME / CONFIG_FILENAME,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env if provider == CUSTOM_PROVIDER else None,
    )
    importlib.reload(config)
    click.echo(f"  ✓ Provider and model saved to {MENISCUS_HOME / CONFIG_FILENAME}")


@cli.group("config", invoke_without_command=True)
@click.pass_context
def config_command(ctx: click.Context) -> None:
    """Configure or inspect the active provider, key, and model."""
    if ctx.invoked_subcommand is None:
        _configure_model_interactively()


@config_command.command("set")
def config_set() -> None:
    """Interactively change the active provider, key, and model."""
    _configure_model_interactively()


@config_command.command("show")
def config_show() -> None:
    """Show active model configuration without exposing the API key."""
    import os

    from meniscus import config

    provider = config.DEFAULT_MODEL_PROVIDER
    if provider == CUSTOM_PROVIDER:
        key_env = config.CUSTOM_API_KEY_ENV
        base_url = config.CUSTOM_BASE_URL
    else:
        preset = config.PROVIDER_PRESETS.get(provider or "")
        key_env = preset.api_key_env if preset else None
        base_url = preset.base_url if preset else None

    click.echo(f"provider: {provider or 'not configured'}")
    click.echo(f"model:    {config.MODEL or 'not configured'}")
    if base_url:
        click.echo(f"base URL: {base_url}")
    if key_env:
        value = os.environ.get(key_env, "")
        state = f"set (…{value[-4:]})" if value else "not set"
        click.echo(f"API key:  {key_env} — {state}")
    else:
        click.echo("API key:  not required")


@cli.command()
def init() -> None:
    """Set up Meniscus: home dir, LLM provider, health check, capture, and AI tool wiring."""
    import shutil

    from meniscus.home import MENISCUS_HOME, ensure_home

    click.echo("")
    _print_banner()

    ensure_home()
    click.echo(f"  Memory home: {MENISCUS_HOME}")
    click.echo("")

    click.echo("  Step 1 of 4 — Connect your language model")
    click.echo("  Meniscus uses your chosen model to turn useful parts of your AI")
    click.echo("  sessions into searchable memory. You bring the provider and API key.")
    click.echo("")
    _configure_model_interactively()

    click.echo("")

    click.echo("  Step 2 of 4 — Checking your setup")
    click.echo("")
    _init_doctor()
    click.echo("")

    click.echo("  Step 3 of 4 — Auto-capture your AI sessions")
    click.echo("")
    click.echo("  Meniscus can silently watch your Claude Code / Codex conversations")
    click.echo("  and save what's worth remembering — automatically, in the background.")
    click.echo("")
    click.echo("    [1] Background service  — always on, starts at login  (recommended)")
    click.echo("    [2] Manual              — run `men watch` yourself when you want to capture")
    click.echo("    [3] Skip                — log via `men add` or meniscus_log in your AI tool")
    click.echo("")

    choice = click.prompt(
        "  Choice",
        type=click.Choice(["1", "2", "3"]),
        default="1",
        show_choices=False,
    )

    if choice == "1":
        _install_background_service()
    elif choice == "2":
        click.echo("  → Run `men watch --catch-up --distill` to capture and process new sessions.")
    else:
        click.echo("  → Log manually: `men add \"note\"` or let your AI tools call meniscus_log.")

    click.echo("")

    click.echo("  Step 4 of 4 — Connect your AI tools")
    click.echo("")

    men_mcp_path = shutil.which("men-mcp")
    if not men_mcp_path:
        click.echo("  ✗ men-mcp not found on PATH — skipping.")
        click.echo("    Ensure Meniscus is installed and ~/.local/bin is on your PATH.")
    else:
        _wire_mcp_tools(men_mcp_path)

    click.echo("")
    click.echo("  All done. Your memory is ready.")
    click.echo("")
    click.echo('  Try: men ask "what have I been working on?"')
    click.echo("")


@cli.command("help")
def help_command() -> None:
    """Show Meniscus usage guide with commands and configuration reference."""
    from meniscus.home import CONFIG_FILENAME, ENV_FILENAME, MENISCUS_HOME

    env_path = MENISCUS_HOME / ENV_FILENAME
    cfg_path = MENISCUS_HOME / CONFIG_FILENAME

    click.echo("")
    _print_banner()

    click.echo("  Meniscus — local structured memory for AI agents")
    click.echo("")

    click.echo("  GETTING STARTED")
    click.echo("    men init            interactive setup: API key, health check, AI tool wiring")
    click.echo("    men config set      change provider, API key, or model interactively")
    click.echo("    men config show     show the active provider and model safely")
    click.echo("    men doctor          check what's installed and configured")
    click.echo("")

    click.echo("  DAILY USE")
    click.echo('    men add "note"      save a note or observation directly')
    click.echo('    men ask "question"  ask your memory a question in plain language')
    click.echo("    men watch           capture Claude Code / Codex sessions in real-time")
    click.echo("    men import <path>   bulk-import a file or directory into memory")
    click.echo("    men process         process any pending events (extract facts)")
    click.echo("")

    click.echo("  INSPECT")
    click.echo("    men status          event / fact / entity counts")
    click.echo("    men tables          all DB tables with row counts")
    click.echo('    men sql "SELECT ..."  run a read-only SQL query against the DB')
    click.echo("    men list events     browse stored events")
    click.echo("    men show event N    full detail for event N")
    click.echo("")

    click.echo("  CONFIGURATION")
    click.echo("")
    click.echo("    `men init` and `men config set` show one recommended model for each")
    click.echo("    provider. Accept it or enter another model ID from that provider.")
    click.echo("")
    click.echo(f"    API keys live in  {env_path}")
    click.echo("    Store as many keys as you like; only the one matching the active")
    click.echo("    provider (below) is ever read.")
    click.echo("")
    click.echo(f"    The active provider + model live in  {cfg_path}")
    click.echo("    You normally never need to edit this file; use `men config set`.")
    click.echo("")
    click.echo("      [model]")
    click.echo('      provider = "openrouter"   # see the provider list below')
    click.echo('      model = "..."              # model ID from your provider')
    click.echo("")
    click.echo("      # only used when provider = \"custom\":")
    click.echo('      # base_url    = "https://vendor.example/v1"')
    click.echo('      # api_key_env = "MY_VENDOR_API_KEY"   # name of the key in .env')
    click.echo("")
    click.echo("      [spend]")
    click.echo("      max_run_cost_usd = 20.0   # hard ceiling per batch run")
    click.echo("")
    click.echo("      [retrieval]")
    click.echo("      default_limit = 30    # max facts returned per recall")
    click.echo("      neighbour_max = 20    # neighbour facts gathered around a match")
    click.echo("")
    click.echo("      [ingestion]")
    click.echo("      concurrency = 20      # parallel LLM calls during bulk import")
    click.echo("")
    click.echo("    Providers (key env var — what it is):")
    click.echo("      openrouter    OPENROUTER_API_KEY  — one key, hundreds of models")
    click.echo("      openai        OPENAI_API_KEY      — direct OpenAI access")
    click.echo("      anthropic     ANTHROPIC_API_KEY   — direct Claude access (native API)")
    click.echo("      gemini        GEMINI_API_KEY      — direct Gemini access")
    click.echo("      groq          GROQ_API_KEY        — fast inference, open-weight models")
    click.echo("      xai           XAI_API_KEY         — direct Grok access (not Groq — different company)")
    click.echo("      huggingface   HF_TOKEN             — open-weight models via Inference Providers")
    click.echo("      ollama        no key               — free, runs models locally")
    click.echo("      custom        base_url + api_key_env you set — any other OpenAI-compatible endpoint")
    click.echo("")
    click.echo("    Embeddings run locally and are managed by Meniscus; there is no")
    click.echo("    embedding provider or model to choose during setup.")
    click.echo("")


def _init_doctor() -> None:
    """Compact health check for men init."""
    from meniscus import config
    from meniscus.db import get_connection, get_db_path, init_db
    from meniscus.exceptions import MeniscusError, ModelUnavailableError

    def line(ok: bool | None, name: str, detail: str) -> None:
        mark = "✓" if ok else ("•" if ok is None else "✗")
        click.echo(f"  {mark} {name:<14} {detail}")

    try:
        line(True, "database", str(get_db_path()))
    except Exception as exc:
        line(False, "database", str(exc))

    try:
        import sqlite3 as _s3
        import sqlite_vec
        p = _s3.connect(":memory:")
        p.enable_load_extension(True)
        sqlite_vec.load(p)
        p.close()
        line(True, "sqlite-vec", "ready")
    except ImportError:
        line(False, "sqlite-vec", 'missing — run: uv tool install "meniscus[all]"')
    except Exception as exc:
        line(False, "sqlite-vec", str(exc))

    try:
        from meniscus.providers import get_model
        model = get_model()
        ok, detail = _test_llm(model)
        line(ok, "LLM provider", detail)
    except ModelUnavailableError as exc:
        line(False, "LLM provider", str(exc))
    except Exception as exc:
        line(False, "LLM provider", str(exc))

    conn = get_connection()
    try:
        init_db(conn)
        line(True, "startup", "database initializes cleanly")
        line(*_probe_write_path(conn))
    except MeniscusError as exc:
        line(False, "startup", str(exc))
    finally:
        conn.close()


def _probe_write_path(conn) -> tuple[bool, str, str]:
    """Insert a throwaway event and roll it back.

    Capture is deliberately silent, so a broken write path produces no user-visible
    signal — an obsolete trigger or a lost index can reject every insert while the
    schema still opens and reads cleanly. Only a real insert exercises that path.
    """

    import sqlite3
    from datetime import datetime, timezone

    try:
        conn.execute(
            "INSERT INTO events (source, content, timestamp, content_hash) "
            "VALUES (?, ?, ?, ?)",
            (
                _WRITE_PROBE_MARKER,
                _WRITE_PROBE_MARKER,
                datetime.now(timezone.utc).isoformat(),
                _WRITE_PROBE_MARKER,
            ),
        )
        return True, "write path", "events accept new rows"
    except sqlite3.Error as exc:
        return False, "write path", f"cannot record new events: {exc}"
    finally:
        # The probe must never survive: roll back whether it succeeded or failed.
        conn.rollback()


def _install_background_service() -> None:
    """Install a per-user supervisor that keeps `men watch` running continuously."""

    import platform
    import sys

    system = platform.system()
    if system == _MACOS:
        installer = _install_launchagent
    elif system == _LINUX:
        installer = _install_systemd_unit
    else:
        click.echo(f"  • No background service for {system}.")
        click.echo("    Run `men watch --catch-up` manually to start capturing.")
        return

    # Keep the environment's interpreter path, including a venv/uv-tool
    # symlink. Resolving it can turn ``.../tools/meniscus/bin/python`` into the
    # base Homebrew interpreter, which no longer has Meniscus installed.
    python_path = sys.executable
    if not _seed_capture(python_path):
        return
    installer(python_path)


def _uninstall_background_service() -> None:
    """Stop background capture and remove the supervisor, so it cannot come back.

    Stopping the process alone is not enough: both supervisors are configured to
    restart it and to start it again at login. Turning capture off means removing
    the definition, not just the running instance.
    """

    import platform

    system = platform.system()
    if system == _MACOS:
        _uninstall_launchagent()
    elif system == _LINUX:
        _uninstall_systemd_unit()
    else:
        click.echo(f"  • No background service is installed on {system}.")


def _launchd_service() -> str:
    """The launchd service target: one spelling, used by install, stop, and status."""

    import os

    return f"gui/{os.getuid()}/{_BACKGROUND_SERVICE_LABEL}"


def _launchagent_plist_path():
    from pathlib import Path

    return (
        Path.home() / "Library" / "LaunchAgents" / f"{_BACKGROUND_SERVICE_LABEL}.plist"
    )


def _systemd_unit_name() -> str:
    return f"{_BACKGROUND_SERVICE_LABEL}.service"


def _systemd_unit_path():
    from pathlib import Path

    return Path.home() / ".config" / "systemd" / "user" / _systemd_unit_name()


def _uninstall_launchagent() -> None:
    import subprocess

    service = _launchd_service()
    plist_path = _launchagent_plist_path()

    if not plist_path.exists():
        click.echo(_NOT_INSTALLED_NOTICE)
        return

    subprocess.run([_LAUNCHCTL, "bootout", service], capture_output=True)
    plist_path.unlink(missing_ok=True)
    _report_stopped(
        _still_running_after_grace(_launchagent_running),
        f"{_LAUNCHCTL} bootout {service}",
    )


def _uninstall_systemd_unit() -> None:
    import shutil
    import subprocess

    unit_name = _systemd_unit_name()
    unit_path = _systemd_unit_path()

    if not unit_path.exists():
        click.echo(_NOT_INSTALLED_NOTICE)
        return

    has_systemctl = shutil.which(_SYSTEMCTL) is not None
    if has_systemctl:
        subprocess.run(
            [_SYSTEMCTL, "--user", "disable", "--now", unit_name], capture_output=True
        )
    unit_path.unlink(missing_ok=True)
    if has_systemctl:
        subprocess.run([_SYSTEMCTL, "--user", "daemon-reload"], capture_output=True)
    _report_stopped(
        _still_running_after_grace(lambda: _systemd_unit_active(unit_name)),
        f"{_SYSTEMCTL} --user disable --now {unit_name}",
    )


def _still_running_after_grace(is_running) -> bool:
    """Poll until the service is gone, and only then call a stop failed.

    Stopping is asynchronous: the supervisor returns as soon as it has signalled
    the job, while the job itself may take seconds to finish what it was doing
    and exit. Checking once, immediately, reports a service that is on its way
    out as one that refused to stop.
    """

    import time

    deadline = time.monotonic() + _SERVICE_STOP_TIMEOUT_SECONDS
    while True:
        if not is_running():
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(_SERVICE_STOP_POLL_SECONDS)


def _report_stopped(still_running: bool, manual: str) -> None:
    if still_running:
        click.echo("  ✗ Background capture is still running.")
        click.echo(f"    Stop it manually with: {manual}")
        return
    click.echo("  ✓ Background capture stopped and removed")
    click.echo("    Turn it back on with `men watch --start`.")


def _launchagent_running() -> bool:
    import subprocess

    printed = subprocess.run(
        [_LAUNCHCTL, "print", _launchd_service()], capture_output=True, text=True
    )
    return _LAUNCHD_RUNNING_MARKER in printed.stdout


def _systemd_unit_active(unit_name: str) -> bool:
    import shutil
    import subprocess

    if shutil.which(_SYSTEMCTL) is None:
        return False
    active = subprocess.run(
        [_SYSTEMCTL, "--user", "is-active", unit_name], capture_output=True, text=True
    )
    return active.stdout.strip() == _SYSTEMD_ACTIVE_STATE


def _seed_capture(python_path: str) -> bool:
    """Capture existing history once, so the supervised run starts from a known cursor."""

    import subprocess

    seed = subprocess.run(
        [python_path, *_WATCH_SEED_ARGS],
        capture_output=True,
        text=True,
    )
    if seed.returncode != 0:
        click.echo(f"  ✗ Could not initialize capture: {(seed.stderr or seed.stdout).strip()}")
        return False
    return True


def _report_service(started: bool, detail: str, logs: str, manual: str) -> None:
    """Report what the supervisor actually did — never assume a clean exit means running."""

    if started:
        click.echo("  ✓ Background service installed and started")
        click.echo(f"    Logs → {logs}")
    else:
        click.echo(f"  ✗ Service did not stay running: {detail}")
        click.echo(f"    Logs → {logs}")
        click.echo(f"    Retry with: {manual}")


def _install_launchagent(python_path: str) -> None:
    """Write and bootstrap a macOS LaunchAgent for continuous capture and processing."""

    import os
    import plistlib
    import subprocess
    import time

    from meniscus.home import MENISCUS_HOME

    domain = f"gui/{os.getuid()}"
    service = _launchd_service()
    plist_path = _launchagent_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    err_log = MENISCUS_HOME / _CAPTURE_ERR_LOG

    plist = {
        "Label": _BACKGROUND_SERVICE_LABEL,
        "ProgramArguments": [python_path, *_WATCH_SUPERVISED_ARGS],
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": {"MENISCUS_HOME": str(MENISCUS_HOME)},
        "StandardOutPath": str(MENISCUS_HOME / _CAPTURE_OUT_LOG),
        "StandardErrorPath": str(err_log),
    }

    if plist_path.exists():
        subprocess.run([_LAUNCHCTL, "bootout", service], capture_output=True)

    with plist_path.open("wb") as plist_file:
        plistlib.dump(plist, plist_file)

    result = subprocess.run(
        [_LAUNCHCTL, "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _report_service(
            False,
            result.stderr.strip() or f"{_LAUNCHCTL} rejected the job",
            str(err_log),
            f"{_LAUNCHCTL} bootstrap {domain} {plist_path}",
        )
        return

    subprocess.run([_LAUNCHCTL, "kickstart", "-k", service], capture_output=True)
    # `bootstrap` only means launchd accepted the job. A job whose interpreter it
    # cannot reach — a venv under a TCC-protected folder, say — exits immediately
    # and would otherwise be reported as a success.
    time.sleep(_SERVICE_START_GRACE_SECONDS)
    _report_service(
        _launchagent_running(),
        _last_log_line(err_log) or "the process exited right after launch",
        str(err_log),
        f"{_LAUNCHCTL} kickstart -k {service}",
    )


# systemd captures stdout/stderr into the journal, so the unit declares no log paths.
_SYSTEMD_UNIT_TEMPLATE = """\
[Unit]
Description=Meniscus continuous capture

[Service]
ExecStart={exec_start}
Environment=MENISCUS_HOME={meniscus_home}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""


def _install_systemd_unit(python_path: str) -> None:
    """Write and enable a systemd --user unit for continuous capture and processing."""

    import shutil
    import subprocess
    import time

    from meniscus.home import MENISCUS_HOME

    unit_name = _systemd_unit_name()
    journal = f"journalctl --user -u {unit_name}"

    if shutil.which(_SYSTEMCTL) is None:
        click.echo("  • No systemd on this machine, so there is nothing to supervise the capture.")
        click.echo("    Run `men watch --catch-up` manually, or supervise it with your init system.")
        return

    unit_path = _systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(
        _SYSTEMD_UNIT_TEMPLATE.format(
            exec_start=" ".join([python_path, *_WATCH_SUPERVISED_ARGS]),
            meniscus_home=MENISCUS_HOME,
        )
    )

    subprocess.run([_SYSTEMCTL, "--user", "daemon-reload"], capture_output=True)
    result = subprocess.run(
        [_SYSTEMCTL, "--user", "enable", "--now", unit_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _report_service(
            False,
            result.stderr.strip() or f"{_SYSTEMCTL} rejected the unit",
            journal,
            f"{_SYSTEMCTL} --user enable --now {unit_name}",
        )
        return

    # `Restart=always` means a unit that crash-loops still reports success from
    # `enable --now`; only a second look tells us it stayed up.
    time.sleep(_SERVICE_START_GRACE_SECONDS)
    active = _systemd_unit_active(unit_name)
    _report_service(
        active,
        "the unit is not active",
        journal,
        f"{_SYSTEMCTL} --user restart {unit_name}",
    )
    if active:
        _warn_if_linger_disabled()


def _warn_if_linger_disabled() -> None:
    """Say so when systemd will stop capture at logout, and only then.

    A systemd user service is torn down when the user's last session ends unless
    lingering is enabled. Enabling it needs an authorization Meniscus cannot ask
    for on the user's behalf, so the next best thing is to say so once, here,
    where it is actionable — and to stay silent when it does not apply.
    """

    import getpass
    import shutil
    import subprocess

    if shutil.which(_LOGINCTL) is None:
        return

    user = getpass.getuser()
    result = subprocess.run(
        [_LOGINCTL, "show-user", user, "--property=Linger"],
        capture_output=True,
        text=True,
    )
    # Only speak up on a definite "no" — an unreadable answer is not a problem.
    if result.returncode != 0 or result.stdout.strip() != _LINGER_DISABLED_OUTPUT:
        return

    click.echo("    Capture will pause when you log out. To keep it running:")
    click.echo(f"      {_LOGINCTL} enable-linger {user}")


def _last_log_line(path) -> str:
    """Return the last non-empty line of a log, for reporting why a service died."""

    try:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def _wire_mcp_tools(men_mcp_path: str) -> None:
    """Detect installed AI tools and offer to wire MCP into each."""
    import shutil
    import subprocess

    tools = []
    if shutil.which("claude"):
        tools.append(("Claude Code", "claude"))
    if shutil.which("codex"):
        tools.append(("Codex", "codex"))

    if not tools:
        click.echo("  No supported AI tools detected on PATH (Claude Code, Codex).")
        click.echo("  Add Meniscus to any MCP client manually:")
        click.echo(f'    command: "{men_mcp_path}"')
        return

    for tool_name, bin_name in tools:
        click.echo(f"  Detected: {tool_name}")
        if click.confirm(f"  Wire Meniscus into {tool_name}?", default=True):
            scope = ["--scope", "user"] if bin_name == "claude" else []
            result = subprocess.run(
                [bin_name, "mcp", "add", *scope, "meniscus", "--", men_mcp_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                click.echo(f"  ✓ Added to {tool_name} — restart it to activate")
            else:
                err = (result.stderr + result.stdout).strip().lower()
                if "already" in err or "exists" in err:
                    click.echo(f"  • Already wired into {tool_name}")
                else:
                    click.echo(f"  ✗ Failed: {(result.stderr or result.stdout).strip()}")
                    manual = " ".join(
                        [bin_name, "mcp", "add", *scope, "meniscus", "--", men_mcp_path]
                    )
                    click.echo(f"    Run manually: {manual}")
        else:
            click.echo(f"  Skipped {tool_name}.")


def _get_resources():
    """Open the DB and best-effort model resources for the write path.

    A missing/unavailable model does NOT fail capture: intake is deterministic,
    so events are saved as pending and processed later. Genuine configuration
    errors (dimension mismatch, unknown provider) still surface via init_db /
    the provider factories.
    """

    from meniscus.db import get_connection, init_db
    from meniscus.providers import get_embedding_if_available, get_model_if_available
    from meniscus.startup import announce_embedding_state

    conn = get_connection()
    try:
        init_db(conn)
        announce_embedding_state()
        model = get_model_if_available()
        embedding_model = get_embedding_if_available()
    except Exception:
        conn.close()
        raise
    return conn, model, embedding_model


def _get_retrieval_resources():
    """Open the DB and best-effort model resources for retrieval."""

    from meniscus import config
    from meniscus.db import get_connection, init_db
    from meniscus.providers import get_embedding_if_available, get_model_if_available

    conn = get_connection()
    try:
        init_db(conn)
        # No "embeddings disabled" notice on read/query commands — it would
        # pollute machine-readable output (e.g. `ask --json`). It stays on the
        # ingest commands, where the heads-up actually matters.

        model = get_model_if_available()
        embedding_model = get_embedding_if_available()
        embedding_ready = (
            config.EMBEDDING_PROVIDER == config.EMBEDDING_DISABLED
            or embedding_model is not None
        )
    except Exception:
        conn.close()
        raise

    return conn, model, embedding_model, embedding_ready


@cli.command()
@click.argument("message", required=False)
@click.option("--source", default="cli", help="Source identifier for this event.")
def add(message: str | None, source: str) -> None:
    if message is None or message == "-":
        message = sys.stdin.read()
    if not message.strip():
        click.echo(
            "No message. Pass text as an argument, or pipe/redirect it via stdin "
            "(e.g. `pbpaste | men add`).",
            err=True,
        )
        raise SystemExit(2)

    conn, model, embedding_model = _get_resources()
    try:
        from meniscus.pipeline import ingest_and_process

        event_ids = ingest_and_process(conn, message, source, model, embedding_model)
        if not event_ids:
            click.echo("Duplicate content — no new event created.")
            return
        for event_id in event_ids:
            fact_count = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE event_id = ?", (event_id,)
            ).fetchone()[0]
            if fact_count:
                click.echo(f"Saved event {event_id} — {fact_count} fact(s) extracted.")
            else:
                click.echo(
                    f"Saved event {event_id} — pending "
                    "(run `men process` once a model is available)."
                )
    finally:
        conn.close()


@cli.command("import")
@click.argument("path")
def import_path(path: str) -> None:
    """Import a file or directory and batch process created events."""

    conn, model, embedding_model = _get_resources()
    try:
        from meniscus.pipeline import import_and_process

        def _progress(done: int, total: int) -> None:
            click.echo(f"\r  processing {done}/{total}…", nl=False)

        event_ids = import_and_process(
            conn, path, model, embedding_model, on_progress=_progress
        )
        if model is not None and event_ids:
            click.echo("")  # end the progress line
        if not event_ids:
            click.echo(
                "No new events — the path had no supported (.txt/.md) files, or they "
                "were all duplicates already in memory."
            )
            return
        placeholders = ",".join("?" for _ in event_ids)
        pending = conn.execute(
            f"SELECT COUNT(*) FROM events "
            f"WHERE id IN ({placeholders}) AND extraction_status = ?",
            [*event_ids, ExtractionStatus.PENDING],
        ).fetchone()[0]
        if pending:
            click.echo(
                f"Imported {len(event_ids)} event(s); {pending} still pending. "
                "Run `men process` once a model is configured."
            )
        else:
            click.echo(f"Imported and processed {len(event_ids)} event(s).")
    finally:
        conn.close()


@cli.command()
def process() -> None:
    """Process all pending events."""

    conn, model, embedding_model = _get_resources()
    try:
        from meniscus.pipeline import process_pending_events

        if model is None:
            click.echo(
                "No model configured, so nothing was processed. Run `men config set`, "
                "then run `men process` again."
            )
            return

        pending = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = ?",
            (ExtractionStatus.PENDING,),
        ).fetchone()[0]
        if pending == 0:
            click.echo("Nothing to process — no pending events. You're all caught up.")
            return

        click.echo(f"Processing {pending} pending event(s) (parallel)…")

        def _progress(done: int, total: int) -> None:
            click.echo(f"\r  processed {done}/{total}…", nl=False)

        processed = process_pending_events(
            conn, model, embedding_model, on_progress=_progress
        )
        click.echo("")  # end the progress line
        remaining = pending - len(processed)
        if remaining > 0:
            click.echo(
                f"Stopped after {len(processed)}: the model became unavailable, so "
                f"{remaining} event(s) remain pending. Run `men process` again to resume."
            )
        else:
            click.echo(f"Done — processed {len(processed)} event(s).")
    finally:
        conn.close()


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be captured; write nothing.")
@click.option("--catch-up", is_flag=True, help="Skip existing history; capture only from now on.")
@click.option("--once", is_flag=True, help="Capture new turns once and exit (no tailing).")
@click.option("--distill", is_flag=True, help="Also extract facts (LLM cost); default captures raw only.")
@click.option("--interval", default=3.0, show_default=True, help="Seconds between polls when tailing.")
@click.option("--start", is_flag=True, help="Start capturing in the background.")
@click.option("--stop", is_flag=True, help="Stop capturing in the background.")
def watch(
    dry_run: bool,
    catch_up: bool,
    once: bool,
    distill: bool,
    interval: float,
    start: bool,
    stop: bool,
) -> None:
    """Silently capture Claude Code and Codex conversations into memory."""

    import time as _time

    if start and stop:
        raise click.UsageError("--start and --stop cannot be combined.")
    if stop:
        _uninstall_background_service()
        return
    if start:
        _install_background_service()
        return

    from meniscus.db import get_connection, init_db
    from meniscus.event_intake import ingest_event
    from meniscus.transcript_ingest import discover_sources, parse_turns, read_new_lines

    conn = get_connection()
    init_db(conn)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ingest_cursors "
        "(path TEXT PRIMARY KEY, byte_offset INTEGER NOT NULL DEFAULT 0)"
    )
    conn.commit()

    if dry_run:
        total = 0
        sources = discover_sources()
        for source in sources:
            lines, _ = read_new_lines(source.path, 0)
            turns = parse_turns(source.kind, lines)
            if turns:
                click.echo(f"  {len(turns):>5} turns  {source.label}")
                total += len(turns)
        click.echo(f"\nWould capture {total} turn(s) across {len(sources)} transcript(s). Nothing written (--dry-run).")
        conn.close()
        return

    def _offset(path: str) -> int:
        row = conn.execute("SELECT byte_offset FROM ingest_cursors WHERE path = ?", (path,)).fetchone()
        return int(row["byte_offset"]) if row else 0

    def _drain() -> int:
        captured = 0
        for source in discover_sources():
            key = str(source.path)
            offset = _offset(key)
            lines, new_offset = read_new_lines(source.path, offset)
            if new_offset == offset:
                continue
            for turn in parse_turns(source.kind, lines):
                ingest_event(conn, turn.text, source.label, timestamp=turn.timestamp or None)
                captured += 1
            conn.execute(
                "INSERT INTO ingest_cursors (path, byte_offset) VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET byte_offset = excluded.byte_offset",
                (key, new_offset),
            )
            conn.commit()
        return captured

    def _distill() -> None:
        from meniscus.exceptions import MeniscusError
        from meniscus.pipeline import process_pending_events
        from meniscus.providers import get_embedding_model, get_model

        pending = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = ?",
            (ExtractionStatus.PENDING,),
        ).fetchone()[0]
        if pending == 0:
            return

        try:
            model, embedding_model = get_model(), get_embedding_model()
            processed = process_pending_events(conn, model, embedding_model)
            remaining = pending - len(processed)
            if remaining:
                click.echo(
                    f"  {remaining} raw turn(s) remain pending; Meniscus will retry.",
                    err=True,
                )
        except MeniscusError as exc:
            click.echo(
                f"  Memory processing unavailable ({exc}); raw turns remain pending.",
                err=True,
            )

    if catch_up:
        for source in discover_sources():
            conn.execute(
                "INSERT INTO ingest_cursors (path, byte_offset) VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET byte_offset = excluded.byte_offset",
                (str(source.path), source.path.stat().st_size),
            )
        conn.commit()
        click.echo("Caught up to now — past turns skipped, capturing from here on.")
    else:
        captured = _drain()
        click.echo(f"Captured {captured} new turn(s).")

    if distill:
        _distill()

    if once:
        conn.close()
        return

    click.echo(f"Watching for new turns every {interval:g}s… (Ctrl-C to stop)")
    try:
        while True:
            _time.sleep(interval)
            new = _drain()
            if new:
                click.echo(f"  +{new} turn(s)")
                if distill:
                    _distill()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        conn.close()


@cli.command()
@click.argument("question")
@click.option("--json", "as_json", is_flag=True, help="Print raw structured facts.")
@click.option("--limit", default=None, type=int, help="Maximum facts to return.")
def ask(question: str, as_json: bool, limit: int | None) -> None:
    """Ask Meniscus using deterministic fact retrieval plus optional synthesis."""

    from meniscus import config
    from meniscus.exceptions import ModelUnavailableError
    from meniscus.fact_retrieval import recent_facts, retrieve
    from meniscus.time_bounds import normalize_time_window

    resolved_limit = (
        config.RETRIEVAL_DEFAULT_LIMIT if limit is None else limit
    )
    def status(message: str) -> None:
        # Keep JSON output machine-readable while making interactive waits
        # visible. Status goes to stderr so answer text remains pipe-friendly.
        if not as_json:
            click.echo(f"  {message}", err=True)

    status("Opening your memory…")
    conn, model, embedding_model, embedding_ready = _get_retrieval_resources()
    try:
        params = RetrievalParams(text=question)
        model_ready = model is not None
        _ = embedding_ready
        if model_ready and model is not None:
            status("Understanding your question…")
            prompt = RETRIEVAL_PARAM_PROMPT_TEMPLATE.format(
                today=date.today().isoformat(),
                question=question,
            )
            try:
                params = model.generate_structured(prompt, RetrievalParams)
            except ModelUnavailableError:
                model_ready = False
                params = RetrievalParams(text=question)

        try:
            normalize_time_window(params.start, params.end)
        except ValueError:
            model_ready = False
            params = RetrievalParams(text=question)

        degraded: list[str] = []

        def note_degraded(note: str) -> None:
            if note not in degraded:
                degraded.append(note)

        if params.text:
            status("Searching your memory…")
            facts = retrieve(
                conn,
                text=params.text,
                start=params.start,
                end=params.end,
                limit=resolved_limit,
                embedding_model=embedding_model,
                on_degraded=note_degraded,
            )
        else:
            status("Looking through recent memory…")
            facts = recent_facts(
                conn,
                start=params.start,
                end=params.end,
                limit=resolved_limit,
            )

        facts_payload = [asdict(fact) for fact in facts]
        if as_json:
            click.echo(
                json.dumps(
                    {"facts": facts_payload, "count": len(facts), "degraded": degraded}
                    if degraded
                    else {"facts": facts_payload, "count": len(facts)}
                )
            )
            return
        for note in degraded:
            click.echo(f"  ⚠ {note}", err=True)
        if model is None:
            click.echo(
                "Note: no LLM configured, so this shows raw matched facts instead of "
                "a written answer. Run `men doctor` to check your setup.\n",
                err=True,
            )
            click.echo(json.dumps({"facts": facts_payload, "count": len(facts)}))
            return
        if not model_ready:
            click.echo(
                "Note: the LLM call failed (bad key, no credit, or unreachable), so this "
                "shows raw matched facts instead of a written answer. Run `men doctor` "
                "to check your key.\n",
                err=True,
            )
            click.echo(json.dumps({"facts": facts_payload, "count": len(facts)}))
            return

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            question=question,
            facts=json.dumps(facts_payload),
        )
        try:
            status("Writing your answer…")
            answer = model.generate_structured(prompt, AskAnswer)
        except ModelUnavailableError:
            click.echo(
                "Note: the LLM call failed (bad key, no credit, or unreachable), so this "
                "shows raw matched facts instead of a written answer. Run `men doctor` "
                "to check your key.\n",
                err=True,
            )
            click.echo(json.dumps({"facts": facts_payload, "count": len(facts)}))
            return
        if answer.answered and facts and answer.answer.strip():
            click.echo(answer.answer)
            return
        no_answer = answer.answer if not answer.answered else ""
        _print_unanswered(no_answer, facts, params)
    finally:
        conn.close()


def _print_unanswered(
    message: str,
    facts: list,
    params: RetrievalParams,
) -> None:
    """Print model prose plus deterministic inspection commands."""

    click.echo(message.strip() or "I don't have anything on that in your memory.")

    event_ids: list[int] = []
    if facts:
        click.echo("")
        click.echo("Closest facts:")
        for fact in facts:
            click.echo(f"  [{_date_part(fact.timestamp)}] {fact.text} (event {fact.event_id})")
            if fact.event_id not in event_ids:
                event_ids.append(fact.event_id)

    start, end = _guidance_window(params, facts)
    click.echo("")
    click.echo("Try:")
    for event_id in event_ids:
        click.echo(f"  men show event {event_id}")
    click.echo(f"  {_windowed_command('men list events', start, end)}")
    click.echo("  men status")


def _guidance_window(
    params: RetrievalParams,
    facts: list,
) -> tuple[str | None, str | None]:
    dates = [d for fact in facts if (d := _date_part(fact.timestamp)) is not None]
    return (
        _date_part(params.start) or (min(dates) if dates else None),
        _date_part(params.end) or (max(dates) if dates else None),
    )


def _date_part(value: str | None) -> str | None:
    if not value:
        return None
    return value[:10]


def _format_date_range(start: str | None, end: str | None) -> str:
    if start and end and start != end:
        return f"{start} to {end}"
    return start or end or "date unknown"


def _windowed_command(
    command: str,
    start: str | None,
    end: str | None,
) -> str:
    parts = [command]
    if start:
        parts.append(f"--since {start}")
    if end:
        parts.append(f"--until {end}")
    return " ".join(parts)


@cli.group("list")
def list_group() -> None:
    """List stored events."""


@list_group.command("events")
@click.option("--since", default=None, help="Only events at/after this ISO date.")
@click.option("--until", default=None, help="Only events at/before this ISO date.")
@click.option("--source", default=None, help="Only events from this source.")
@click.option("--limit", default=20, help="Number of events to show.")
def list_events(
    since: str | None,
    until: str | None,
    source: str | None,
    limit: int,
) -> None:
    """Show stored events."""

    from meniscus.time_bounds import normalize_time_window

    try:
        start_bound, end_bound = normalize_time_window(since, until)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    clauses: list[str] = []
    params: list[object] = []
    if start_bound is not None:
        clauses.append("timestamp >= ?")
        params.append(start_bound)
    if end_bound is not None:
        clauses.append("timestamp <= ?")
        params.append(end_bound)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.append(max(limit, 0))

    conn = _read_connection()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, source, extraction_status, content "
            f"FROM events {where}ORDER BY timestamp DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
        if not rows:
            click.echo("No events found.")
            return
        for row in rows:
            preview = row["content"][:80].replace("\n", " ")
            marker = "+" if row["extraction_status"] == ExtractionStatus.COMPLETED else "o"
            click.echo(
                f"[{marker}] {row['id']:>5} | {row['timestamp'][:19]} | "
                f"{row['source']:<15} | {preview}"
            )
    finally:
        conn.close()


@cli.group("show")
def show_group() -> None:
    """Show details for an event."""


@show_group.command("event")
@click.argument("event_id", type=int)
def show_event(event_id: int) -> None:
    """Show event details."""

    conn = _read_connection()
    try:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if event is None:
            click.echo(f"Event {event_id} not found.", err=True)
            sys.exit(1)

        entities = conn.execute(
            "SELECT DISTINCT en.canonical_name "
            "FROM facts f "
            "JOIN fact_entity_edges fe ON fe.fact_id = f.id "
            "JOIN entities en ON fe.entity_id = en.id "
            "WHERE f.event_id = ? "
            "ORDER BY en.canonical_name",
            (event_id,),
        ).fetchall()
        # Every fact carries the model that wrote it; surface it so attribution is
        # auditable without dropping into `men sql`.
        extractions = conn.execute(
            "SELECT provider, model, prompt_version, extracted_at "
            "FROM extractions WHERE event_id = ? ORDER BY id",
            (event_id,),
        ).fetchall()

        click.echo(f"Event #{event['id']}")
        click.echo(f"  Source:   {event['source']}")
        click.echo(f"  Time:     {event['timestamp']}")
        click.echo(f"  Status:   {event['extraction_status']}")
        click.echo(f"  Entities: {', '.join(row['canonical_name'] for row in entities)}")
        for row in extractions:
            click.echo(
                f"  Facts by: {row['provider']}/{row['model']} "
                f"({row['prompt_version']}) at {row['extracted_at']}"
            )
        click.echo("")
        click.echo(event["content"])
    finally:
        conn.close()


@cli.command()
def status() -> None:
    """System stats."""

    from meniscus import config

    conn = _read_connection()
    try:
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        pending_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = ?",
            (ExtractionStatus.PENDING,),
        ).fetchone()[0]
        completed_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = ?",
            (ExtractionStatus.COMPLETED,),
        ).fetchone()[0]
        fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        latest = conn.execute("SELECT MAX(timestamp) FROM events").fetchone()[0]
        click.echo("Meniscus Status")
        click.echo(f"  Events:   {event_count} ({completed_count} completed, {pending_count} pending)")
        click.echo(f"  Last event: {_describe_age(latest)}")
        click.echo(f"  Facts:    {fact_count}")
        click.echo(f"  Entities: {entity_count}")
        if config.EMBEDDING_PROVIDER == config.EMBEDDING_DISABLED:
            click.echo("  Embeddings: disabled")
        else:
            embedding_count = conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
            click.echo(f"  Facts embedded: {embedding_count}")
            click.echo(f"  Facts without embeddings: {max(fact_count - embedding_count, 0)}")
        if pending_count:
            click.echo(
                f"\n  {pending_count} event(s) pending — run `men process` to process them."
            )
    finally:
        conn.close()


def _describe_age(timestamp: str | None) -> str:
    """Render how long ago an event landed, so a stalled capture is visible at a glance."""

    from datetime import datetime, timezone

    if not timestamp:
        return "never — nothing captured yet"
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    if seconds < _AGE_JUST_NOW_SECONDS:
        age = "just now"
    elif seconds < _AGE_MINUTES_SECONDS:
        age = f"{round(seconds / _SECONDS_PER_MINUTE)}m ago"
    elif seconds < _AGE_HOURS_SECONDS:
        age = f"{round(seconds / _SECONDS_PER_HOUR)}h ago"
    else:
        age = f"{round(seconds / _SECONDS_PER_DAY)}d ago"
    stale = "  ← capture may have stopped" if seconds > _STALE_CAPTURE_SECONDS else ""
    return f"{timestamp} ({age}){stale}"


_SHADOW_SUFFIXES = (
    "_data", "_idx", "_docsize", "_config", "_content",
    "_chunks", "_info", "_rowids", "_vector_chunks00",
)
_READ_ONLY_SQL_PREFIXES = {"select", "pragma", "explain", "with"}


def _is_shadow_table(name: str) -> bool:
    return name.startswith("sqlite_") or name.endswith(_SHADOW_SUFFIXES)


@cli.command()
def tables() -> None:
    """List the memory database's tables with row counts."""

    conn = _read_connection()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        click.echo(f"{'table':<26}{'rows':>10}")
        click.echo("-" * 36)
        for row in rows:
            name = row["name"]
            if _is_shadow_table(name):
                continue
            try:
                identifier = '"' + name.replace('"', '""') + '"'
                count = conn.execute(f"SELECT COUNT(*) FROM {identifier}").fetchone()[0]
            except Exception:
                count = "-"
            click.echo(f"{name:<26}{count:>10}")
    finally:
        conn.close()


@cli.command()
@click.argument("query")
def sql(query: str) -> None:
    """Run a read-only SQL query (SELECT/PRAGMA/EXPLAIN/WITH) against the memory."""

    import sqlite3

    words = query.lstrip().lower().split(None, 1)
    if not words or words[0] not in _READ_ONLY_SQL_PREFIXES:
        raise click.UsageError("Only read-only queries (SELECT / PRAGMA / EXPLAIN / WITH) are allowed.")

    conn = _read_connection()
    try:
        try:
            rows = conn.execute(query).fetchall()
        except sqlite3.OperationalError as exc:
            raise click.UsageError(str(exc)) from exc
        if not rows:
            click.echo("(no rows)")
            return
        columns = list(rows[0].keys())
        click.echo(" | ".join(columns))
        click.echo("-+-".join("-" * len(c) for c in columns))
        for row in rows:
            click.echo(" | ".join(_sql_cell(row[c]) for c in columns))
        click.echo(f"\n({len(rows)} rows)")
    finally:
        conn.close()


def _sql_cell(value: object, width: int = 60) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _read_connection():
    from meniscus.db import get_connection, init_db

    conn = get_connection()
    init_db(conn)
    conn.execute("PRAGMA query_only=ON")
    return conn


if __name__ == "__main__":
    cli()
