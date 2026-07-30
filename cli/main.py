"""Command line interface for Meniscus."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date

import click
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class RetrievalParams(BaseModel):
    """Structured query parameters extracted from a natural-language question."""

    text: str | None = None
    start: str | None = None
    end: str | None = None


class AskAnswer(BaseModel):
    """Final answer for the CLI ask command."""

    answered: bool
    answer: str


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


class _FriendlyGroup(click.Group):
    """Turn known setup errors into a clean one-line message, not a traceback."""

    def invoke(self, ctx: click.Context):
        from exceptions import MeniscusError

        try:
            return super().invoke(ctx)
        except MeniscusError as exc:
            # MeniscusError messages already carry the fix hint (e.g. "Install
            # sqlite-vec, or set EMBEDDING_PROVIDER='none'"). Print it plainly.
            click.echo(f"Error: {exc}", err=True)
            click.echo("Run `men doctor` to see what's missing.", err=True)
            raise SystemExit(1)


@click.group(cls=_FriendlyGroup)
def cli() -> None:
    """Meniscus: local structured memory for AI agents."""


@cli.command()
def doctor() -> None:
    """Check the environment and report what (if anything) is missing."""

    import config
    from db import get_connection, get_db_path, init_db
    from exceptions import MeniscusError, ModelUnavailableError

    def line(name: str, ok: bool | None, detail: str) -> None:
        mark = "✓" if ok else ("•" if ok is None else "✗")
        click.echo(f"  {mark} {name:<14} {detail}")

    click.echo("Meniscus setup check\n")

    # database path
    try:
        db_path = get_db_path()
        line("database", True, str(db_path))
    except Exception as exc:  # pragma: no cover - defensive
        line("database", False, f"could not resolve path: {exc}")

    # sqlite-vec
    provider = config.EMBEDDING_PROVIDER
    if provider == "none":
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
                'not installed  →  pipx install "meniscus[vec]"  '
                '(or set EMBEDDING_PROVIDER="none")',
            )
        except Exception as exc:
            line("sqlite-vec", False, f"installed but cannot load: {exc}")

    # embedding config
    line(
        "embeddings",
        None,
        f'provider="{provider}", {config.EMBEDDING_DIMENSIONS} dims',
    )

    # LLM provider (bring your own — required to extract & group). Actually try
    # to BUILD it: this catches both a missing key and a missing SDK, which a
    # bare env-var check would miss.
    from providers import get_model

    llm = config.DEFAULT_MODEL_PROVIDER
    try:
        get_model()
        line("LLM provider", True, f'provider="{llm}" ready')
    except ModelUnavailableError as exc:
        line(
            "LLM provider",
            False,
            f"{exc}  (required to extract & group)",
        )
    except ImportError:
        line(
            "LLM provider",
            False,
            f'SDK not installed  →  pip install "meniscus[{llm}]"  '
            "(required to extract & group)",
        )
    except Exception as exc:  # pragma: no cover - defensive
        line("LLM provider", False, str(exc))

    # does the system actually initialize?
    conn = get_connection()
    try:
        init_db(conn)
        line("startup", True, "database initializes cleanly")
        startup_ok = True
    except MeniscusError as exc:
        line("startup", False, str(exc))
        startup_ok = False
    finally:
        conn.close()

    click.echo("")
    if startup_ok:
        click.echo(
            "Ready. You can `men add` notes now; if no model is set they queue "
            "until you configure one and run `men process`."
        )
    else:
        click.echo("Fix the ✗ items above, then run `men doctor` again.")


def _get_resources():
    """Open the DB and best-effort model resources for the write path.

    A missing/unavailable model does NOT fail capture: intake is deterministic,
    so events are saved as pending and processed later. Genuine configuration
    errors (dimension mismatch, unknown provider) still surface via init_db /
    the provider factories.
    """

    from db import get_connection, init_db
    from exceptions import ModelUnavailableError
    from providers import get_embedding_model, get_model
    from startup import announce_embedding_state

    conn = get_connection()
    try:
        init_db(conn)
        announce_embedding_state()
        try:
            model = get_model()
        except (ModelUnavailableError, ImportError):
            model = None
        try:
            embedding_model = get_embedding_model()
        except (ModelUnavailableError, ImportError):
            embedding_model = None
    except Exception:
        conn.close()
        raise
    return conn, model, embedding_model


def _get_retrieval_resources():
    """Open the DB and best-effort model resources for retrieval."""

    from db import get_connection, init_db
    from exceptions import ModelUnavailableError
    from providers import get_embedding_model, get_model

    conn = get_connection()
    try:
        init_db(conn)
        # No "embeddings disabled" notice on read/query commands — it would
        # pollute machine-readable output (e.g. `ask --json`). It stays on the
        # ingest commands, where the heads-up actually matters.

        try:
            model = get_model()
        except (ModelUnavailableError, ImportError):
            model = None

        embedding_ready = True
        try:
            embedding_model = get_embedding_model()
        except (ModelUnavailableError, ImportError):
            embedding_model = None
            embedding_ready = False
    except Exception:
        conn.close()
        raise

    return conn, model, embedding_model, embedding_ready


@cli.command()
@click.argument("message", required=False)
@click.option("--source", default="cli", help="Source identifier for this event.")
def add(message: str | None, source: str) -> None:
    """Ingest a single event and process immediately.

    Pass the note as an argument, or omit it (or pass '-') to read the whole
    note from stdin — the clean way to capture multi-line or pasted text without
    fighting shell quoting:

        pbpaste | men add
        men add < note.md
        men add           # then type, and press Ctrl-D when done
    """

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
        from pipeline import ingest_and_process

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
        from pipeline import import_and_process

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
            f"WHERE id IN ({placeholders}) AND extraction_status = 'pending'",
            event_ids,
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
        from pipeline import process_pending_events

        if model is None:
            click.echo(
                "No model configured, so nothing was processed. Set GEMINI_API_KEY "
                "(see `men doctor`) and run `men process` again."
            )
            return

        pending = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = 'pending'"
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
@click.argument("question")
@click.option("--json", "as_json", is_flag=True, help="Print raw structured facts.")
@click.option("--limit", default=None, type=int, help="Maximum facts to return.")
def ask(question: str, as_json: bool, limit: int | None) -> None:
    """Ask Meniscus using deterministic fact retrieval plus optional synthesis."""

    import config
    from exceptions import ModelUnavailableError
    from fact_retrieval import recent_facts, retrieve
    from time_bounds import normalize_time_window

    resolved_limit = (
        config.RETRIEVAL_DEFAULT_LIMIT if limit is None else limit
    )
    conn, model, embedding_model, embedding_ready = _get_retrieval_resources()
    try:
        params = RetrievalParams(text=question)
        model_ready = model is not None
        _ = embedding_ready
        if model_ready and model is not None:
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

        if params.text:
            facts = retrieve(
                conn,
                text=params.text,
                start=params.start,
                end=params.end,
                limit=resolved_limit,
                embedding_model=embedding_model,
            )
        else:
            facts = recent_facts(
                conn,
                start=params.start,
                end=params.end,
                limit=resolved_limit,
            )

        facts_payload = [asdict(fact) for fact in facts]
        if as_json or not model_ready or model is None:
            click.echo(json.dumps({"facts": facts_payload, "count": len(facts)}))
            return

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            question=question,
            facts=json.dumps(facts_payload),
        )
        try:
            answer = model.generate_structured(prompt, AskAnswer)
        except ModelUnavailableError:
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
    """List stored events or threads."""


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

    from time_bounds import normalize_time_window

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
            marker = "+" if row["extraction_status"] == "completed" else "o"
            click.echo(
                f"[{marker}] {row['id']:>5} | {row['timestamp'][:19]} | "
                f"{row['source']:<15} | {preview}"
            )
    finally:
        conn.close()


@list_group.command("threads")
@click.option("--since", default=None, help="Only threads with an event at/after this ISO date.")
@click.option("--until", default=None, help="Only threads with an event at/before this ISO date.")
@click.option("--limit", default=20, help="Number of threads to show.")
def list_threads(since: str | None, until: str | None, limit: int) -> None:
    """Show thread summaries."""

    from time_bounds import normalize_time_window

    try:
        start_bound, end_bound = normalize_time_window(since, until)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    event_clauses: list[str] = []
    params: list[object] = []
    if start_bound is not None:
        event_clauses.append("e.timestamp >= ?")
        params.append(start_bound)
    if end_bound is not None:
        event_clauses.append("e.timestamp <= ?")
        params.append(end_bound)
    if event_clauses:
        where = (
            "WHERE EXISTS ("
            "SELECT 1 FROM event_thread_edges ete2 "
            "JOIN events e ON e.id = ete2.event_id "
            f"WHERE ete2.thread_id = t.id AND {' AND '.join(event_clauses)}"
            ") "
        )
    else:
        where = ""
    params.append(max(limit, 0))

    conn = _read_connection()
    try:
        rows = conn.execute(
            "SELECT t.id, t.title, t.summary, t.created_at, t.updated_at, "
            "COUNT(ete.event_id) as event_count "
            "FROM threads t "
            "LEFT JOIN event_thread_edges ete ON t.id = ete.thread_id "
            f"{where}"
            "GROUP BY t.id "
            "ORDER BY t.updated_at DESC, t.id DESC LIMIT ?",
            params,
        ).fetchall()
        if not rows:
            click.echo("No threads found.")
            return
        for row in rows:
            click.echo(
                f"{row['id']:>5} | {(row['title'] or '(untitled)'):<50} | "
                f"{row['event_count']} events | {row['updated_at'][:19]}"
            )
    finally:
        conn.close()


@cli.group("show")
def show_group() -> None:
    """Show details for an event or thread."""


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
            "SELECT en.canonical_name "
            "FROM event_entity_edges ee "
            "JOIN entities en ON ee.entity_id = en.id "
            "WHERE ee.event_id = ? "
            "ORDER BY en.canonical_name",
            (event_id,),
        ).fetchall()
        thread = conn.execute(
            "SELECT thread_id FROM event_thread_edges WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        click.echo(f"Event #{event['id']}")
        click.echo(f"  Source:   {event['source']}")
        click.echo(f"  Time:     {event['timestamp']}")
        click.echo(f"  Status:   {event['extraction_status']}")
        click.echo(f"  Thread:   {thread['thread_id'] if thread else '(unassigned)'}")
        click.echo(f"  Entities: {', '.join(row['canonical_name'] for row in entities)}")
        click.echo("")
        click.echo(event["content"])
    finally:
        conn.close()


@show_group.command("thread")
@click.argument("thread_id", type=int)
def show_thread(thread_id: int) -> None:
    """Show a thread with all events in chronological order."""

    conn = _read_connection()
    try:
        thread = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if thread is None:
            click.echo(f"Thread {thread_id} not found.", err=True)
            sys.exit(1)
        click.echo(f"Thread #{thread['id']}: {thread['title'] or '(untitled)'}")
        click.echo(thread["summary"])
        rows = conn.execute(
            "SELECT e.id, e.timestamp, e.source, e.content "
            "FROM events e "
            "JOIN event_thread_edges ete ON e.id = ete.event_id "
            "WHERE ete.thread_id = ? "
            "ORDER BY e.timestamp ASC",
            (thread_id,),
        ).fetchall()
        for row in rows:
            click.echo("")
            click.echo(f"[{row['id']}] {row['timestamp']} ({row['source']})")
            click.echo(row["content"])
    finally:
        conn.close()


@cli.command("rebuild-threads")
def rebuild_threads() -> None:
    """Clear derived thread state and recompute from source of truth."""

    conn, model, _embedding_model = _get_resources()
    try:
        from db import transactional
        from thread_assigner import assign_thread
        from thread_summarizer import summarize_thread
        from meniscus_types import ExtractionStatus

        with transactional(conn) as txn:
            txn.execute("DELETE FROM assignment_log")
            txn.execute("DELETE FROM event_thread_edges")
            txn.execute("DELETE FROM threads")

        events = conn.execute(
            "SELECT id FROM events WHERE extraction_status = ? ORDER BY timestamp ASC",
            (ExtractionStatus.COMPLETED,),
        ).fetchall()
        if not events:
            click.echo(
                "Nothing to rebuild — no processed events yet. "
                "Add and process some notes first."
            )
            return

        affected: set[int] = set()
        for row in events:
            event_id = int(row["id"])
            entity_rows = conn.execute(
                "SELECT entity_id FROM event_entity_edges WHERE event_id = ?",
                (event_id,),
            ).fetchall()
            entity_ids = [int(r["entity_id"]) for r in entity_rows]
            with transactional(conn) as txn:
                affected.add(assign_thread(txn, event_id, entity_ids))

        summary = f"Rebuilt {len(affected)} thread(s) from {len(events)} event(s)."
        if model is None:
            # Reassignment is deterministic and done; titles/summaries need a model.
            click.echo(
                summary + " Titles and summaries skipped — no model configured; "
                "run `men rebuild-threads` again with a model to regenerate them."
            )
        else:
            for thread_id in sorted(affected):
                summarize_thread(conn, thread_id, model)
            click.echo(summary)
    finally:
        conn.close()


@cli.command()
def status() -> None:
    """System stats."""

    import config

    conn = _read_connection()
    try:
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        pending_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = 'pending'"
        ).fetchone()[0]
        completed_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE extraction_status = 'completed'"
        ).fetchone()[0]
        fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        click.echo("Meniscus Status")
        click.echo(f"  Events:   {event_count} ({completed_count} completed, {pending_count} pending)")
        click.echo(f"  Facts:    {fact_count}")
        click.echo(f"  Entities: {entity_count}")
        if config.EMBEDDING_PROVIDER == "none":
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


_SHADOW_SUFFIXES = (
    "_data", "_idx", "_docsize", "_config", "_content",
    "_chunks", "_info", "_rowids", "_vector_chunks00",
)


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
                count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
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

    if query.lstrip().lower().split(None, 1)[0] not in ("select", "pragma", "explain", "with"):
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
    from db import get_connection, init_db

    conn = get_connection()
    init_db(conn)
    return conn


if __name__ == "__main__":
    cli()
