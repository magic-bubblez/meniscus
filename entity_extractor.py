"""LLM-based entity extraction for events."""

from __future__ import annotations

import sqlite3

import config
from model_interface import ModelInterface
from models import ExtractionResult

# Bumped when the prompt changes; recorded per extraction for reproducibility.
EXTRACTION_PROMPT_VERSION: str = "extract-v3-entities+facts-cacheable"

EXTRACTION_PROMPT_TEMPLATE: str = """\
# Your role
You read one thing a person recorded -- a note, a message, a fragment of something
they were working on -- together with when they recorded it. From it you produce
two things:

- the ENTITIES it is about -- the specific, nameable things that let this record
  connect to others;
- the FACTS it states -- the individual claims it makes, each rewritten to stand
  on its own.

Read only this text. Name what it is genuinely about, and state only what it
actually claims -- faithfully, adding nothing the text does not contain. You are
not summarizing and not interpreting; you are re-expressing what is already here
in a cleaner, smaller form.

# Part 1 -- Entities
An entity is a specific, nameable thing the text is about: a technology, a tool, a
topic, a technique, a named project, a person, an idea. The test is connection --
would another record about the same thing name this same entity? Pull the things
that carry the meaning, not the words that merely fill it out.

- Prefer specific over generic. "garbage collection" identifies something;
  "programming" is too broad to connect anything.
- Keep compound entities whole. "system calls", "linked list", "refresh token"
  are each one entity; do not shred them into words that mean something different
  apart than together.
- Leave out scaffolding: stopwords, articles, prepositions, filler, generic verbs
  and adjectives ("did", "made", "good"). If a word could sit in any record at
  all, it is noise.
- Lean toward completeness. Extract every entity that genuinely identifies what
  the text is about -- central and secondary alike -- and stop only at generic
  filler. An entity left out is a link this record will never make; an entity that
  turns out common is quietly down-weighted later. Completeness is not padding:
  never inflate the count with noise.

Give each entity a clean canonical name so the same thing written different ways
lands in one place: lowercase, singular, and abbreviations written out to their
full form as the name. For each, include aliases -- a short form, an abbreviation,
a synonym -- so "machine learning" and "ML", or "kubernetes" and "k8s", resolve to
one entity; when there is no alternative form, the alias list is empty. Read only
this text to name its entities; reconciling them against entities seen elsewhere
happens deterministically after you.

You may name an entity the text plainly and unambiguously implies, but do not
guess beyond that. Extract at most [ENTITY_CAP] entities; if the text somehow
holds more, keep the most identifying ones.

# Part 2 -- Facts
A fact is a single claim the text makes, rewritten so it can be understood on its
own. Two properties, both required:

- Atomic -- one claim per fact. If a sentence asserts two things ("I met David and
  moved the wedding to June"), it is two facts. A claim you cannot divide without
  losing part of it is the right size; dividing past that point yields fragments,
  not facts.
- Self-contained -- understandable with nothing else in front of the reader.
  Resolve every reference to what it points at: "he" to the person named, "there"
  to the place named, "that" to the thing named. Resolve using the entities you
  named in Part 1 and nothing else -- if this text does not make the referent
  clear, leave it unresolved rather than guess. An unresolved fact is weak; an
  invented one is wrong, and a wrong fact is worse than a missing one, because the
  missing one can still be recovered from this record and the wrong one cannot.

The test for both: read the fact alone, months from now, with this record gone. If
it still says something true and complete, it is a fact. If it leaves you asking
"who?", "which?", or "when?", it is not finished.

Keep exactly what the text commits to. "I might quit" is not "I quit"; "we agreed
to consider June" is not "the wedding is in June". Do not harden a possibility into
a certainty, soften a decision into a maybe, or add certainty the writer did not
express. You strip packaging; you never change the claim.

Each record carries its timestamp, shown as "Recorded:" below. Resolve time-relative
words against it -- "yesterday", "last week", "next month" become the actual point
they name -- so each fact still locates itself in time when read later. If the text
names no time, add none.

Capture every claim worth remembering -- the central ones and the incidental ones
the person may later want to find. Leave out only what states nothing: greetings,
filler, thinking-aloud that asserts no claim. A claim you drop is a fact this
record will never surface, so keep a real but minor claim rather than discard it.
Never manufacture a claim the text does not make; if the text states nothing worth
keeping, return no facts.

# What you return
A structured result with two lists:
- entities -- each a canonical name and a list of aliases (possibly empty);
- facts -- each a single self-contained declarative statement, in the order the
  claims arise in the text.

Recorded: {timestamp}
Source: {source}
Content: {content}
"""


def build_extraction_prompt(source: str, content: str, timestamp: str) -> str:
    """Build the extraction prompt for one event."""

    return (
        EXTRACTION_PROMPT_TEMPLATE
        .replace("[ENTITY_CAP]", str(config.ENTITY_CAP))
        .format(source=source, content=content, timestamp=timestamp)
    )


def extract_from_content(
    source: str,
    content: str,
    model: ModelInterface,
    timestamp: str,
) -> ExtractionResult:
    """Extract entities + facts from raw content (no DB access; parallel-safe)."""

    prompt = build_extraction_prompt(source, content, timestamp)
    result = model.generate_structured(prompt, ExtractionResult)
    if len(result.entities) > config.ENTITY_CAP:
        return ExtractionResult(
            entities=result.entities[: config.ENTITY_CAP],
            facts=result.facts,
        )
    return result


def extract_entities(
    conn: sqlite3.Connection,
    event_id: int,
    model: ModelInterface,
) -> ExtractionResult:
    """Extract entities + facts from an event using the configured LLM."""

    row = conn.execute(
        "SELECT content, source, timestamp FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Event {event_id} not found")

    return extract_from_content(row["source"], row["content"], model, row["timestamp"])
