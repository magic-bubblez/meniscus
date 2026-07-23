"""LLM-based entity extraction for events."""

from __future__ import annotations

import sqlite3

import config
from model_interface import ModelInterface
from models import ExtractionResult

EXTRACTION_PROMPT_TEMPLATE: str = """\
# Your role
You read one piece of text a person recorded -- a note, a message, a fragment of
something they were working on -- and identify the concepts it is about. These
concepts become the connective tissue of their memory: the same concept,
extracted from two different notes, is what later links those notes together. So
your job is comprehension -- name what this text is genuinely about, faithfully,
and at a grain that will recur.

# What counts as a concept
A concept is a specific, nameable thing the text is about: a technology, a tool,
a topic, a technique, a named project, a person, an idea. The test is connection
-- would another note about the same thing name this same concept? If yes, it is
worth extracting. Pull out the things that carry the meaning of the text, not the
words that merely fill it out.

Prefer specific over generic. A term that would match half the person's notes
tells you nothing; a term that identifies this particular subject is what you
want -- for instance, "garbage collection" identifies something, whereas
"programming" is usually too broad to connect anything meaningfully.

Keep compound concepts whole. Things like "system calls", "linked list", or
"refresh token" are each a single concept; do not shred them into component
words, which mean something different apart than together.

# What to leave out
Do not extract stopwords, articles, prepositions, filler, or generic verbs and
adjectives ("did", "made", "good", "important"). Extract a sentence's subject
matter, not its scaffolding. If a word carries no identifying weight -- if it
could sit in any note at all -- it is noise, not a concept.

# How thorough to be
Lean toward completeness. Extract every concept that genuinely identifies what
the text is about -- the central ones and the secondary ones alike -- and stop
only where you reach generic filler. Prefer including a real but minor concept
over omitting it: a concept you leave out is a link this note will never make,
while a concept that turns out to be common is quietly down-weighted later. This
is not a license to pad -- completeness means capturing every identifying
concept, never inflating the count with generic noise.

# Canonical form
Give each concept a clean canonical name, so the same idea written different ways
lands in the same place: lowercase ("linked list", not "Linked List"), singular
("process", not "processes"), and the full form of any abbreviation as the name.
You do not need to consider whether a concept has appeared before or how it was
named elsewhere -- read only this text and name its concepts faithfully. Matching
this against what already exists happens deterministically after you; your job is
to see this text clearly, nothing more.

# Aliases
For each concept, include any alternative forms for the same thing -- an
abbreviation, its expansion, a common synonym. Aliases are how "ML" and "machine
learning", or "k8s" and "kubernetes", come to be understood as one concept rather
than two: give the full form as the name and the short form as an alias. When
there is no alternative form, the list is simply empty.

# Bounds
- Extract only concepts the text actually raises; do not add related ideas it
  doesn't mention.
- You may name a concept the text clearly and unambiguously implies, but do not
  guess beyond that.
- Extract at most [ENTITY_CAP] concepts; if the text somehow holds more, keep the
  most identifying ones.

# What you return
A structured result: a list of concepts, each with a canonical name and a list of
aliases (possibly empty).

Source: {source}
Content: {content}
"""


def build_extraction_prompt(source: str, content: str) -> str:
    """Build the extraction prompt for one event."""

    return (
        EXTRACTION_PROMPT_TEMPLATE
        .replace("[ENTITY_CAP]", str(config.ENTITY_CAP))
        .format(source=source, content=content)
    )


def extract_from_content(
    source: str,
    content: str,
    model: ModelInterface,
) -> ExtractionResult:
    """Extract entities from raw content (no DB access).

    This is the parallel-safe core used by the batch import path: it touches no
    database, so it can run in a thread pool concurrently with other extractions.
    """

    prompt = build_extraction_prompt(source, content)
    result = model.generate_structured(prompt, ExtractionResult)
    if len(result.entities) > config.ENTITY_CAP:
        return ExtractionResult(entities=result.entities[: config.ENTITY_CAP])
    return result


def extract_entities(
    conn: sqlite3.Connection,
    event_id: int,
    model: ModelInterface,
) -> ExtractionResult:
    """Extract entities from an event's content using the configured LLM."""

    row = conn.execute(
        "SELECT content, source FROM events WHERE id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Event {event_id} not found")

    return extract_from_content(row["source"], row["content"], model)
