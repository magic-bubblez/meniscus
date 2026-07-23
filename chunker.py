"""Content chunking for large ingested texts."""

from __future__ import annotations

import re

from config import CHUNK_SIZE_WORDS

_HEADING_RE = re.compile(r"(?m)(?=^#{1,6}\s+)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def count_words(text: str) -> int:
    """Count whitespace-delimited words."""

    return len(text.split())


def chunk_content(content: str) -> list[str]:
    """Split content into chunks no larger than CHUNK_SIZE_WORDS."""

    if count_words(content) <= CHUNK_SIZE_WORDS:
        return [content]

    for splitter in (_split_headings, _split_paragraphs, _split_sentences):
        segments = [segment for segment in splitter(content) if segment.strip()]
        chunks = _greedy_merge(segments)
        if chunks is not None:
            return chunks

    words = content.split()
    return [
        " ".join(words[i : i + CHUNK_SIZE_WORDS])
        for i in range(0, len(words), CHUNK_SIZE_WORDS)
    ]


def _split_headings(content: str) -> list[str]:
    return _HEADING_RE.split(content)


def _split_paragraphs(content: str) -> list[str]:
    return content.split("\n\n")


def _split_sentences(content: str) -> list[str]:
    return _SENTENCE_RE.split(content)


def _greedy_merge(segments: list[str]) -> list[str] | None:
    """Merge segments into the fewest chunks that satisfy the word cap."""

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for segment in segments:
        segment = segment.strip()
        segment_words = count_words(segment)
        if segment_words > CHUNK_SIZE_WORDS:
            return None
        if current and current_words + segment_words > CHUNK_SIZE_WORDS:
            chunks.append("\n\n".join(current))
            current = [segment]
            current_words = segment_words
        else:
            current.append(segment)
            current_words += segment_words

    if current:
        chunks.append("\n\n".join(current))
    return chunks
