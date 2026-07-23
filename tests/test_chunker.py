from __future__ import annotations

import chunker


def test_chunk_content_small():
    assert chunker.chunk_content("hello world") == ["hello world"]


def test_chunk_content_word_fallback(monkeypatch):
    monkeypatch.setattr(chunker, "CHUNK_SIZE_WORDS", 3)
    chunks = chunker.chunk_content("one two three four five six seven")
    assert [chunker.count_words(chunk) for chunk in chunks] == [3, 3, 1]


def test_greedy_merge():
    monkey = ["one two", "three", "four five"]
    chunks = chunker._greedy_merge(monkey)
    assert chunks == ["one two\n\nthree\n\nfour five"]
