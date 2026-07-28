"""Tests for the evaluation-only fixed-size chunking baseline."""

from __future__ import annotations

import pytest

from rag_pipeline.fixed_chunking import chunk_fixed_size


def test_short_text_returns_single_chunk() -> None:
    assert chunk_fixed_size("Kurzer Text.", max_chars=100, overlap_chars=20) == [
        "Kurzer Text.",
    ]


def test_empty_and_whitespace_text_returns_no_chunks() -> None:
    assert chunk_fixed_size("") == []
    assert chunk_fixed_size("   \n\n  ") == []


def test_chunks_respect_max_chars() -> None:
    text = " ".join(f"wort{i}" for i in range(500))

    chunks = chunk_fixed_size(text, max_chars=200, overlap_chars=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_words_are_never_split() -> None:
    text = " ".join(f"wort{i}" for i in range(500))
    words = set(text.split())

    chunks = chunk_fixed_size(text, max_chars=180, overlap_chars=30)

    for chunk in chunks:
        for token in chunk.split():
            assert token in words


def test_overlap_repeats_trailing_context() -> None:
    text = " ".join(f"wort{i}" for i in range(500))

    chunks = chunk_fixed_size(text, max_chars=200, overlap_chars=60)

    for previous, current in zip(chunks, chunks[1:]):
        first_word = current.split()[0]
        assert first_word in previous


def test_zero_overlap_covers_full_text_without_repeats() -> None:
    text = " ".join(f"wort{i}" for i in range(300))

    chunks = chunk_fixed_size(text, max_chars=150, overlap_chars=0)

    assert " ".join(chunks).split() == text.split()


def test_deterministic_for_identical_input() -> None:
    text = " ".join(f"wort{i}" for i in range(400))

    assert chunk_fixed_size(text) == chunk_fixed_size(text)


def test_oversized_single_token_still_makes_progress() -> None:
    text = "a" * 5000

    chunks = chunk_fixed_size(text, max_chars=100, overlap_chars=10)

    assert chunks
    assert sum(len(chunk) for chunk in chunks) >= 5000 - 100


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        chunk_fixed_size("text", max_chars=0)
    with pytest.raises(ValueError):
        chunk_fixed_size("text", max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_fixed_size("text", max_chars=100, overlap_chars=-1)
