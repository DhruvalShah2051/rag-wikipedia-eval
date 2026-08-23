"""
Tests for the ingestion chunking rule.

Chunk size and overlap are the two knobs that most directly affect retrieval
accuracy, and Phase 2 will sweep them, so the boundary behaviour needs to be
pinned down before anything starts varying it.
"""

import pytest

from chunking import chunk_text
from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def words(n, prefix="w"):
    """A text of exactly n distinguishable words."""
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_becomes_a_single_chunk():
    text = words(10)
    assert chunk_text(text, chunk_size=100, overlap=20) == [text]


def test_first_chunk_holds_the_first_chunk_size_words():
    text = words(100)
    assert chunk_text(text, chunk_size=100, overlap=20)[0] == text


def test_a_short_tail_produces_a_redundant_final_chunk():
    """
    KNOWN BEHAVIOUR, not a desired one.

    When the remaining words after the last full chunk all fall inside that
    chunk's span, the loop still emits them as a final chunk - a verbatim
    substring of its predecessor. 100 words at chunk_size=100/overlap=20 yields
    two chunks, the second being words 80-99 again.

    That inflates the stored chunk count and puts duplicate text in the vector
    store. It is pinned here rather than fixed because fixing it changes the
    recorded 416-chunk corpus and every measurement taken over it; the natural
    place to change it is the Phase 2 chunking sweep, which re-measures anyway.
    """
    chunks = chunk_text(words(100), chunk_size=100, overlap=20)

    assert len(chunks) == 2
    assert chunks[1] in chunks[0]


def test_a_long_tail_produces_a_genuinely_new_final_chunk():
    """The complement of the case above: a tail with new words is not redundant."""
    chunks = chunk_text(words(300), chunk_size=100, overlap=20)

    assert chunks[-1] not in chunks[-2]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n"])
def test_empty_or_whitespace_text_yields_no_chunks(text):
    assert chunk_text(text, chunk_size=100, overlap=20) == []


def test_no_chunk_exceeds_the_requested_size():
    chunks = chunk_text(words(1000), chunk_size=100, overlap=20)
    assert chunks
    assert all(len(c.split()) <= 100 for c in chunks)


def test_consecutive_chunks_share_exactly_the_overlap():
    chunk_size, overlap = 100, 20
    chunks = chunk_text(words(500), chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:]):
        tail = previous.split()[-overlap:]
        head = following.split()[:overlap]
        assert tail == head


def test_every_word_survives_chunking():
    """Overlap must not cause a word to be dropped between chunk boundaries."""
    text = words(437)
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    seen = []
    for chunk in chunks:
        for word in chunk.split():
            if word not in seen:
                seen.append(word)

    assert seen == text.split()


def test_whitespace_is_normalised():
    """Chunks are rebuilt with single spaces, so ragged source text is cleaned up."""
    chunks = chunk_text("alpha   beta\n\ngamma\tdelta", chunk_size=100, overlap=20)
    assert chunks == ["alpha beta gamma delta"]


def test_defaults_come_from_config():
    """
    The default arguments must track config.py, since that is what the sweep in
    Phase 2 will change.
    """
    text = words(CHUNK_SIZE_WORDS * 3)
    assert chunk_text(text) == chunk_text(
        text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS
    )
