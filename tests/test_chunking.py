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


def test_text_of_exactly_chunk_size_becomes_one_chunk():
    """
    Regression test for the defect found in Phase 1 and fixed in Phase 2.

    The loop used to advance past a chunk that had already reached the end of
    the text, emitting the trailing `overlap` words again as a standalone chunk
    that was a verbatim substring of its predecessor. 100 words at
    chunk_size=100/overlap=20 produced two chunks, the second being words 80-99
    a second time.
    """
    text = words(100)
    assert chunk_text(text, chunk_size=100, overlap=20) == [text]


def test_no_chunk_is_contained_in_another():
    """
    The general form of the same defect: no chunk should ever be a substring of
    another, at any text length. Lengths either side of a chunk boundary are the
    cases that used to fail.
    """
    for length in (100, 180, 181, 250, 300, 437, 500):
        chunks = chunk_text(words(length), chunk_size=100, overlap=20)
        for i, chunk in enumerate(chunks):
            for j, other in enumerate(chunks):
                if i != j:
                    assert chunk not in other, (
                        f"at {length} words, chunk {i} is contained in chunk {j}"
                    )


def test_a_long_tail_still_produces_a_final_chunk():
    """The fix must not truncate: a tail with new words still gets a chunk."""
    chunks = chunk_text(words(300), chunk_size=100, overlap=20)

    assert chunks[-1].split()[-1] == "w299"


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
