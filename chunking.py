"""
Text chunking for the ingestion pipeline.

Lives in its own module rather than inside 3_chunk_and_embed.py because that
filename starts with a digit and so cannot be imported - by tests or by
anything else. The chunking rule is the one part of ingestion worth testing on
its own, since chunk size and overlap directly affect retrieval accuracy.
"""

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """
    Split text into overlapping chunks of `chunk_size` words.
    Overlap helps avoid losing context that falls on a chunk boundary.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but overlap with previous chunk

    return chunks
