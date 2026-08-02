"""
Step 3: Chunk the fetched articles, embed each chunk, and store in Postgres.

Run this after 1_fetch_wikipedia.py and 2_setup_db.py.
"""

import os
import glob
import psycopg2
from sentence_transformers import SentenceTransformer
from config import (
    DB_CONFIG,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)

RAW_DATA_DIR = "data/raw"


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


def process_and_store_all():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Clear any existing chunks so re-running this script doesn't duplicate data
    cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
    conn.commit()

    txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
    total_chunks = 0

    for filepath in txt_files:
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        if not chunks:
            continue

        # Embed all chunks for this article in one batch (faster than one-by-one)
        embeddings = model.encode(chunks, show_progress_bar=False)

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO document_chunks (source_title, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (title, idx, chunk, embedding.tolist()),
            )

        conn.commit()
        total_chunks += len(chunks)
        print(f"[OK] '{title}': {len(chunks)} chunks embedded and stored.")

    cur.close()
    conn.close()

    print(f"\nDone. Stored {total_chunks} chunks from {len(txt_files)} articles.")


if __name__ == "__main__":
    process_and_store_all()
