"""
Ingestion: chunk the fetched articles, embed each chunk, store in Postgres.

Extracted from 3_chunk_and_embed.py so the Phase 2 sweep can re-ingest the
corpus at a different chunk size between evaluation runs. That script begins
with a digit and cannot be imported.
"""

import glob
import os

import psycopg2
from sentence_transformers import SentenceTransformer

from chunking import chunk_text
from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    DB_CONFIG,
    EMBEDDING_MODEL_NAME,
)
from schema import ivfflat_probes, rebuild_index

RAW_DATA_DIR = "data/raw"

# Loaded on first use and cached, matching rag_pipeline. Re-ingesting six times
# during a sweep should not reload the weights six times.
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def ingest_corpus(chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS, verbose=True):
    """
    Re-chunk, re-embed, and re-store every article in RAW_DATA_DIR.

    Returns the total number of chunks stored, which the sweep logs to MLflow -
    chunk count is the most direct consequence of changing the chunk size, and
    logging it makes the corpus behind each run identifiable after the fact.
    """
    model = _get_embedding_model()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Clear existing chunks so re-running never duplicates data. This is why the
    # sweep must restore the baseline when it finishes: whatever ran last is
    # what stays in the database.
    cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
    conn.commit()

    txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
    if not txt_files:
        cur.close()
        conn.close()
        raise FileNotFoundError(
            f"No articles in {RAW_DATA_DIR}. Run 1_fetch_wikipedia.py first."
        )

    total_chunks = 0

    for filepath in txt_files:
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
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
        if verbose:
            print(f"[OK] '{title}': {len(chunks)} chunks embedded and stored.")

    # ivfflat builds its partitions from the rows present at creation time, and
    # the TRUNCATE above invalidated whatever index existed. Rebuilding is not
    # optional: a stale index does not error, it just quietly returns the wrong
    # nearest neighbours. This was measured as a 10-point retrieval loss.
    lists = rebuild_index(cur)
    conn.commit()

    cur.close()
    conn.close()

    if verbose:
        print(f"\nDone. Stored {total_chunks} chunks from {len(txt_files)} articles.")
        print(f"Rebuilt ivfflat index with lists={lists}, "
              f"queried at probes={ivfflat_probes(lists)}.")

    return total_chunks


def count_chunks():
    """Number of chunks currently in the store, without re-ingesting."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM document_chunks;")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count
