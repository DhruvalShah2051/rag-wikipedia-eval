"""
Database schema for the chunk store.

Extracted from 2_setup_db.py so the same DDL can be reused - most importantly by
the pgvector integration tests, which build a real schema in CI. Duplicating the
DDL in the test suite would let the tested schema drift from the real one, which
is the one thing an integration test must never allow.

The index sizing here is not cosmetic. ivfflat is an approximate index: it
partitions vectors into `lists` and scans only `probes` of them per query, so a
badly sized index silently returns the wrong nearest neighbours instead of
failing. Phase 2 measured exactly that - a fixed `lists = 10` over ~414 chunks
scanned a tenth of the corpus per query and dropped the correct chunk on 2 of
10 benchmark questions, which had been recorded as a 90% retrieval score.
"""

import math

from config import EMBEDDING_DIM

# pgvector's own sizing guidance: lists = rows / 1000 for corpora up to a
# million rows. At this project's scale that yields a single list, which means
# the index degenerates to an exhaustive scan - the correct behaviour for 414
# chunks, and it grows into a real approximate index automatically if the corpus
# ever does.
ROWS_PER_LIST = 1000


def ivfflat_lists(row_count):
    """Number of ivfflat partitions appropriate for a corpus of `row_count` rows."""
    return max(1, row_count // ROWS_PER_LIST)


def ivfflat_probes(lists):
    """
    Partitions to scan per query. sqrt(lists) is pgvector's suggested starting
    point; probes == lists would be exhaustive.
    """
    return max(1, round(math.sqrt(lists)))


def create_extension(cur):
    """Enable pgvector. The extension must already be installed on the server."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def create_table(cur):
    """Create the table that stores document chunks and their embeddings."""
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            source_title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM})
        );
    """)


def create_index(cur, lists=None):
    """
    Index for approximate nearest-neighbor search.

    ivfflat builds its partitions from the rows present when the index is
    created, so this must run *after* the table is populated. An index built on
    rows that were later truncated describes centroids for data that no longer
    exists, and recall collapses without any error being raised.
    """
    if lists is None:
        cur.execute("SELECT count(*) FROM document_chunks;")
        lists = ivfflat_lists(cur.fetchone()[0])

    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {lists});
    """)
    return lists


def rebuild_index(cur):
    """
    Drop and recreate the index against whatever is currently in the table.

    Called after every ingestion. Re-ingesting truncates and reloads, which
    leaves any existing ivfflat index trained on rows that are gone.
    """
    cur.execute("DROP INDEX IF EXISTS document_chunks_embedding_idx;")
    return create_index(cur)


def create_schema(conn):
    """Full schema setup: extension, table, and index."""
    cur = conn.cursor()
    create_extension(cur)
    create_table(cur)
    create_index(cur)
    cur.close()
