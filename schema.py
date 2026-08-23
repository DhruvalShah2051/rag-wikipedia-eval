"""
Database schema for the chunk store.

Extracted from 2_setup_db.py so the same DDL can be reused - most importantly by
the pgvector integration tests, which build a real schema in CI. Duplicating the
DDL in the test suite would let the tested schema drift from the real one, which
is the one thing an integration test must never allow.
"""

from config import EMBEDDING_DIM

# Number of ivfflat partitions. Small, because the corpus is small (416 chunks).
# Exported so callers that need exhaustive, deterministic search - the tests -
# can set ivfflat.probes to match.
IVFFLAT_LISTS = 10


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


def create_index(cur):
    """
    Index for fast approximate nearest-neighbor search.
    ivfflat works well for small-to-medium datasets like this one.

    Note that ivfflat builds its partitions from the rows present when the index
    is created, so it is best created after the table has been populated.
    """
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {IVFFLAT_LISTS});
    """)


def create_schema(conn):
    """Full schema setup: extension, table, and index."""
    cur = conn.cursor()
    create_extension(cur)
    create_table(cur)
    create_index(cur)
    cur.close()
