"""
Step 2: Set up the Postgres database schema.

Requires:
  - Postgres running locally
  - pgvector extension installed (see README for install instructions)
  - A database created matching config.DB_CONFIG['dbname']

Run this once before embedding/storing any chunks.
"""

import psycopg2
from config import DB_CONFIG, EMBEDDING_DIM


def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Enable the pgvector extension (must be installed on the Postgres server first)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create the table that stores document chunks and their embeddings
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            source_title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding VECTOR({EMBEDDING_DIM})
        );
    """)

    # Index for fast approximate nearest-neighbor search.
    # ivfflat works well for small-to-medium datasets like this one.
    cur.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)

    print("Database schema ready: 'document_chunks' table created with pgvector index.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    setup_database()
