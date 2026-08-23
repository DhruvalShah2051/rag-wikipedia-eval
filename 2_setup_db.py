"""
Step 2: Set up the Postgres database schema.

Requires:
  - Postgres running locally
  - pgvector extension installed (see README for install instructions)
  - A database created matching config.DB_CONFIG['dbname']

Run this once before embedding/storing any chunks.
"""

import psycopg2
from config import DB_CONFIG
from schema import create_schema


def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    # The DDL itself lives in schema.py so the integration tests can build the
    # exact same schema without duplicating it.
    create_schema(conn)

    print("Database schema ready: 'document_chunks' table created with pgvector index.")

    conn.close()


if __name__ == "__main__":
    setup_database()
