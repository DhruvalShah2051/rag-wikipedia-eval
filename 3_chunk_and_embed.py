"""
Step 3: Chunk the fetched articles, embed each chunk, and store in Postgres.

Run this after 1_fetch_wikipedia.py and 2_setup_db.py.

The work itself lives in ingest.py, so the Phase 2 sweep can re-ingest at a
different chunk size without shelling out to this script.
"""

from ingest import ingest_corpus

if __name__ == "__main__":
    ingest_corpus()
