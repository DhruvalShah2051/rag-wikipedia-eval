"""
Retrieval against a real Postgres with pgvector.

The unit tests fake the database, which proves the pipeline sends the right
query but says nothing about whether pgvector answers it correctly. These tests
seed a small fixture corpus, run the real embedding model, and check that
semantic search actually returns the right article.

They skip themselves when no database is reachable. In CI they run against a
`pgvector/pgvector:pg16` service container; locally they build their tables in a
dedicated `rag_test` schema so the real corpus is never touched.
"""

import pytest

from rag_pipeline import retrieve_chunks
from schema import IVFFLAT_LISTS, create_index, create_table
from conftest import FIXTURE_DOCS, TEST_SCHEMA

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("How do plants turn sunlight into chemical energy?", "Photosynthesis"),
        ("What makes an eruption explosive?", "Volcano"),
        ("How are gradients computed for a neural network's weights?", "Backpropagation"),
    ],
)
def test_retrieval_returns_the_expected_source(seeded_db, query, expected_source):
    """The nearest chunk should come from the article that actually answers it."""
    results = retrieve_chunks(query, top_k=3)

    assert results, "retrieval returned nothing"
    assert results[0]["source_title"] == expected_source


def test_retrieval_respects_top_k(seeded_db):
    for k in (1, 3, 5):
        assert len(retrieve_chunks("photosynthesis", top_k=k)) == k


def test_results_are_ordered_nearest_first(seeded_db):
    results = retrieve_chunks("How do plants turn sunlight into chemical energy?", top_k=5)

    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_retrieved_chunks_match_the_seeded_text(seeded_db):
    """Guards the row -> dict mapping against a column-order change in the SQL."""
    results = retrieve_chunks("How do plants turn sunlight into chemical energy?", top_k=3)

    for result in results:
        assert result["source_title"] in FIXTURE_DOCS
        assert result["chunk_text"] in FIXTURE_DOCS[result["source_title"]]


def test_distances_are_in_cosine_range(seeded_db):
    """pgvector's <=> operator returns cosine distance, which lies in [0, 2]."""
    results = retrieve_chunks("volcano", top_k=5)

    assert all(0.0 <= r["distance"] <= 2.0 for r in results)


def test_schema_ddl_applies_against_a_real_server(test_schema):
    """
    Runs the same DDL that 2_setup_db.py runs, including the ivfflat index that
    the retrieval fixtures deliberately skip. This is what catches a schema or
    index definition that only fails on a real Postgres - a wrong vector
    dimension, or an operator class pgvector does not have.
    """
    cur = test_schema.cursor()
    create_table(cur)
    create_index(cur)

    cur.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND indexname = %s;",
        (TEST_SCHEMA, "document_chunks_embedding_idx"),
    )
    row = cur.fetchone()
    cur.close()

    assert row is not None, "ivfflat index was not created"
    indexdef = row[0]
    assert "ivfflat" in indexdef
    assert "vector_cosine_ops" in indexdef
    assert f"lists='{IVFFLAT_LISTS}'" in indexdef.replace('"', "'")
