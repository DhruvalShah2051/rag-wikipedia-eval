"""
Unit tests for retrieval and prompt assembly.

Postgres and the embedding model are both faked here, so these run with no
database, no API key, and no model download. The same retrieval path is
exercised against a real pgvector instance in test_retrieval_integration.py.
"""

import pytest

import rag_pipeline
from conftest import FakeConnection, FakeGroqClient
from config import GROQ_MODEL, IVFFLAT_PROBES, TOP_K
from rag_pipeline import Generation, answer_question, build_prompt, generate_answer, retrieve_chunks


SAMPLE_ROWS = [
    ("Backpropagation", "Backpropagation computes gradients via the chain rule.", 0.12),
    ("Gradient descent", "Gradient descent steps along the negative gradient.", 0.31),
]


@pytest.fixture
def fake_db(monkeypatch, fake_encoder):
    """Point retrieve_chunks at a fake cursor and a fake encoder."""
    connection = FakeConnection(SAMPLE_ROWS)

    monkeypatch.setattr(rag_pipeline, "_get_embedding_model", lambda: fake_encoder)
    monkeypatch.setattr(rag_pipeline.psycopg2, "connect", lambda **kwargs: connection)

    return connection


def select_statement(connection):
    """
    The retrieval SELECT, found by content rather than by position.

    retrieve_chunks issues a session SET before the query, and more setup
    statements may appear later; a test that hardcodes executed[0] breaks on
    every such change without saying anything useful.
    """
    for sql, params in connection.cursor_obj.executed:
        if "SELECT" in sql.upper():
            return sql, params
    raise AssertionError("no SELECT was executed")


# --------------------------------------------------------------------------
# retrieve_chunks
# --------------------------------------------------------------------------


def test_retrieve_shapes_rows_into_dicts(fake_db):
    results = retrieve_chunks("what is backpropagation?")

    assert results == [
        {
            "source_title": "Backpropagation",
            "chunk_text": "Backpropagation computes gradients via the chain rule.",
            "distance": 0.12,
        },
        {
            "source_title": "Gradient descent",
            "chunk_text": "Gradient descent steps along the negative gradient.",
            "distance": 0.31,
        },
    ]


def test_retrieve_embeds_the_query(fake_db, fake_encoder):
    retrieve_chunks("what is backpropagation?")
    assert fake_encoder.encoded == ["what is backpropagation?"]


def test_retrieve_passes_top_k_to_the_query(fake_db):
    retrieve_chunks("anything", top_k=7)

    _sql, params = select_statement(fake_db)
    assert params[1] == 7


def test_retrieve_defaults_to_configured_top_k(fake_db):
    retrieve_chunks("anything")

    _sql, params = select_statement(fake_db)
    assert params[1] == TOP_K


def test_retrieve_sends_the_embedding_as_a_vector_parameter(fake_db, fake_encoder):
    """
    The embedding must be passed as a bound parameter cast to ::vector, never
    interpolated into the SQL string.
    """
    retrieve_chunks("anything")

    sql, params = select_statement(fake_db)
    assert "%s::vector" in sql
    assert params[0] == [0.0] * fake_encoder.dim


def test_retrieve_orders_by_cosine_distance_ascending(fake_db):
    """Lower pgvector cosine distance means more similar, so ASC is nearest-first."""
    retrieve_chunks("anything")

    sql, _params = select_statement(fake_db)
    assert "<=>" in sql
    assert "ORDER BY distance ASC" in sql


def test_retrieve_sets_ivfflat_probes_on_every_connection(fake_db):
    """
    Each call opens a fresh connection and ivfflat.probes resets to 1 on each
    one, so it has to be set per call rather than once at setup. Leaving it at
    the default over a badly sized index is what recorded a genuine 10/10
    retrieval as 90%.
    """
    retrieve_chunks("anything")

    probe_statements = [
        (sql, params) for sql, params in fake_db.cursor_obj.executed
        if "ivfflat.probes" in sql
    ]
    assert probe_statements, "ivfflat.probes was never set"

    _sql, params = probe_statements[0]
    assert params == (IVFFLAT_PROBES,)


def test_probes_are_set_before_the_query_runs(fake_db):
    """Setting probes after the SELECT would have no effect on it."""
    retrieve_chunks("anything")

    order = [sql for sql, _params in fake_db.cursor_obj.executed]
    probes_at = next(i for i, sql in enumerate(order) if "ivfflat.probes" in sql)
    select_at = next(i for i, sql in enumerate(order) if "SELECT" in sql.upper())

    assert probes_at < select_at


def test_retrieve_closes_cursor_and_connection(fake_db):
    """Every call opens its own connection; leaking them would exhaust Postgres."""
    retrieve_chunks("anything")

    assert fake_db.cursor_obj.closed is True
    assert fake_db.closed is True


# --------------------------------------------------------------------------
# build_prompt
# --------------------------------------------------------------------------


def test_prompt_includes_every_chunk_and_its_source():
    chunks = [
        {"source_title": "Overfitting", "chunk_text": "Fits noise in the training set."},
        {"source_title": "Neural network", "chunk_text": "Layers of connected units."},
    ]

    prompt = build_prompt("What is overfitting?", chunks)

    for chunk in chunks:
        assert chunk["chunk_text"] in prompt
        assert f"[Source: {chunk['source_title']}]" in prompt


def test_prompt_includes_the_question():
    prompt = build_prompt("What is overfitting?", [])
    assert "What is overfitting?" in prompt


def test_prompt_grounds_the_model_in_the_context():
    """
    The anti-hallucination instruction and the refusal string are load-bearing:
    without them the model answers from pretraining and the harness stops
    measuring retrieval at all.
    """
    prompt = build_prompt("q", [])

    assert "ONLY the context provided below" in prompt
    assert "I don't have enough information to answer that." in prompt
    assert "Do not use outside knowledge." in prompt


def test_prompt_demands_an_explained_answer():
    """
    This instruction is what moved answer accuracy off terse category-name
    replies. Rule 2 of the grading rubric marks those INCORRECT, so the two
    have to stay in step.
    """
    prompt = build_prompt("q", [])

    assert "Answer in 1-2 complete sentences" in prompt
    assert "category name" in prompt


def test_prompt_with_no_chunks_is_still_well_formed():
    prompt = build_prompt("What is overfitting?", [])

    assert "CONTEXT:" in prompt
    assert "QUESTION:" in prompt
    assert prompt.rstrip().endswith("ANSWER:")


# --------------------------------------------------------------------------
# generate_answer - token and latency capture
# --------------------------------------------------------------------------


def test_generation_carries_token_counts_from_the_response():
    """
    Token counts come from Groq's usage block, so they are reported rather than
    estimated. Phase 2 logs them to MLflow as a per-run metric.
    """
    client = FakeGroqClient("an answer", prompt_tokens=340, completion_tokens=42)

    generation = generate_answer("q", [], client=client)

    assert generation.answer == "an answer"
    assert generation.prompt_tokens == 340
    assert generation.completion_tokens == 42
    assert generation.total_tokens == 382


def test_generation_records_latency():
    client = FakeGroqClient("an answer")

    generation = generate_answer("q", [], client=client)

    assert generation.latency_s >= 0.0


def test_generation_uses_the_configured_model_and_temperature():
    client = FakeGroqClient("an answer")

    generate_answer("q", [], client=client)

    assert client.calls[0]["model"] == GROQ_MODEL
    assert client.calls[0]["temperature"] == 0.1


def test_answer_question_threads_generation_through(fake_db):
    """The evaluation harness reads cost off result['generation']."""
    client = FakeGroqClient("an answer", prompt_tokens=10, completion_tokens=5)

    result = answer_question("q", client=client)

    assert result["answer"] == "an answer"
    assert isinstance(result["generation"], Generation)
    assert result["generation"].total_tokens == 15
    assert len(result["retrieved_chunks"]) == len(SAMPLE_ROWS)
