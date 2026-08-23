"""
Shared fixtures and fakes for the test suite.

Two tiers of test live here:

  * unit tests, which never touch a database, a network, or an API key. These
    run everywhere, including CI, and are the default.
  * integration tests, marked `@pytest.mark.integration`, which need a real
    Postgres with pgvector. They skip themselves when no database is reachable,
    so plain `pytest` is safe to run on any machine.
"""

import os

import psycopg2
import pytest

from config import DB_CONFIG, EMBEDDING_DIM, EMBEDDING_MODEL_NAME

# Postgres schema the integration tests build their tables in. Deliberately not
# `public`: developers run this suite against the same local database that holds
# the real 416-chunk corpus, and the tests must not be able to touch it.
TEST_SCHEMA = "rag_test"


# --------------------------------------------------------------------------
# Unit-test fakes
# --------------------------------------------------------------------------


class FakeCursor:
    """
    Stands in for a psycopg2 cursor. Records the SQL and parameters it was given
    so tests can assert on what the pipeline actually sent to Postgres.
    """

    def __init__(self, rows):
        self.rows = rows
        self.executed = []  # list of (sql, params)
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)
        self.closed = False
        self.committed = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed += 1

    def close(self):
        self.closed = True


class FakeEncoder:
    """
    Stands in for a SentenceTransformer. Returns a fixed-length vector of the
    right dimension without loading 90MB of weights.
    """

    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.encoded = []

    def encode(self, text, **kwargs):
        self.encoded.append(text)
        if isinstance(text, list):
            return [_FakeVector(self.dim) for _ in text]
        return _FakeVector(self.dim)


class _FakeVector:
    """Just enough of a numpy array for the pipeline: `.tolist()`."""

    def __init__(self, dim):
        self.dim = dim

    def tolist(self):
        return [0.0] * self.dim


class FakeGroqResponse:
    """
    Minimal shape of a Groq chat completion: .choices[0].message.content plus
    the .usage block the pipeline reads token counts from.
    """

    def __init__(self, content, prompt_tokens=100, completion_tokens=25):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )()


class FakeGroqClient:
    """
    Stands in for a Groq client. Returns a canned reply and records the request,
    so generation and grading can be tested without a network call or an API key.

    `errors` is a list of exceptions to raise before the reply finally succeeds,
    which is how the retry path is exercised.
    """

    def __init__(self, reply, prompt_tokens=100, completion_tokens=25, errors=None):
        self.reply = reply
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.errors = list(errors or [])
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return FakeGroqResponse(self.reply, self.prompt_tokens, self.completion_tokens)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


# --------------------------------------------------------------------------
# Integration fixtures
# --------------------------------------------------------------------------

# Three deliberately unrelated topics. Retrieval assertions are only meaningful
# if the correct source is unambiguously nearest - near-synonymous ML topics
# would make a passing test say more about luck than about retrieval.
FIXTURE_DOCS = {
    "Photosynthesis": [
        "Photosynthesis is the process by which green plants use sunlight to "
        "synthesise nutrients from carbon dioxide and water.",
        "Chlorophyll in the chloroplasts absorbs light energy and converts it "
        "into chemical energy stored as glucose.",
        "The light-dependent reactions release oxygen as a by-product of "
        "splitting water molecules.",
    ],
    "Volcano": [
        "A volcano is a rupture in the crust of a planet through which lava, "
        "ash and gases escape from a magma chamber below the surface.",
        "Volcanic eruptions are classified by their explosivity, which depends "
        "on the viscosity of the magma and its dissolved gas content.",
        "Shield volcanoes form broad, gently sloping cones built from repeated "
        "flows of low-viscosity basaltic lava.",
    ],
    "Backpropagation": [
        "Backpropagation computes the gradient of a loss function with respect "
        "to each weight in a neural network by applying the chain rule.",
        "Gradients flow backwards from the output layer to the input layer, "
        "which is what gives the algorithm its name.",
        "The computed gradients are then used by an optimiser such as gradient "
        "descent to update the network's parameters.",
    ],
}


@pytest.fixture(scope="session")
def embedding_model():
    """The real embedding model. Loaded once per session - it is slow to build."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="session")
def pg_connection():
    """
    A connection to the test database, or a skip if none is reachable.

    This is what keeps `pytest` with no arguments safe to run on a laptop with
    no Postgres running.

    Skipping is the wrong behaviour in CI, though: a service container that
    failed to start would turn every integration test into a skip and the job
    would still report success. Setting RAG_REQUIRE_DB turns the skip into a
    failure, so the CI job can only pass by actually reaching a database.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.Error as exc:
        message = f"no Postgres reachable at {DB_CONFIG['host']}:{DB_CONFIG['port']} ({exc})"
        if os.getenv("RAG_REQUIRE_DB"):
            pytest.fail(f"RAG_REQUIRE_DB is set but {message}", pytrace=False)
        pytest.skip(message)

    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture
def test_schema(pg_connection):
    """
    An empty schema named by TEST_SCHEMA, dropped again afterwards.

    pgvector's extension is created in `public` and left there; the test tables
    are created inside TEST_SCHEMA, and the search path resolves the `vector`
    type from public.
    """
    cur = pg_connection.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
    cur.execute(f"CREATE SCHEMA {TEST_SCHEMA};")
    cur.execute(f"SET search_path TO {TEST_SCHEMA}, public;")

    yield pg_connection

    cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE;")
    cur.execute("SET search_path TO public;")
    cur.close()


@pytest.fixture
def seeded_db(test_schema, embedding_model, monkeypatch):
    """
    A populated `document_chunks` table inside the test schema, with the
    pipeline's own connections pointed at that schema.

    retrieve_chunks() opens its own connection from DB_CONFIG, so the search
    path has to be set as a libpq connection option rather than on the fixture's
    session - otherwise the pipeline would look in `public` and find the real
    corpus, or nothing at all.

    No ivfflat index is created here. ivfflat is an approximate index whose
    recall depends on `lists`, `probes`, and corpus size; asserting an exact
    ranking through it on a nine-row table would be flaky by construction.
    Exact search gives the retrieval assertions a deterministic answer, and
    test_schema.py covers the indexed DDL separately.
    """
    import rag_pipeline
    from schema import create_table

    monkeypatch.setitem(
        rag_pipeline.DB_CONFIG, "options", f"-c search_path={TEST_SCHEMA},public"
    )

    cur = test_schema.cursor()
    create_table(cur)

    for title, chunks in FIXTURE_DOCS.items():
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO document_chunks (source_title, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (title, idx, chunk, embedding.tolist()),
            )

    cur.close()
    yield test_schema
