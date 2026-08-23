"""
Core RAG pipeline: given a question, retrieve relevant chunks from Postgres
and generate a grounded answer using Groq.

This module is imported by both 4_query.py (interactive use) and
5_evaluate.py (automated evaluation), so the retrieval/generation logic
lives in one place.
"""

from dataclasses import dataclass

import psycopg2
from sentence_transformers import SentenceTransformer
from groq import Groq
from config import (
    DB_CONFIG,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    IVFFLAT_PROBES,
    TOP_K,
)
from llm import chat_completion


@dataclass
class Generation:
    """
    One generated answer and what it cost.

    Groq reports token counts on every response, so none of this is estimated.
    Phase 2 logs the token and latency fields to MLflow as run metrics.
    """

    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float

# Both are built on first use and then cached, so repeated calls (e.g. during
# evaluation) stay fast without paying the cost at import time. Importing this
# module must not download a 90MB model or require an API key - tests, tooling,
# and CI all import it, and the Groq constructor raises when the key is absent.
_embedding_model = None
_groq_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def retrieve_chunks(query, top_k=TOP_K):
    """
    Embed the query and find the top_k most similar chunks in Postgres
    using pgvector's cosine distance operator (<=>).
    """
    query_embedding = _get_embedding_model().encode(query).tolist()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Every call opens a fresh connection, and ivfflat.probes resets to 1 on each
    # one. Setting it explicitly is what makes retrieval reproducible: leaving it
    # at the default over a badly sized index is what turned a genuine 10/10 into
    # a recorded 90%. Logged to MLflow so each result is attributable.
    cur.execute("SET ivfflat.probes = %s;", (IVFFLAT_PROBES,))

    cur.execute(
        """
        SELECT source_title, chunk_text, embedding <=> %s::vector AS distance
        FROM document_chunks
        ORDER BY distance ASC
        LIMIT %s;
        """,
        (query_embedding, top_k),
    )

    results = cur.fetchall()
    cur.close()
    conn.close()

    # Lower distance = more similar. Return as list of dicts for readability.
    return [
        {"source_title": row[0], "chunk_text": row[1], "distance": row[2]}
        for row in results
    ]


def build_prompt(query, retrieved_chunks):
    """
    Context engineering step: assemble the retrieved chunks and the user's
    question into a structured prompt that instructs the model to answer
    ONLY from the provided context (reduces hallucination).
    """
    context_block = "\n\n".join(
        f"[Source: {c['source_title']}]\n{c['chunk_text']}" for c in retrieved_chunks
    )

    prompt = f"""You are a helpful assistant answering questions using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not use outside knowledge.

Answer in 1-2 complete sentences. Do not respond with just a category name, label, or single term -
briefly explain the mechanism, reasoning, or "how/why" behind your answer, even if the question sounds
like it wants a short answer.

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""

    return prompt


def generate_answer(query, retrieved_chunks, client=None):
    """
    Send the assembled prompt to Groq and return a Generation.

    `client` is injectable so tests can generate against a stub without a
    network call or an API key.
    """
    prompt = build_prompt(query, retrieved_chunks)

    response, latency_s = chat_completion(
        client or _get_groq_client(),
        model=GROQ_MODEL,
        prompt=prompt,
        temperature=0.1,  # low temperature for factual, grounded answers
    )

    usage = response.usage
    return Generation(
        answer=response.choices[0].message.content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_s=latency_s,
    )


def answer_question(query, top_k=TOP_K, verbose=False, client=None):
    """
    Full end-to-end pipeline: retrieve -> build prompt -> generate.
    Returns the answer, the retrieved chunks, and what the call cost.
    """
    retrieved = retrieve_chunks(query, top_k=top_k)
    generation = generate_answer(query, retrieved, client=client)

    if verbose:
        print(f"\nQuery: {query}")
        print("\nRetrieved chunks:")
        for c in retrieved:
            print(f"  - [{c['source_title']}] (distance={c['distance']:.4f}) {c['chunk_text'][:100]}...")
        print(f"\nAnswer: {generation.answer}")
        print(f"({generation.total_tokens} tokens, {generation.latency_s:.2f}s)")

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "answer": generation.answer,
        "generation": generation,
    }


if __name__ == "__main__":
    # Quick manual test
    result = answer_question("What is backpropagation used for?", verbose=True)